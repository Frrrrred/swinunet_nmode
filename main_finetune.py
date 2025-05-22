import gc
import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from contextlib import contextmanager
# import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader, generate_images_for_dataset_train
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from apex import amp


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    # 已固定参数值，原require=True
    parser.add_argument('--cfg', type=str,
                        default='configs/swinUnet_nmODE_finetune__swin_base__img512_window8__100ep.yaml',
                        metavar="FILE", help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--dataset-path', type=str, help='path to dataset')
    parser.add_argument('--pretrained', type=str, help='path to pre-trained model')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true', default=False,
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args = parser.parse_args()
    config = get_config(args)

    return args, config


class MixedSoftTargetFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, beta=1.0):
        super(MixedSoftTargetFocalLoss, self).__init__()
        assert alpha < 1.0
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta  # 用于控制误分类样本的权重

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_one_hot = F.one_hot(target, num_classes=x.size(-1)).float()

        # SoftTargetCrossEntropy
        log_probs = F.log_softmax(x, dim=-1)
        soft_target_loss = torch.sum(-target_one_hot * log_probs, dim=-1).mean()

        # Focal Loss
        probs = F.softmax(x, dim=-1)
        focal_loss = torch.sum(-target_one_hot * (1 - probs) ** self.gamma * log_probs, dim=-1).mean()

        # 计算误分类样本的权重
        pred_probs, pred_labels = torch.max(probs, dim=-1)
        true_probs = torch.sum(probs * target_one_hot, dim=-1)
        misclassification_weight = torch.exp(self.beta * (pred_probs - true_probs))

        # MixedSoftTargetFocalLoss
        loss = self.alpha * soft_target_loss + (1 - self.alpha) * focal_loss
        loss = loss * misclassification_weight.mean()
        return loss


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


@contextmanager
def pretrain_model(config, logger):
    model_pretrain = build_model(config, is_pretrain=True)
    load_pretrained(config, model_pretrain, logger, is_diffent=False)
    try:
        yield model_pretrain
    finally:
        del model_pretrain
        gc.collect()
        torch.cuda.empty_cache()


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_train_mask, data_loader_val, data_loader_val_mask, mixup_fn = build_loader(config, logger, is_pretrain=False)

    assert len(data_loader_train) == len(data_loader_train_mask), "data_loader_train and data_loader_train_mask should have the same length"
    assert len(data_loader_val) == len(data_loader_val_mask), "data_loader_val and data_loader_val_mask should have the same length"

    # with pretrain_model(config, logger) as model_pretrain:
    #     data_loader_train = generate_images_for_dataset_train(config, model_pretrain, data_loader_train, logger)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=False)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=False)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # 用于分类头的损失函数
    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    #     # criterion = SoftTargetCrossEntropy()
    #     criterion = MixedSoftTargetFocalLoss()
    # elif config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()
    # 用于分割头的损失函数
    # criterion = BceDiceLoss()
    criterion = torch.nn.BCEWithLogitsLoss()

    max_accuracy = 0.0
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        _, max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, data_loader_val_mask, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} busi images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return
    elif config.PRETRAINED:
        load_pretrained(config, model, logger, is_diffent=True)

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    now_saves = []
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)
        data_loader_train_mask.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, data_loader_train_mask, optimizer, epoch, mixup_fn, lr_scheduler)

        acc1, acc5, loss = validate(config, data_loader_val, data_loader_val_mask, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} busi images: {acc1:.1f}%")
        if (epoch == 0 or (acc1 > max_accuracy)):
            max_accuracy = max(max_accuracy, acc1)
            save_checkpoint(config, epoch, model, max_accuracy, 0., optimizer, lr_scheduler, logger, now_saves)
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, data_mask_loader, optimizer, epoch, mixup_fn, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    logger.info(f'Current learning rate for different parameter groups: {[it["lr"] for it in optimizer.param_groups]}')

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    # 用于分割头
    data_mask_loader = iter(data_mask_loader)
    for idx, (samples, _) in enumerate(data_loader):  # idx, (samples, targets)用于分类，idx, (samples, _)用于分割
        samples = samples.cuda(non_blocking=True)
        # 用于分类头
        # targets = targets.cuda(non_blocking=True)
        # 用于分割头
        targets, *_ = next(data_mask_loader)
        targets = targets.cuda(non_blocking=True)

        # outputs = model(samples, mask_samples)
        outputs = model(samples, None)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[-1]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, data_loader_mask, model):
    # 用于分类头
    # criterion = torch.nn.CrossEntropyLoss()
    # 用于分割头
    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    # acc1_meter = AverageMeter()
    # acc5_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()

    def calculate_metrics(output, target):
        # 将sigmoid输出转为二值化mask
        pred = torch.sigmoid(output)  # 适用于BCEWithLogitsLoss
        pred_mask = (pred > 0.5).float()

        # 计算Dice系数
        smooth = 1e-6
        intersection = (pred_mask * target).sum(dim=(1,2))
        union = pred_mask.sum(dim=(1,2)) + target.sum(dim=(1,2))
        dice = (2. * intersection + smooth) / (union + smooth)

        # 计算IoU
        iou = (intersection + smooth) / (union - intersection + smooth)

        return dice.mean(), iou.mean()

    end = time.time()
    # 用于分割头
    data_loader_mask = iter(data_loader_mask)
    for idx, (images, _) in enumerate(data_loader):  # idx, (samples, targets)用于分类，idx, (samples, _)用于分割
        images = images.cuda(non_blocking=True)
        # 用于分类头
        # target = target.cuda(non_blocking=True)
        # 用于分割头
        target, *_ = next(data_loader_mask)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        dice, iou = calculate_metrics(output.squeeze(1), target.squeeze(1))
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # acc1 = reduce_tensor(acc1)
        # acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)
        dice = reduce_tensor(dice)
        iou = reduce_tensor(iou)

        loss_meter.update(loss.item(), target.size(0))
        # acc1_meter.update(acc1.item(), target.size(0))
        # acc5_meter.update(acc5.item(), target.size(0))
        dice_meter.update(dice.item(), target.size(0))
        iou_meter.update(iou.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            # logger.info(
            #     f'Test: [{idx}/{len(data_loader)}]\t'
            #     f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #     f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
            #     f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
            #     f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
            #     f'Mem {memory_used:.0f}MB')
            logger.info(
                 f'Test: [{idx}/{len(data_loader)}]\t'
                 f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                 f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                 f'Dice {dice_meter.val:.3f} ({dice_meter.avg:.3f})\t'
                 f'IoU {iou_meter.val:.3f} ({iou_meter.avg:.3f})\t'
                 f'Mem {memory_used:.0f}MB')

    # logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    logger.info(f' * Dice {dice_meter.avg:.3f} IoU {iou_meter.avg:.3f}')
    # return acc1_meter.avg, acc5_meter.avg, loss_meter.avg
    return dice_meter.avg, iou_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)

    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print conf
    logger.info(config.dump())

    main(config)
