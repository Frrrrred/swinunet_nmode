from configs.config_setting import setting_config
import numpy as np
from tqdm import tqdm
import torch
from thop import profile
from sklearn.metrics import confusion_matrix
from utils import save_imgs


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer,
                    delta_optimizer,
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    # switch to train mode
    model.train()

    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        delta_optimizer.zero_grad()

        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        if 'ege' in setting_config.network:
            gt_pre, out = model(images)
            loss = criterion(gt_pre, out, targets)
        else:
            out = model(images)
            loss = criterion(out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('train_loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = (f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr},'
                        f' delta: {model.delta.data}')
            print(log_info)
            logger.info(log_info)
    scheduler.step(loss)
    return step


# def train_one_epoch(train_loader,
#                     model,
#                     criterion,
#                     optimizer,
#                     scheduler,
#                     epoch,
#                     step,
#                     logger,
#                     config,
#                     writer):
#     # switch to train mode
#     model.train()
#
#     delta_lr = 0.0001
#     delta_momentum = 0.9  # 动量因子
#     delta_update = 0.0  # 动量累积
#     loss_list = []
#
#     for iter, data in enumerate(train_loader):
#         step += iter
#         optimizer.zero_grad()
#
#         images, targets = data
#         images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
#
#         if 'ege' in setting_config.network:
#             gt_pre, out = model(images)
#             loss = criterion(gt_pre, out, targets)
#         else:
#             out = model(images)
#             loss = criterion(out, targets)
#
#         loss.backward()
#         optimizer.step()
#
#         loss_list.append(loss.item())
#         now_lr = optimizer.state_dict()['param_groups'][0]['lr']
#         writer.add_scalar('train_loss', loss, global_step=step)
#
#         if iter % config.print_interval == 0:
#             log_info = (f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr},'
#                         f' delta: {model.delta.data}')
#             print(log_info)
#             logger.info(log_info)
#             with torch.no_grad():
#                 delta_grad = model.delta.grad
#                 # 动量更新
#                 delta_update = delta_momentum * delta_update + (1 - delta_momentum) * delta_grad
#                 # 根据梯度大小调整动态调整学习率
#                 adaptive_lr = delta_lr / (1 + torch.norm(delta_grad))
#                 model.delta -= adaptive_lr * delta_update
#                 model.delta.data.clamp_(0, 1)  # 保证delta在[0, 1]之间
#     scheduler.step(loss)
#     return step


def val_one_epoch(test_loader,
                  model,
                  criterion,
                  delta_optimizer,
                  epoch,
                  logger,
                  config):
    # switch to evaluate mode
    model.eval()

    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            if 'ege' in setting_config.network:
                gt_pre, out = model(img)
                loss = criterion(gt_pre, out, msk)
            else:
                out = model(img)
                loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = (f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc},'
                    f' accuracy: {accuracy}, specificity: {specificity}, sensitivity: {sensitivity},'
                    f' confusion_matrix: {confusion}')
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info,config.study_name)
        logger.info(log_info)

    if epoch % 20 == 0:
        delta_optimizer.step()
        model.delta.data.clamp_(0, 1)  # 保证delta在[0, 1]之间
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    # switch to evaluate mode
    model.eval()

    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            if 'ege' in setting_config.network:
                gt_pre, out = model(img)
                loss = criterion(gt_pre, out, msk)
            else:
                out = model(img)
                loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                          test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        dummy_input = torch.randn(8, 3, 256, 256).cuda()
        flops, params = profile(model, (dummy_input,))

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = (f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc},'
                    f' accuracy: {accuracy}, specificity: {specificity}, sensitivity: {sensitivity},'
                    f' confusion_matrix: {confusion},flops: {flops},params: {params}')
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list),miou
