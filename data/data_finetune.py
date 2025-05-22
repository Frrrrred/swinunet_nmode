import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
# import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup, create_transform
# from timm.data.transforms import _pil_interpolation_to_str as _pil_interp
from torchvision.transforms import InterpolationMode


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        random.seed()
        np.random.seed()
        new_seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
        torch.manual_seed(new_seed)
        torch.cuda.manual_seed_all(new_seed)


def _pil_interp(interpolation):
    if interpolation == 'bicubic':
        return InterpolationMode.BICUBIC
    elif interpolation == 'bilinear':
        return InterpolationMode.BILINEAR
    elif interpolation == 'nearest':
        return InterpolationMode.NEAREST
    else:
        raise ValueError(f"Unsupported interpolation: {interpolation}")


def build_loader_finetune(config, logger):
    # 用于分类头
    # config.defrost()
    # dataset_train, _, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config, logger=logger)
    # config.freeze()
    # dataset_val, _, _ = build_dataset(is_train=False, config=config, logger=logger)

    # 用于分割头
    dataset_train, dataset_train_mask, _ = build_dataset(is_train=True, config=config, logger=logger)
    dataset_val, dataset_val_mask, _ = build_dataset(is_train=False, config=config, logger=logger)
    logger.info(f"Build dataset: train images = {len(dataset_train)}, val images = {len(dataset_val)}")

    num_tasks = 1
    global_rank = 0
    sampler_train = DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_train_mask = DataLoader(
        dataset_train_mask, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    data_loader_val_mask = DataLoader(
        dataset_val_mask, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    # setup mixup / cutmix
    # 用于分类头
    # mixup_fn = None
    # mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    # if mixup_active:
    #     mixup_fn = Mixup(
    #         mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
    #         prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
    #         label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
    # 用于分割头
    mixup_fn = None

    return dataset_train, dataset_val, data_loader_train, data_loader_train_mask, data_loader_val, data_loader_val_mask, mixup_fn


def build_dataset(is_train, config, logger):
    transform = build_transform(is_train, config)
    logger.info(f'Fine-tune data transform, is_train={is_train}:\n{transform}')

    prefix = 'train' if is_train else 'val'
    root = os.path.join(config.DATA.DATA_PATH, prefix)
    rand_int = random.randint(0, 2 ** 32 - 1)
    set_seed(rand_int)
    dataset = datasets.ImageFolder(root, transform=transform)
    nb_classes = config.MODEL.NUM_CLASSES

    prefix_mask = 'train' if is_train else 'val'
    mask_root = os.path.join(os.path.join(config.DATA.DATA_PATH, 'mask'), prefix_mask)
    set_seed(rand_int)
    mask_dataset = datasets.ImageFolder(mask_root, transform=transform)

    set_seed(None)
    return dataset, mask_dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            # size = int((256 / 224) * config.DATA.IMG_SIZE)
            size = int(config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.RandomHorizontalFlip())
    t.append(transforms.RandomVerticalFlip())
    t.append(transforms.RandomRotation(90))
    t.append(transforms.ToTensor())
    # t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


class VAEUNet(nn.Module):
    def __init__(self, config, model, latent_dim=512, num_classes=3):
        super(VAEUNet, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embed_dim = config.MODEL.SWIN.EMBED_DIM
        self.depths = len(config.MODEL.SWIN.DEPTHS) + 1
        self.embed_size = config.DATA.IMG_SIZE // 4

        self.encoder = model.encoder

        self.fc_mu = nn.Linear(self.embed_dim * self.embed_size ** 2 // 16, latent_dim // 16)
        self.fc_download = nn.ModuleList(
            [nn.Linear(self.embed_dim * (self.embed_size ** 2) // 2 ** i, latent_dim // 2 ** i) for i in
             range(self.depths)])
        self.fc_logvar = nn.Linear(self.embed_dim * self.embed_size ** 2 // 16, latent_dim // 16)
        self.fc_logvar_download = nn.ModuleList(
            [nn.Linear(self.embed_dim * (self.embed_size ** 2) // 2 ** i, latent_dim // 2 ** i) for i in
             range(self.depths)])

        self.decoder_fc = nn.Linear(latent_dim // 16 + num_classes, self.embed_dim * self.embed_size ** 2 // 16)
        self.decoder_fc_download = nn.ModuleList(
            [nn.Linear(latent_dim // 2 ** i + num_classes, self.embed_dim * (self.embed_size ** 2) // (2 ** i)) for i in
             range(self.depths)])

        self.decoder = model.decoder
        self.out = nn.Conv2d(in_channels=self.decoder.out_channels, out_channels=3, kernel_size=1)

    def encode(self, x):
        h, h_download = self.encoder(x, None, None)
        h = h.view(h.size(0), -1)
        h_download = [h_download[idx].view(h_download[idx].size(0), -1) for idx in range(self.depths)]

        mu = self.fc_mu(h)
        mu_download = [fc(h_download[idx]) for idx, fc in enumerate(self.fc_download)]

        logvar = self.fc_logvar(h)
        logvar_download = [fc(h_download[idx]) for idx, fc in enumerate(self.fc_logvar_download)]
        return mu, mu_download, logvar, logvar_download

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, z_download, labels):
        # 将标签信息与潜在变量结合
        z = torch.cat([z, labels], dim=1)
        z_download = [torch.cat([z_download[idx], labels], dim=1) for idx in range(self.depths)]

        h = self.decoder_fc(z)
        h = h.view(-1, self.embed_size // 16 * self.embed_size // 16, self.embed_dim * 16)

        h_download = [self.decoder_fc_download[idx](z_download[idx]) for idx in range(self.depths)]
        h_download = [h_download[idx].view(-1, self.embed_size // 2 ** idx * self.embed_size // 2 ** idx,
                                           self.embed_dim * 2 ** idx)
                      for idx in range(self.depths)]

        y = self.decoder(h, h_download)
        y = self.out(y)
        return y

    def forward(self, x, labels):
        x = x.cuda()
        labels = labels.cuda()
        mu, mu_download, logvar, logvar_download = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z_download = [self.reparameterize(mu_download[idx], logvar_download[idx]) for idx in range(self.depths)]
        recon_x = self.decode(z, z_download, labels)
        return recon_x, mu, mu_download, logvar, logvar_download


def VAE_loss_function(recon_x, x, mu, mu_download, logvar, logvar_download, beta=1.0):
    recon_loss = nn.MSELoss()(recon_x, x)  # 重构误差

    # 多尺度 KL 散度
    total_kl_loss = 0.
    for idx in range(len(mu_download)):
        kl_loss_download = 0.5 * torch.sum(
            logvar_download[idx].exp() + mu_download[idx].pow(2) - logvar_download[idx] - 1)
        total_kl_loss += kl_loss_download
    return recon_loss + beta * total_kl_loss


def generate_images_for_dataset_train(config, model, data_loader_train, logger):
    latent_dim = 64
    num_images = len(data_loader_train.dataset)
    num_classes = config.MODEL.NUM_CLASSES

    # 训练过程
    model_VAE = VAEUNet(config=config, model=model, latent_dim=latent_dim, num_classes=num_classes).cuda()
    optimizer = torch.optim.AdamW(model_VAE.parameters(), lr=1e-7)

    num_epochs = 100
    model_VAE.train()
    for epoch in range(num_epochs):
        for x, labels in data_loader_train:
            x = x.cuda()
            labels = F.one_hot(labels, num_classes=num_classes).float().cuda()
            recon_x, mu, mu_download, logvar, logvar_download = model_VAE(x, labels)
            loss = VAE_loss_function(recon_x, x, mu, mu_download, logvar, logvar_download)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.info(f"VAE_model: Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    # 生成新的带标签图像
    model_VAE.eval()
    with torch.no_grad():
        batch_size = 10
        generated_images_list = []
        labels_list = []

        num_batches = (num_images + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_images - batch_idx * batch_size)

            # 从标准正态分布中采样潜在变量
            z = torch.randn(current_batch_size, latent_dim // 16).cuda()
            z_download = [torch.randn(current_batch_size, latent_dim // 2 ** i).cuda() for i in
                          range(model_VAE.depths)]

            # 随机生成标签
            labels = torch.randint(0, num_classes, (current_batch_size,)).cuda()
            labels = F.one_hot(labels, num_classes=num_classes).float()

            # 解码生成图像
            generated_images = model_VAE.decode(z, z_download, labels)

            # 将生成的图像和标签添加到列表中
            generated_images_list.append(generated_images.cpu())
            labels_list.append(labels.cpu())

            logger.info(f"Generated {len(generated_images_list) * batch_size} / {num_images} images.")

        # 将所有生成的图像和标签拼接成一个完整的张量
        generated_images = torch.cat(generated_images_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        logger.info("Finished generating images for dataset_train.")

    # 生成新的数据集
    generated_dataset = TensorDataset(generated_images.cpu(), labels.cpu())
    generated_data_loader = DataLoader(
        generated_dataset,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    # 保存生成的图像
    for idx, (images, labels) in enumerate(generated_data_loader):
        for i in range(images.size(0)):
            image = images[i].cpu().numpy().transpose(1, 2, 0)
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save(f"generated_images/{idx * config.DATA.BATCH_SIZE + i}.jpg")

    total_dataset = ConcatDataset([data_loader_train.dataset, generated_data_loader.dataset])
    num_tasks = 1
    global_rank = 0
    sampler = DistributedSampler(
        total_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    total_data_loader = DataLoader(
        total_dataset, sampler=sampler,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )
    return total_data_loader
