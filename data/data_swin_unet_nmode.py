import os
import random

import numpy as np
import torch
# import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


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
    
    
class MaskGenerator:
    def __init__(self, input_size=512, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class SwinUnet_nmODE:
    def __init__(self, config, mask_patch_size, mask_ratio):
        self.transform_img = T.Compose([
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
 
        if config.MODEL.TYPE == 'swinUnet_nmODE':
            model_patch_size = config.MODEL.SWIN.PATCH_SIZE
        else:
            raise NotImplementedError
        
        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=mask_patch_size,
            model_patch_size=model_patch_size,
            mask_ratio=mask_ratio,
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return img, mask

    def convert_to_rgb(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_swinUnet_nmODE(config, logger):
    total_datasets = []
    total_mask_datasets = []
    rand_int = random.randint(0, 2 ** 32)
    for mps in config.DATA.MASK_PATCH_SIZE:
        for mr in config.DATA.MASK_RATIO:
            transform = SwinUnet_nmODE(config, mps, mr)
            logger.info(f'Pre-train data transform:\n{transform}')

            # 加载 train 和 val 数据集
            train_path = os.path.join(config.DATA.DATA_PATH, 'train')
            val_path = os.path.join(config.DATA.DATA_PATH, 'val')
            set_seed(rand_int)
            train_dataset = ImageFolder(train_path, transform=transform)
            val_dataset = ImageFolder(val_path, transform=transform)
            dataset = ConcatDataset([train_dataset, val_dataset])
            logger.info(f'Build dataset: train + val images = {len(dataset)}')
            total_datasets.append(dataset)

            # 加载 mask 下的 train 和 val 数据集
            mask_path = os.path.join(config.DATA.DATA_PATH, 'mask')
            mask_train_path = os.path.join(mask_path, 'train')
            mask_val_path = os.path.join(mask_path, 'val')
            set_seed(rand_int)
            train_mask_dataset = ImageFolder(mask_train_path, transform=transform)
            val_mask_dataset = ImageFolder(mask_val_path, transform=transform)
            mask_dataset = ConcatDataset([train_mask_dataset, val_mask_dataset])
            total_mask_datasets.append(mask_dataset)

            rand_int = random.randint(0, 2 ** 32)
    
    set_seed(None)
    combined_dataset = ConcatDataset(total_datasets)
    combined_dataloader = DataLoader(combined_dataset, config.DATA.BATCH_SIZE, sampler=None, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    combined_mask_dataset = ConcatDataset(total_mask_datasets)
    combined_mask_dataloader = DataLoader(combined_mask_dataset, config.DATA.BATCH_SIZE, sampler=None, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    return combined_dataloader, combined_mask_dataloader
