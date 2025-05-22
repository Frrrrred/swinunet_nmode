from .data_swin_unet_nmode import build_loader_swinUnet_nmODE
from .data_finetune import build_loader_finetune, generate_images_for_dataset_train


def build_loader(config, logger, is_pretrain):
    if is_pretrain:
        return build_loader_swinUnet_nmODE(config, logger)
    else:
        return build_loader_finetune(config, logger)
