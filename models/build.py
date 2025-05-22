from .swin_transformer import build_predictor
from .swin_unet_nmode import build_swinUnet_nmODE


def build_model(config, is_pretrain=True):
    if is_pretrain:
        model = build_swinUnet_nmODE(config)
    else:
        model_type = config.MODEL.TYPE
        if model_type == 'swinUnet_nmODE':
            model = build_predictor(config)
        else:
            raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model
