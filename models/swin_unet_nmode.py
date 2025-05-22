import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

from .swin_transformer import SwinTransformer
from .nmode_decoder import nmODEDecoder


class FeatureLevelMask(nn.Module):
    def __init__(self, mask_ratio=0.05, noise_mean=0.0, noise_std=0.0001):
        super(FeatureLevelMask, self).__init__()

        self.mask_ratio = mask_ratio
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def forward(self, x):
        B, L, C = x.shape

        mask = torch.rand(B, L, device=x.device) < self.mask_ratio  # 表示各个(B,L)是否需要掩码
        mask = mask.unsqueeze(2).expand(-1, -1, C)

        # 生成截断噪声
        noise = torch.randn_like(x) * self.noise_std + self.noise_mean
        noise = torch.clamp(noise, self.noise_mean - 2 * self.noise_std, self.noise_mean + 2 * self.noise_std)
        x_masked = x.clone()
        x_masked[mask] = x_masked[mask] + noise[mask]  # 只在掩码位置添加噪声

        return x_masked


class SwinUnetEncoder(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

        self.feature_mask = FeatureLevelMask(mask_ratio=0.05, noise_mean=0., noise_std=0.0001)
        
        self.mask_conv = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size) 

    def forward(self, x, mask, mask_img):
        x = self.patch_embed(x)

        if mask is not None:
            B, L, _ = x.shape

            mask_tokens = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
            x = x * (1. - w) + mask_tokens * w
        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []
        
        if mask_img is not None:
            mask_img = self.mask_conv(mask_img)
            mask_img = mask_img.flatten(2).transpose(1, 2)
            x = (x + mask_img) / 2

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
            x = self.feature_mask(x)

        x_downsample.append(x)
        x = self.norm(x)
        return x, x_downsample

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class swinUnet_nmODE(nn.Module):
    def __init__(self, encoder, stride, decoder):
        super().__init__()
        self.encoder = encoder
        self.stride = stride
        self.decoder = decoder

        self.out = nn.Sequential(
            nn.Conv2d(
                in_channels=self.decoder.out_channels,
                out_channels=self.stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.stride),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask, mask_img=None):
        x0, x_downsample = self.encoder(x, mask, mask_img)
        x_rec = self.decoder(x0, x_downsample)
        x_rec = self.out(x_rec)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_swinUnet_nmODE(config):
    encoder = SwinUnetEncoder(
        img_size=config.DATA.IMG_SIZE,
        patch_size=config.MODEL.SWIN.PATCH_SIZE,
        in_chans=config.MODEL.SWIN.IN_CHANS,
        num_classes=0,
        embed_dim=config.MODEL.SWIN.EMBED_DIM,
        depths=config.MODEL.SWIN.DEPTHS,
        num_heads=config.MODEL.SWIN.NUM_HEADS,
        window_size=config.MODEL.SWIN.WINDOW_SIZE,
        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        qk_scale=config.MODEL.SWIN.QK_SCALE,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        ape=config.MODEL.SWIN.APE,
        patch_norm=config.MODEL.SWIN.PATCH_NORM,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    decoder = nmODEDecoder(
        config=config,
        activefunc='gelu',
        drop_rate=config.MODEL.DROP_RATE,
        img_size=config.DATA.IMG_SIZE,
        patch_size=config.MODEL.SWIN.PATCH_SIZE,
        embed_dim=config.MODEL.SWIN.EMBED_DIM,
        num_classes=0,
        in_channels=128,  # encoder处最后一层的输出通道数
        out_channels=4,  # 可自定义，代表特征数量
        depths=config.MODEL.SWIN.DEPTHS
    )
    stride = 1

    model = swinUnet_nmODE(encoder=encoder, stride=stride, decoder=decoder)

    return model
