import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import to_2tuple

from .nmODE_block import EAD, FED


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class nmODEDecoder(nn.Module):
    def __init__(self, config, activefunc='gelu', drop_rate=0.1, img_size=224, patch_size=4, embed_dim=96, num_classes=0,
                 in_channels=512, out_channels=4, depths=None):
        super().__init__()
        if depths is None:
            depths = [2, 2, 2, 2]

        self.config = config
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.norm = nn.LayerNorm(3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 使用 nmODE_block 解码器进行特征聚合
        if activefunc == 'relu':
            self.act = nn.ReLU()
        elif activefunc == 'gelu':
            self.act = nn.GELU()
        elif activefunc == 'tanh':
            self.act = nn.Tanh()
        self.out_channels = out_channels
        self.drop = nn.Dropout(p=drop_rate)
        self.layers_up = nn.ModuleList()
        self.skip_patch_expands = nn.ModuleList()

        channels = in_channels
        for i_layer in range(0, self.num_layers + 1):
            expands = nn.ModuleList()
            for i in range(i_layer, i_layer + 2):
                skip_patch_expand = PatchExpand(
                    input_resolution=(int(self.patches_resolution[0] // (2 ** (self.num_layers - i))),
                                      int(self.patches_resolution[1] // (2 ** (self.num_layers - i)))),
                    dim=int(embed_dim * 2 ** (self.num_layers - i)), dim_scale=2, norm_layer=nn.LayerNorm)
                expands.append(skip_patch_expand)

            if i_layer == 0:
                layer_up = FED.CR_B(in_channel=channels, out_channel=self.out_channels, kernel_size=3, stride=1,
                                    padding=1)
            else:
                layer_up = EAD.CR_B(in_channel=channels, out_channel=self.out_channels, kernel_size=3, stride=1,
                                    padding=1)
            channels = channels // 2

            self.layers_up.append(layer_up)
            self.skip_patch_expands.append(expands)

    # Dencoder and Skip connection with PatchExpand
    def forward(self, x, x_downsample):
        y = torch.zeros(x.shape[0], self.out_channels, self.img_size[0], self.img_size[1]).cuda()
        input_weight = torch.ones(5, 2).cuda()
        scale = 16
        y_left = None

        for inx, layer_up in enumerate(self.layers_up):
            for i in range(0, 2):
                x_downsample[inx] = self.skip_patch_expands[4 - inx][i](x_downsample[inx])

            B, L, C = x_downsample[inx].shape
            H = W = int(L ** 0.5)
            x_downsample[inx] = x_downsample[inx].reshape(B, C, H, W)

        for inx, layer_up in enumerate(self.layers_up):
            y, y_left = layer_up(x_downsample[4 - inx], y, 1 / 5, input_weight[inx][0], input_weight[inx][1], scale,
                                 y_left)
            if inx != 4:
                y = self.act(y)
            y = self.drop(y)

            scale = scale // 2

        return y

    def flops(self):
        flops = 0
        for _, layer in enumerate(self.layers_up):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
