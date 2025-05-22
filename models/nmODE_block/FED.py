import math

import torch
import torch.nn.functional as F
import torch.nn as nn


class C_RB(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,padding=0,dilation=1):
        super().__init__()

        self.g = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)

        self.sigma = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x, y, delta, weight_x, weight_y, scale, left=None):
        exp = math.exp(-delta)

        y_next = exp * weight_y * y + (1 - exp) * self.sigma(
            weight_y * y + F.interpolate(self.g(weight_x * x), scale_factor=scale, mode='bilinear'))
        y_left = (1 - exp) * weight_y * y + (1 - exp) * self.sigma(
            weight_y * y + F.interpolate(self.g(weight_x * x), scale_factor=scale, mode='bilinear'))
        return y_next, y_left


class C_BR(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,padding=0,dilation=1):
        super().__init__()

        self.g = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)

        self.sigma = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x, y, delta, weight_x, weight_y, scale, left=None):
        exp = math.exp(-delta)

        y_next = exp * weight_y * y + (1 - exp) * self.sigma(
            weight_y * y + F.interpolate(self.g(weight_x * x), scale_factor=scale, mode='bilinear'))
        y_left = (1 - exp) * weight_y * y + (1 - exp) * self.sigma(
            weight_y * y + F.interpolate(self.g(weight_x * x), scale_factor=scale, mode='bilinear'))
        return y_next, y_left


class CR_B(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,padding=0,dilation=1):
        super().__init__()

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation),
            nn.ReLU()
        )

        self.sigma = nn.Sequential(
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x, y, delta, weight_x, weight_y, scale, left=None):
        exp = math.exp(-delta)

        y_next = exp * weight_y * y + (1 - exp) * self.sigma(
            weight_y * y + F.interpolate(self.g(weight_x * x), scale_factor=scale, mode='bilinear'))
        y_left = (1 - exp) * weight_y * y + (1 - exp) * self.sigma(
            weight_y * y + F.interpolate(self.g(weight_x * x), scale_factor=scale, mode='bilinear'))
        return y_next, y_left


class CB_R(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,padding=0,dilation=1):
        super().__init__()

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation),
            nn.BatchNorm2d(out_channel)
        )

        self.sigma = nn.Sequential(
            nn.ReLU()
        )

    def forward(self, x, y, delta, weight_x, weight_y, scale, left=None):
        exp = math.exp(-delta)

        y_next = exp * weight_y * y + (1 - exp) * self.sigma(
            weight_y * y + F.interpolate(self.g(weight_x * x), scale_factor=scale, mode='bilinear'))
        y_left = (1 - exp) * weight_y * y + (1 - exp) * self.sigma(
            weight_y * y + F.interpolate(self.g(weight_x * x), scale_factor=scale, mode='bilinear'))
        return y_next, y_left
