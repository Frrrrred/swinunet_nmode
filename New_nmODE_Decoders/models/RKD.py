import torch.nn.functional as F
import torch.nn as nn


class C_RB_g(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,padding=0):
        super().__init__()

        self.g = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding)

    def forward(self, x, weight_x, scale):
        g_y = F.interpolate(self.g(weight_x * x), scale_factor=scale, mode='bilinear')
        return g_y


class C_RB(nn.Module):
    def __init__(self,in_channel,in_channel_post,out_channel,kernel_size=3,stride=1,padding=0,dilation=1):
        super().__init__()
        self.g_post = nn.Conv2d(in_channels=in_channel_post,out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)

        self.sigma = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x, x_post, y, delta, weight_x, weight_x_post, weight_y, scale, scale_post, g_y):
        g_y_post = F.interpolate(self.g_post(weight_x_post * x_post), scale_factor=scale_post, mode='bilinear')
        average_g_y = (g_y_post + g_y) / 2

        k1 = -weight_y * y + self.sigma(weight_y * y + g_y)
        y_k1 = weight_y * y + 1 / 2 * delta * k1

        k2 = -y_k1 + self.sigma(y_k1 + average_g_y)
        y_k2 = weight_y * y + 1 / 2 * delta * k2

        k3 = -y_k2 + self.sigma(y_k2 + average_g_y)
        y_k3 = weight_y * y + delta * k3

        k4 = -y_k3 + self.sigma(y_k3 + g_y_post)

        y_next = weight_y * y + 1 / 6 * delta * (k1 + 2 * k2 + 2 * k3 + k4)
        return y_next, g_y_post


class C_BR_g(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,padding=0):
        super().__init__()

        self.g = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding)

    def forward(self, x, weight_x, scale):
        g_y = F.interpolate(self.g(weight_x * x), scale_factor=scale, mode='bilinear')
        return g_y


class C_BR(nn.Module):
    def __init__(self,in_channel,in_channel_post,out_channel,kernel_size=3,stride=1,padding=0,dilation=1):
        super().__init__()

        self.g_post = nn.Conv2d(in_channels=in_channel_post,out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)

        self.sigma = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x, x_post, y, delta, weight_x, weight_x_post, weight_y, scale, scale_post, g_y):
        g_y_post = F.interpolate(self.g_post(weight_x_post * x_post), scale_factor=scale_post, mode='bilinear')
        average_g_y = (g_y_post + g_y) / 2

        k1 = -weight_y * y + self.sigma(weight_y * y + g_y)
        y_k1 = weight_y * y + 1 / 2 * delta * k1

        k2 = -y_k1 + self.sigma(y_k1 + average_g_y)
        y_k2 = weight_y * y + 1 / 2 * delta * k2

        k3 = -y_k2 + self.sigma(y_k2 + average_g_y)
        y_k3 = weight_y * y + delta * k3

        k4 = -y_k3 + self.sigma(y_k3 + g_y_post)

        y_next = weight_y * y + 1 / 6 * delta * (k1 + 2 * k2 + 2 * k3 + k4)
        return y_next, g_y_post


class CR_B_g(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,padding=0,dilation=1):
        super().__init__()

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x, weight_x, scale):
        g_y = F.interpolate(self.g(weight_x * x), scale_factor=scale, mode='bilinear')
        return g_y


class CR_B(nn.Module):
    def __init__(self,in_channel,in_channel_post,out_channel,kernel_size=3,stride=1,padding=0,dilation=1):
        super().__init__()

        self.g_post = nn.Sequential(
            nn.Conv2d(in_channels=in_channel_post, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation),
            nn.ReLU()
        )

        self.sigma = nn.Sequential(
            nn.BatchNorm2d(out_channel),
        )

    def forward(self,x,x_post,y,delta,weight_x,weight_x_post,weight_y,scale,scale_post,g_y):
        g_y_post = F.interpolate(self.g_post(weight_x_post * x_post), scale_factor=scale_post, mode='bilinear')
        average_g_y = (g_y_post + g_y) / 2

        k1 = -weight_y * y + self.sigma(weight_y * y + g_y)
        y_k1 = weight_y * y + 1 / 2 * delta * k1

        k2 = -y_k1 + self.sigma(y_k1 + average_g_y)
        y_k2 = weight_y * y + 1 / 2 * delta * k2

        k3 = -y_k2 + self.sigma(y_k2 + average_g_y)
        y_k3 = weight_y * y + delta * k3

        k4 = -y_k3 + self.sigma(y_k3 + g_y_post)

        y_next = weight_y * y + 1 / 6 * delta * (k1 + 2 * k2 + 2 * k3 + k4)
        return y_next, g_y_post


class CB_R_g(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=0, dilation=1):
        super().__init__()

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x, weight_x, scale):
        g_y = F.interpolate(self.g(weight_x * x), scale_factor=scale, mode='bilinear')
        return g_y


class CB_R(nn.Module):
    def __init__(self,in_channel,in_channel_post,out_channel,kernel_size=3,stride=1,padding=0,dilation=1):
        super().__init__()

        self.g_post = nn.Sequential(
            nn.Conv2d(in_channels=in_channel_post, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation),
            nn.BatchNorm2d(out_channel)
        )

        self.sigma = nn.Sequential(
            nn.ReLU()
        )

    def forward(self, x, x_post, y, delta, weight_x, weight_x_post, weight_y, scale, scale_post, g_y):
        g_y_post = F.interpolate(self.g_post(weight_x_post * x_post), scale_factor=scale_post, mode='bilinear')
        average_g_y = (g_y_post + g_y) / 2

        k1 = -weight_y * y + self.sigma(weight_y * y + g_y)
        y_k1 = weight_y * y + 1 / 2 * delta * k1

        k2 = -y_k1 + self.sigma(y_k1 + average_g_y)
        y_k2 = weight_y * y + 1 / 2 * delta * k2

        k3 = -y_k2 + self.sigma(y_k2 + average_g_y)
        y_k3 = weight_y * y + delta * k3

        k4 = -y_k3 + self.sigma(y_k3 + g_y_post)

        y_next = weight_y * y + 1 / 6 * delta * (k1 + 2 * k2 + 2 * k3 + k4)
        return y_next, g_y_post
