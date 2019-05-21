# https://arxiv.org/abs/1505.04597
# https://affinelayer.com/pix2pix/

from functools import partial

import torch
import torch.nn as nn


# TODO
# norm_func = {"batch_norm": nn.BatchNorm2d, "instance_norm": nn.InstanceNorm2d}
# act_func = {"relu": lambda: nn.ReLU(inplace=True), "lrelu": lambda: nn.LeakyReLU(0.2, inplace=True)}


def _conv(
    conv_func, norm_func, act_func, dropout, in_channels, out_channels, kernel_size, stride, padding
):
    layers = [
        conv_func(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not norm_func,
        )
    ]
    if norm_func:
        layers.append(norm_func(out_channels))
    layers.append(act_func())
    if dropout and (0.0 < dropout < 1.0):
        layers.append(nn.Dropout(dropout))
    return layers


class UnetDown(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_func=nn.BatchNorm2d,
        act_func=lambda: nn.ReLU(inplace=True),
        dropout=0.2,
    ):
        super().__init__()
        conv = partial(_conv, nn.Conv2d, norm_func, act_func, dropout)
        self.dn_conv = nn.Sequential(*conv(in_channels, out_channels, 4, 2, 1))
        self.conv = nn.Sequential(
            *conv(out_channels, out_channels, 3, 1, 1), *conv(out_channels, out_channels, 3, 1, 1)
        )

    def forward(self, x):
        return self.conv(self.dn_conv(x))


class UnetUp(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_func=nn.BatchNorm2d,
        act_func=lambda: nn.ReLU(inplace=True),
        dropout=0.3,
    ):
        super().__init__()
        tconv = partial(_conv, nn.ConvTranspose2d, norm_func, act_func, dropout)
        self.up_conv = nn.Sequential(*tconv(in_channels, out_channels, 4, 2, 1))
        self.conv = nn.Sequential(
            *tconv(out_channels * 2, out_channels, 3, 1, 1),
            *tconv(out_channels, out_channels, 3, 1, 1)
        )
        self.out_channels = out_channels

    def forward(self, x, skip):
        x = self.up_conv(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, channel_size=64):
        super().__init__()
        # encoders
        self.dn1 = UnetDown(3, channel_size, norm_func=None)
        self.dn2 = UnetDown(channel_size, channel_size * 2)
        self.dn3 = UnetDown(channel_size * 2, channel_size * 4)
        self.dn4 = UnetDown(channel_size * 4, channel_size * 8)
        self.dn5 = UnetDown(channel_size * 8, channel_size * 8)
        self.dn6 = UnetDown(channel_size * 8, channel_size * 8)
        self.dn7 = UnetDown(channel_size * 8, channel_size * 8)
        self.dn8 = UnetDown(channel_size * 8, channel_size * 8, norm_func=None)
        # decoders
        self.up1 = UnetUp(channel_size * 8, channel_size * 8)
        self.up2 = UnetUp(channel_size * 8, channel_size * 8)
        self.up3 = UnetUp(channel_size * 8, channel_size * 8)
        self.up4 = UnetUp(channel_size * 8, channel_size * 8)
        self.up5 = UnetUp(channel_size * 8, channel_size * 4)
        self.up6 = UnetUp(channel_size * 4, channel_size * 2)
        self.up7 = UnetUp(channel_size * 2, channel_size)
        # out layer
        out_conv = partial(_conv, nn.ConvTranspose2d, nn.BatchNorm2d, nn.Tanh, 0.3)
        self.out = nn.Sequential(*out_conv(channel_size, 3, 4, 2, 1))

    def forward(self, x):
        # x.size() -> [C, 3, 256, 256]
        d1 = self.dn1(x)  # d1.size() -> [C, 64, 128, 128]
        d2 = self.dn2(d1)  # d2.size() -> [C, 128, 64, 64]
        d3 = self.dn3(d2)  # d3.size() -> [C, 256, 32, 32]
        d4 = self.dn4(d3)  # d4.size() -> [C, 512, 16, 16]
        d5 = self.dn5(d4)  # d5.size() -> [C, 512, 8, 8]
        d6 = self.dn6(d5)  # d6.size() -> [C, 512, 4, 4]
        d7 = self.dn7(d6)  # d7.size() -> [C, 512, 2, 2]
        d8 = self.dn8(d7)  # d8.size() -> [C, 512, 1, 1]
        u1 = self.up1(d8, d7)  # u1.size() -> [C, 512, 2, 2]
        u2 = self.up2(u1, d6)  # u2.size() -> [C, 512, 4, 4]
        u3 = self.up3(u2, d5)  # u3.size() -> [C, 512, 8, 8]
        u4 = self.up4(u3, d4)  # u4.size() -> [C, 512, 16, 16]
        u5 = self.up5(u4, d3)  # u5.size() -> [C, 256, 32, 32]
        u6 = self.up6(u5, d2)  # u6.size() -> [C, 128, 64, 64]
        u7 = self.up7(u6, d1)  # u7.size() -> [C, 64, 128, 128])
        out = self.out(u7)  # out.size() -> [C, 3, 256, 256]
        return out
