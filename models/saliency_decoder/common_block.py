import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
import math
import collections.abc as container_abcs
from itertools import repeat
from einops import rearrange as o_rearrange
BATCHNORM = nn.BatchNorm2d

def _ntuple(n):

    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()


def Temporal_Reduce(in_channel, out_channel, temporal_size):
    return nn.Sequential(
        ConvBlock(in_channel, in_channel, temporal_size=temporal_size))


def conv_bn_relu(in_channel, out_channel):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=1),
                         nn.BatchNorm2d(out_channel), 
                         nn.ReLU(True))

def conv3d_bn_relu(in_channel, out_channel):
    return nn.Sequential(nn.Conv3d(in_channel, out_channel, 3, padding=1),
                         nn.BatchNorm3d(out_channel), 
                         nn.ReLU(True))

class UpMem(nn.Module):

    def __init__(
        self,
        size=(7, 12),
        patch_size=7,
        in_chans=3,
        embed_dim=64,
        stride=4,
        padding=2,
        
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size


        self.proj = nn.Sequential(
            nn.Upsample(size=size, mode='bilinear', align_corners=False),
            nn.Conv2d(in_chans,
                      embed_dim,
                      kernel_size=patch_size,
                      padding=padding,
                      stride=stride,
                      bias=False,
                      dilation=padding), nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim,
                      embed_dim,
                      kernel_size=patch_size,
                      padding=padding,
                      stride=stride,
                      bias=False,
                      dilation=padding), nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True))

    def forward(self, x):
        B, C, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x



class ConvBlock(nn.Module):

    def __init__(self, inplanes, planes, temporal_size=1):
        super(ConvBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv = nn.Conv3d(inplanes,
                              planes,
                              kernel_size=(temporal_size, 3, 3),
                              stride=(temporal_size, 1, 1),
                              padding=(0, 1, 1),
                              bias=False)

        norm_layer = nn.BatchNorm3d
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class MLPHead(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(MLPHead, self).__init__()

        self.linear_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_pred(x)
        x = self.sig(x)
        return x


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ReduceTemp(nn.Module):

    def __init__(
        self,
        in_chans=3,
        embed_dim=64,
        temporal_dim=1,
        stride=1,
        padding=0,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_chans,
                      embed_dim,
                      kernel_size=(temporal_dim, 1, 1),
                      stride=(stride, 1, 1),
                      padding=(padding, 0, 0),
                      bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class UpEmbed(nn.Module):

    def __init__(
        self,
        patch_size=7,
        in_chans=3,
        embed_dim=64,
        stride=4,
        padding=2,
        up_or_down="down",
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        if up_or_down == "up":
            sf = 2
        else:
            sf = 0.5

        self.proj = nn.Sequential(
            nn.Upsample(scale_factor=sf, mode='bilinear', align_corners=False),
            nn.Conv2d(in_chans,
                      embed_dim,
                      kernel_size=patch_size,
                      padding=padding,
                      stride=stride,
                      bias=False,
                      dilation=padding), 
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim,
                      embed_dim,
                      kernel_size=patch_size,
                      padding=padding,
                      stride=stride,
                      bias=False,
                      dilation=padding), 
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        B, C, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x
