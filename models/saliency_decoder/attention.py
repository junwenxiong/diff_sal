from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from .common_block import rearrange
BATCHNORM = nn.BatchNorm2d

class Attention(nn.Module):

    def __init__(self,
                 fea_no,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 q_method='dw_bn',
                 kv_method='dw_bn',
                 kernel_size_q=3,
                 kernel_size_kv=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 **kwargs):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out**-0.5
        self.fea_no = fea_no

        self.conv_proj_q = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv3d(dim_in,
                                dim_in,
                                kernel_size=kernel_size_q,
                                padding=padding_q,
                                stride=stride_q,
                                bias=False,
                                groups=dim_in)),
                ('rearrage', Rearrange('b c t h w -> b (t h w) c')),
                ('bn', nn.LayerNorm(dim_in)),
            ]))

        kernel_size_kv = (1, kernel_size_kv, kernel_size_kv)
        stride_kv = (1, stride_kv, stride_kv)

        self.conv_proj_k = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv3d(dim_in,
                                dim_in,
                                kernel_size=kernel_size_kv,
                                padding=padding_kv,
                                stride=stride_kv,
                                bias=False,
                                groups=dim_in)),
                ('rearrage', Rearrange('b c t h w -> b (t h w) c')),
                ('bn', nn.LayerNorm(dim_in)),
            ]))

        self.conv_proj_v = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv3d(dim_in,
                                dim_in,
                                kernel_size=kernel_size_kv,
                                padding=padding_kv,
                                stride=stride_kv,
                                bias=False,
                                groups=dim_in)),
                ('rearrage', Rearrange('b c t h w -> b (t h w) c')),
                ('bn', nn.LayerNorm(dim_in)),
            ]))

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, h, w, messages, bs=None, ts=None, audio_cond=None):
        x =  rearrange(x, 'b (t h w) c -> b c t h w', t=self.fea_no, h=h, w=w)
        if audio_cond != None:
            audio_cond =  rearrange(audio_cond, 'b (t h w) c -> b c t h w', t=self.fea_no, h=h, w=w)
            k = self.conv_proj_k(audio_cond)
        else:
            k = self.conv_proj_k(x)

        q = self.conv_proj_q(x)
        v = self.conv_proj_v(x)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale

        messages['attn'] = attn_score
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

