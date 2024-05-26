import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from .common_block import rearrange, UpEmbed, Mlp
from .attention import *

BATCHNORM = nn.BatchNorm2d


class Conv2dBlock(nn.Module):

    def __init__(
        self,
        task_no,
        dim_in,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d,
        **kwargs
    ):
        super().__init__()

        self.stride_q = kwargs["stride_q"]
        self.embed_dim = dim_in // task_no
        self.task_no = task_no

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = nn.BatchNorm2d(self.embed_dim)
        self.conv1 = nn.Conv2d(
            self.embed_dim, self.embed_dim, kernel_size=3, stride=1, padding=1
        )

        self.align_conv = nn.Conv2d(512, self.embed_dim, kernel_size=1)

    def forward(self, x_list, messages, b, t, audio_cond=None):
        bs, C, H, W = x_list[0].shape

        if audio_cond is not None:
            audio_t = audio_cond.shape[2]
            audio_cond = rearrange(audio_cond, "b c t h w -> (b t) c h w")
            audio_cond = self.align_conv(audio_cond)

            h, w = audio_cond.shape[-2:]
            if h != H and w != W:
                up_rate = H // h
                audio_cond = F.upsample(audio_cond, scale_factor=up_rate)

            audio_cond = rearrange(audio_cond, "(b t) c h w -> b c t h w", t=audio_t)
            x_list[0] = rearrange(x_list[0], "(b t) c h w -> b c t h w", t=t)

            av_feat = F.adaptive_avg_pool3d(
                audio_cond * x_list[0], output_size=(1, H, W)
            )
            av_feat = audio_cond * F.softmax(av_feat, dim=-1)
            x_list[0] = rearrange(av_feat, "b c t h w -> (b t) c h w")

        h, w = x_list[0].shape[2:]
        x = x_list[0]
        x = self.conv1(x)
        x = self.norm1(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        x_list = [rearrange(x, "b (h w) c -> b c h w", h=h, w=w)]
        return x_list


class TransformerBlock(nn.Module):

    def __init__(
        self,
        task_no,
        dim_in,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **kwargs
    ):
        super().__init__()

        self.stride_q = kwargs["stride_q"]
        self.embed_dim = dim_in // task_no
        self.task_no = task_no

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        dim_mlp_hidden = int(self.embed_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=self.embed_dim,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop,
        )

        self.norm = norm_layer(self.embed_dim)
        self.attn = Attention(
            task_no,
            self.embed_dim,
            self.embed_dim,
            num_heads,
            qkv_bias,
            attn_drop,
            drop,
            **kwargs
        )
        self.norm2 = norm_layer(self.embed_dim)
        self.align_conv = nn.Conv2d(512, self.embed_dim, kernel_size=1)

    def forward(self, x_list, messages, b, t, audio_cond=None):
        bs, C, H, W = x_list[0].shape

        # fusing audio feat
        if audio_cond is not None:
            audio_t = audio_cond.shape[2]
            audio_cond = rearrange(audio_cond, "b c t h w -> (b t) c h w")
            audio_cond = self.align_conv(audio_cond)

            h, w = audio_cond.shape[-2:]
            if h != H and w != W:
                up_rate = H // h
                audio_cond = F.upsample(audio_cond, scale_factor=up_rate)

            audio_cond = rearrange(audio_cond, "(b t) c h w -> b c t h w", t=audio_t)
            x_list[0] = rearrange(x_list[0], "(b t) c h w -> b c t h w", t=t)

            av_feat = F.adaptive_avg_pool3d(
                audio_cond * x_list[0], output_size=(1, H, W)
            )
            av_feat = F.softmax(av_feat, dim=-1)
            audio_cond = audio_cond * av_feat
            audio_cond = audio_cond.view(bs, -1, C)
            x_list[0] = rearrange(x_list[0], "b c t h w -> (b t) c h w")

        h, w = x_list[0].shape[2:]
        x = rearrange(x_list[0], "b c h w -> b (h w) c")
        x = (
            self.attn(
                self.norm(x), h, w, messages, b, len(x_list), audio_cond=audio_cond
            )
            + x
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x_list = [rearrange(x, "b (h w) c -> b c h w", h=h, w=w)]
        return x_list


class TransformerStage(nn.Module):

    def __init__(
        self,
        stage_idx,
        tasks,
        patch_size=16,
        patch_stride=16,
        patch_padding=0,
        in_chans=3,
        embed_dim=768,
        depth=1,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_or_down="down",
        init="trunc_norm",
        **kwargs
    ):
        super().__init__()
        assert depth == 1
        self.stage_idx = stage_idx
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.task_no = len(tasks)

        self.rearrage = None

        if patch_size == 0:
            self.patch_embed = None
        else:
            self.patch_embed = [
                UpEmbed(
                    patch_size=patch_size,
                    in_chans=in_chans,
                    stride=patch_stride,
                    padding=patch_padding,
                    embed_dim=embed_dim,
                    up_or_down=up_or_down,
                )
                for _ in range(self.task_no)
            ]

            self.patch_embed = nn.ModuleList(self.patch_embed)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule, but we only use depth=1 here.

        blocks = []
        for j in range(depth):
            blocks.append(
                TransformerBlock(
                    task_no=self.task_no,
                    dim_in=embed_dim * self.task_no,
                    dim_out=embed_dim * self.task_no,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if init == "xavier":
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, BATCHNORM)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, BATCHNORM)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_list, messages, back_fea, audio_cond=None):

        if self.patch_embed != None:
            for i in range(self.task_no):
                x = self.patch_embed[i](x_list[i])
                # backbone skip connection
                if self.stage_idx == 0:
                    x = x + back_fea[0]
                if self.stage_idx == 1:
                    x = x + back_fea[1]
                elif self.stage_idx == 2:
                    x = x + back_fea[2]
                x_list[i] = x

        bs, _, ts = x_list[0].shape[:3]
        for j in range(self.task_no):
            x = x_list[j]
            t = x.shape[2]
            if len(x.shape) == 5:
                x = rearrange(x, "b c t h w -> (b t) c h w")
            x_list[j] = x

        for i, blk in enumerate(self.blocks):
            x_list = blk(x_list, messages, bs, ts, audio_cond)

        for j in range(self.task_no):
            x = x_list[j]
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
            x_list[j] = x

        return x_list
