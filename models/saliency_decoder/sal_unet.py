import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange
from util.registry import OBJECT_REGISTRY
from .common_block import ReduceTemp, MLPHead, conv_bn_relu
from .transformer import TransformerStage

Norm = nn.LayerNorm


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Downsample4x4(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=4, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=4)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


@OBJECT_REGISTRY.register_module()
class SalUNet(nn.Module):
    def __init__(
        self,
        image_based=False,
        img_size=(224, 384),
        frames_len=2,
        tasks=["futr"],
        in_index=[0, 1, 2, 3],
        idx_to_planes={0: 96, 1: 192, 2: 384, 3: 768},
        temporal_size=5,
        mid_num_stages=3,
        futr_num_stages=1,
        ori_embed_dim=768,
        down_embed_dim=96,
        keep_max_len=5,
        exclude_layers=[],
        temporal_list=[1, 9, 9],
        patch_size=[0, 3, 3],
        patch_stride=[0, 1, 1],
        patch_padding=[0, 2, 2],
        up_channel=[768, 384, 192],
        num_heads=[2, 2, 2],
        mlp_ratio=[4.0, 4.0, 4.0],
        drop_path_rate=[0.15, 0.15, 0.15],
        qkv_bias=[True, True, True],
        kv_proj_method=["avg", "avg", "avg"],
        kernel_kv=[2, 4, 8],
        padding_kv=[0, 0, 0],
        stride_kv=[2, 4, 8],
        q_proj_method=["dw_bn", "dw_bn", "dw_bn"],
        kernel_q=[3, 3, 3],
        padding_q=[1, 1, 1],
        stride_q=[1, 1, 1],  # 1 task 1 and 2 task 2
    ):
        super().__init__()

        self.frame_len = frames_len
        self.img_size = img_size
        self.image_based = image_based
        self.img_size = (self.img_size[0], self.img_size[1])

        tasks = torch.arange(0, frames_len).tolist()  # (0, ..., frames_len - 1)
        tasks = [str(x) for x in tasks]

        self.tasks = tasks[0]
        self.in_index = in_index
        self.idx_to_planes = idx_to_planes
        self.in_channels = self.idx_to_planes[3]
        self.up_channel = self.idx_to_planes[3]
        self.down_channel = self.idx_to_planes[0]
        self.temporal_size = temporal_size

        self.invpt_decoder = Decoder(
            tasks=self.tasks,
            temporal_size=temporal_size,
            temporal_list=temporal_list,
            patch_size=patch_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            mid_num_stages=mid_num_stages,
            futr_num_stages=futr_num_stages,
            ori_embed_dim=ori_embed_dim,
            down_embed_dim=down_embed_dim,
            keep_max_len=keep_max_len,
            exclude_layers=exclude_layers,
            up_channel=up_channel,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            qkv_bias=qkv_bias,
            kv_proj_method=kv_proj_method,
            kernel_kv=kernel_kv,
            padding_kv=padding_kv,
            stride_kv=stride_kv,
            q_proj_method=q_proj_method,
            kernel_q=kernel_q,
            padding_q=padding_q,
            stride_q=stride_q,
        )

        self.logits = MLPHead(self.down_channel, 1)

        self.ch = 96
        dropout = 0.1
        self.temb_ch = self.ch * 4
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList(
            [
                torch.nn.Linear(self.ch, self.temb_ch),
                torch.nn.Linear(self.temb_ch, self.temb_ch),
            ]
        )

        self.scale_list = [(7, 12), (14, 24), (28, 48)]
        self.conv_in = torch.nn.Conv2d(1, self.ch, kernel_size=3, stride=1, padding=1)

        self.down1 = Downsample4x4(in_channels=self.ch, with_conv=True)

        out_conv = up_channel[:-1][::-1]

        in_c = self.ch
        self.res_encoder = nn.ModuleList()
        for out_c in out_conv:
            res_block = nn.Sequential(
                ResnetBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    temb_channels=self.temb_ch,
                    dropout=dropout,
                ),
                Downsample(in_channels=out_c, with_conv=True),
            )
            in_c = out_c
            self.res_encoder.append(res_block)

        self.init_weights()

    def init_weights(self):
        # By default we use pytorch default initialization. Heads can have their own init.
        # Except if `logits` is in the name, we override.
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv3d)):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def noise_downsample(self, x, temb):
        """extract noisy feature list

        Args:
            x (_type_): _description_
            temb (_type_): _description_

        Returns:
            _type_: _description_
            [4, 768, 1, 7, 12]
            [4, 384, 1, 14, 24]
            [4, 192, 1, 28, 48]
        """
        feat_x = self.conv_in(x)
        feat_x = self.down1(feat_x)

        feat_x_list = []
        for block in self.res_encoder:
            feat_x = block[0](feat_x, temb)
            feat_x = block[1](feat_x)
            feat_x_list.append(feat_x.unsqueeze(2))
        return feat_x_list[::-1]

    def forward(self, x, t, feat_list, audio_feat_list=None):
        # add time information
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        x_scale_list = self.noise_downsample(x, temb)

        if self.image_based:
            for i, feat in enumerate(feat_list):
                if i > len(x_scale_list) - 1:
                    continue

                if feat.shape[-2:] == x_scale_list[i].shape[-2:]:
                    feat_list[i] = torch.cat([feat_list[i], x_scale_list[i]], dim=2)

        pred = self.invpt_decoder(feat_list, audio_feat_list)
        pred = self.logits(pred)
        # for multi frames
        if len(pred.shape) == 5:
            pred = pred.squeeze(1)

        final_pred = nn.functional.interpolate(
            pred, size=self.img_size, mode="bilinear", align_corners=False
        )
        return final_pred


class Decoder(nn.Module):
    def __init__(
        self,
        tasks=["futr"],
        mid_num_stages=3,
        futr_num_stages=1,
        ori_embed_dim=768,
        down_embed_dim=96,
        keep_max_len=5,
        exclude_layers=[],
        temporal_size=5,
        temporal_list=[1, 9, 9],
        patch_size=[0, 3, 3],
        patch_stride=[0, 1, 1],
        patch_padding=[0, 2, 2],
        up_channel=[768, 384, 192],
        num_heads=[2, 2, 2],
        mlp_ratio=[4.0, 4.0, 4.0],
        drop_path_rate=[0.15, 0.15, 0.15],
        qkv_bias=[True, True, True],
        kv_proj_method=["avg", "avg", "avg"],
        kernel_kv=[2, 4, 8],
        padding_kv=[0, 0, 0],
        stride_kv=[2, 4, 8],
        q_proj_method=["dw_bn", "dw_bn", "dw_bn"],
        kernel_q=[3, 3, 3],
        padding_q=[1, 1, 1],
        stride_q=[1, 1, 1],  # 1 task 1 and 2 task 2
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.act_layer = act_layer
        self.norm_layer = Norm
        self.tasks = tasks

        self.num_stages = mid_num_stages
        self.down_num_stages = futr_num_stages
        self.embed_dim = ori_embed_dim

        self.mt_in_chans = ori_embed_dim
        self.target_channel = ori_embed_dim
        self.down_target_channel = down_embed_dim

        self.keep_max_len = keep_max_len
        self.exclude_layers = exclude_layers

        self.temporal_size = temporal_size
        self.temporal_list = temporal_list
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.up_channel = up_channel
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias
        self.kv_proj_method = kv_proj_method
        self.kernel_kv = kernel_kv
        self.padding_kv = padding_kv
        self.stride_kv = stride_kv
        self.q_proj_method = q_proj_method
        self.kernel_q = kernel_q
        self.padding_q = padding_q
        self.stride_q = stride_q

        self.mt_embed_dims = []
        self.norm_mts = nn.ModuleList()
        self.redu_chan_up = nn.ModuleList()
        self.mid_stages = nn.ModuleList()

        mid_block = TransformerStage

        for i in range(self.num_stages):
            self.make_stage(mid_block, i, self.target_channel, "up")

        # Final convs
        self.mt_proj = conv_bn_relu(self.target_channel, self.down_target_channel)
        trunc_normal_(self.mt_proj[0].weight, std=0.02)

    def make_stage(self, block, i, out_channel, stage_type="up"):
        cur_mt_embed_dim = self.up_channel[i]

        kwargs = {
            "patch_size": self.patch_size[i],
            "patch_stride": self.patch_stride[i],
            "patch_padding": self.patch_padding[i],
            "embed_dim": cur_mt_embed_dim,
            "depth": 1,
            "num_heads": self.num_heads[i],
            "mlp_ratio": self.mlp_ratio[i],
            "qkv_bias": self.qkv_bias[i],
            "drop_rate": 0,
            "attn_drop_rate": 0,
            "drop_path_rate": self.drop_path_rate[i],
            "q_method": self.q_proj_method[i],
            "kv_method": self.kv_proj_method[i],
            "kernel_size_q": self.kernel_q[i],
            "kernel_size_kv": self.kernel_kv[i],
            "padding_q": self.padding_q[i],
            "padding_kv": self.padding_kv[i],
            "stride_kv": self.stride_kv[i],
            "stride_q": self.stride_q[i],
        }

        self.mid_stages.append(
            block(
                stage_idx=i,
                tasks=self.tasks,
                in_chans=self.mt_in_chans,
                act_layer=self.act_layer,
                norm_layer=self.norm_layer,
                up_or_down=stage_type,
                **kwargs
            )
        )
        self.mt_in_chans = cur_mt_embed_dim
        self.norm_mts.append(self.norm_layer(self.mt_in_chans * len(self.tasks)))
        self.mt_embed_dims.append(self.mt_in_chans)
        _redu_chan = ReduceTemp(
            self.mt_in_chans,
            out_channel,
            temporal_dim=self.temporal_list[i],
            stride=self.temporal_list[i],
        )
        self.redu_chan_up.append(_redu_chan)

    def forward(self, back_fea, audio_cond=None):

        messages = {"attn": None}
        x = back_fea[0]
        h, w = x.shape[3:]
        x_list = [x]
        th = h * 2 ** (self.num_stages - 1) * 2
        tw = w * 2 ** (self.num_stages - 1) * 2

        multi_scale_task_feature = 0
        for i in range(self.num_stages):
            x_list = self.mid_stages[i](x_list, messages, back_fea, audio_cond)
            _x_list = [rearrange(_x, "b c t h w -> (b t) (h w) c") for _x in x_list]

            t = x_list[0].shape[2]
            x = torch.cat(_x_list, dim=2)
            x = self.norm_mts[i](x)

            nh = h * 2 ** (i)
            nw = w * 2 ** (i)
            x = rearrange(x, "(b t) (h w) c -> b c t h w", t=t, h=nh, w=nw)

            # upsample
            task_x = self.redu_chan_up[i](x)
            task_x = task_x.squeeze(2)
            task_x = F.interpolate(
                task_x, size=(th, tw), mode="bilinear", align_corners=False
            )

            # add feature from all the scales
            multi_scale_task_feature += task_x

        multi_scale_task_feature = self.mt_proj(multi_scale_task_feature)

        return multi_scale_task_feature
