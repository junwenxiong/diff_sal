import torch
import torch.nn as nn
from einops import rearrange
from util.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register_module()
class VideoSaliencyModel(nn.Module):
    def __init__(
        self,
        channel_list,
        visual_net=None,
        spatiotemp_net=None,
        audio_net=None,
        decoder_net=None,
    ):
        super(VideoSaliencyModel, self).__init__()

        if visual_net is not None:
            self.visual_net = OBJECT_REGISTRY.build(visual_net)
            self.visual_cls = visual_net.pop("type")
            backbone_total_params = sum(
                param.numel() for param in self.visual_net.parameters()
            )
        else:
            self.visual_net = None
            backbone_total_params = 0

        if spatiotemp_net is not None:
            self.spatiotemp_net = OBJECT_REGISTRY.build(spatiotemp_net)
            self.spatiotemp_cls = spatiotemp_net.pop("type")
            spatiotemp_total_params = sum(
                param.numel() for param in self.spatiotemp_net.parameters()
            )
        else:
            self.spatiotemp_net = None
            spatiotemp_total_params = 0

        if audio_net is not None:
            audio_channel = 128
            self.audio_net = OBJECT_REGISTRY.build(audio_net)
            self.fc = nn.Sequential(
                nn.Linear(in_features=audio_channel, out_features=512),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=512, out_features=768),
            )
            self.audio_cls = audio_net.pop("type")
            audio_net = sum(param.numel() for param in self.audio_net.parameters())
        else:
            self.audio_net = None
            audio_net = 0

        if decoder_net is not None:
            self.decoder_net = OBJECT_REGISTRY.build(decoder_net)
            self.decoder_cls = decoder_net.pop("type")
            decoder_total_params = sum(
                param.numel() for param in self.decoder_net.parameters()
            )
        else:
            self.decoder_net = None
            decoder_total_params = 0

        if channel_list is not None:
            self.channel_list = channel_list

        print(
            f"FutrFormer : encoder:{backbone_total_params/1e6} M, spatiotemp: {spatiotemp_total_params/1e6} M, decoder: {decoder_total_params/1e6} M"
        )

    def forward_vggish(self, audio):
        # For VGGish
        bs, T = audio.shape[0], audio.shape[2]
        audio = audio.view(-1, audio.shape[1], audio.shape[3], audio.shape[4])
        with torch.no_grad():
            audio_feat_map = self.audio_net.forward_feat(audio)

        audio_feat_map = rearrange(audio_feat_map, "(b t) c h w -> b c t h w", t=T)
        if self.spatiotemp_net is not None:
            audio_feat_map = self.spatiotemp_net(audio_feat_map)

        return audio_feat_map, audio_feat_map

    def forward(self, data, t):
        """_summary_

        Args:
            vis (_type_):
            aud (_type_): # [b*t, 1, 96, 64]

        Returns:
            _type_: _description_
        """

        imgs = data.get("img", None)
        x = data["input"]

        if self.audio_net:
            audio_input = data.get("audio", None)
            audio_feat, audio_feat_embed = self.forward_vggish(audio_input)
        else:
            audio_feat, audio_feat_embed = None, None

        if self.visual_net and imgs is not None:
            vis_list = self.visual_net(imgs)
        else:
            vis_list = [
                torch.randn((audio_feat.shape[0], 768, 8, 7, 12), device=imgs.device),
                torch.randn((audio_feat.shape[0], 384, 8, 14, 24), device=imgs.device),
                torch.randn((audio_feat.shape[0], 192, 8, 28, 48), device=imgs.device),
                torch.randn((audio_feat.shape[0], 96, 8, 56, 96), device=imgs.device),
            ]

        out = self.decoder_net(x, t, vis_list, audio_feat_embed)
        return out
