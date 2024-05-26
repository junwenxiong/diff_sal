import torch
import torch.nn as nn
import torch.nn.functional as F
# from util.registry import OBJECT_REGISTRY

class VGGish_v2(nn.Module):
    def __init__(self):
        super(VGGish_v2, self).__init__()
        
        layers = []
        in_channels = 1
        for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*layers)

        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True))
        
        self.fc = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        
        self._freeze_stages()

    def _freeze_stages(self):
        self.fc.eval()
        for param in self.fc.parameters():
            param.requires_grad = False

        self.features.eval()
        for m in self.features:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        self.embeddings.eval()
        for m in self.features:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        x = self.features(x)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.embeddings(x)
        
    def forward_feat(self, x):
        x = self.features(x)
        # x = F.avg_pool2d(x, (3, 2), stride=(3, 2))
        x = F.pad(x, (4, 4, 0, 1), mode="replicate")
        x = self.fc(x)
        x = x.view(x.size(0), -1, x.size(1))
        return x

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True))

    def forward(self, x):
        x1 = self.features(x)
        x = torch.transpose(x1, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return x1, self.embeddings(x)
    def forward_feat(self, x):
        x = self.features(x)

        return x


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# @OBJECT_REGISTRY.register_module()
class VGGish(VGG):
    def __init__(self,  pretrained=True):
        super().__init__(make_layers())

        model_urls = {
            'vggish': 'data/pretrained_models/vggish.pth',
            'pca': 'data/pretrained_models/vggish_pca_params.pth'
        }

        if pretrained:
            # state_dict = hub.load_state_dict_from_url(model_urls['vggish'], progress=progress)
            state_dict = torch.load(model_urls['vggish'])
            print(model_urls['vggish'])
            super().load_state_dict(state_dict)


    def forward(self, x):
        x1, x2 = VGG.forward(self, x)
        return x1, x2
    
if __name__ == "__main__":

    from models.audio_attention import Transformer
    model = Transformer(dim=512, depth=1, heads=2, dim_head=64, mlp_dim=256)
    # model = AudioAttNet(512*7*12, seq_len=9)
    input_x = torch.randn(2, 9*7*12, 512)
    out = model(input_x)
    out = out.reshape(2, 512, 9, 7, 12)
    print(out.shape)