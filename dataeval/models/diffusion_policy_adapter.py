# dataeval/models/diffusion_policy_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from PIL import Image
from dataeval.paths import RESNET18_WEIGHT_PATH

# 默认 ImageNet 权重路径
IMAGENET_WEIGHT_PATH = str(RESNET18_WEIGHT_PATH)

class SpatialSoftmax2d(nn.Module):
    """Spatial Softmax，将 [B,C,H,W] -> [B,C*2]"""
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        softmax = F.softmax(x, dim=-1)
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1,1,H,device=x.device),
            torch.linspace(-1,1,W,device=x.device),
            indexing='ij'
        )
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)
        exp_x = (softmax * pos_x).sum(dim=-1)
        exp_y = (softmax * pos_y).sum(dim=-1)
        return torch.cat([exp_x, exp_y], dim=-1)  # [B,C*2]

class DPResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用标准 ResNet18
        backbone = models.resnet18(pretrained=False)

        # 不替换 BN，保留原 BN
        # 去掉全局平均池化和 fc
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        self.spatial_softmax = SpatialSoftmax2d()
        self.out_dim = 512 * 2  # C*2

        # 加载 ImageNet 权重
        state_dict = torch.load(IMAGENET_WEIGHT_PATH, map_location='cpu')
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state and param.shape == own_state[name].shape:
                own_state[name].copy_(param)
        self.load_state_dict(own_state)

    def forward(self, x):
        x = self.features(x)
        x = self.spatial_softmax(x)
        return x  # [B, 512*2]

class DPResNetAdapter:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DPResNet18().to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, frames):
        imgs = []
        for f in frames:
            img = Image.fromarray(f)
            img = img.resize((224, 224))
            img = torch.tensor(np.array(img).transpose(2,0,1), dtype=torch.float32) / 255.0
            imgs.append(img)
        x = torch.stack(imgs, dim=0).to(self.device)
        feats = self.model(x)
        return feats.cpu().numpy()

    @torch.no_grad()
    def extract_batch(self, batch_frames):
        batch_feats = []
        for frames in batch_frames:
            batch_feats.append(self.extract(frames))
        return np.stack(batch_feats, axis=0)
