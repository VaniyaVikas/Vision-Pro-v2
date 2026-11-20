import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DropPath(nn.Module):
    """Stochastic depth."""
    def __init__(self, p=0.0):
        super().__init__(); self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        mask = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(keep)
        return x * mask / keep


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.qkv, self.proj = heads, nn.Linear(dim, dim * 3), nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)       # fused / flash kernel when available
        return self.proj(x.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, dim, heads, dp, mlp_ratio=4):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(dim), Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        self.dp = DropPath(dp)

    def forward(self, x):
        x = x + self.dp(self.attn(self.n1(x)))
        return x + self.dp(self.mlp(self.n2(x)))


class ViT(nn.Module):
    """Vision Transformer written from scratch (pre-norm, CLS token, learned pos-emb)."""
    def __init__(self, num_classes, img_size=224, patch=16, dim=384, depth=12, heads=6, drop_path=0.1):
        super().__init__()
        self.grid = img_size // patch
        self.embed = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.grid ** 2 + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList(Block(dim, heads, drop_path * i / max(depth - 1, 1)) for i in range(depth))
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, num_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x)[:, 0])


VIT_SPECS = {"vit_tiny": dict(dim=192, depth=12, heads=3), "vit_small": dict(dim=384, depth=12, heads=6)}


def build_model(name, num_classes, img_size=224, patch=16, pretrained=False, drop_path=0.1):
    if name in VIT_SPECS:
        return ViT(num_classes, img_size, patch, drop_path=drop_path, **VIT_SPECS[name])
    if name == "resnet50":
        m = torchvision.models.resnet50(weights="DEFAULT" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "convnext_tiny":
        m = torchvision.models.convnext_tiny(weights="DEFAULT" if pretrained else None,
                                             stochastic_depth_prob=drop_path)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    raise ValueError(f"unknown model '{name}'")


def cam_target(model):
    """(layer, to_map) for Grad-CAM. to_map turns the layer output into (B,C,h,w)."""
    if isinstance(model, ViT):
        g = model.grid
        layer = model.blocks[-2] if len(model.blocks) > 1 else model.blocks[-1]  # last block's patch tokens get no gradient
        return layer, lambda a: a[:, 1:].transpose(1, 2).reshape(a.size(0), -1, g, g)
    if hasattr(model, "layer4"):
        return model.layer4, lambda a: a
    return model.features[-1], lambda a: a
