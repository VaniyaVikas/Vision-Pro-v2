import argparse
from dataclasses import dataclass, fields


@dataclass
class TrainConfig:
    data_dir: str = "data"            # data/train/<class>/*.jpg, data/val/..., optional data/test/...
    out_dir: str = "runs/exp1"
    model: str = "vit_small"          # vit_tiny | vit_small | resnet50 | convnext_tiny
    pretrained: bool = False          # torchvision weights (resnet50 / convnext_tiny)
    img_size: int = 224
    patch: int = 16                   # ViT only
    epochs: int = 30
    batch_size: int = 64              # per GPU
    accum_steps: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    label_smoothing: float = 0.1
    mix_prob: float = 0.8             # chance a batch gets MixUp/CutMix
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    ema_decay: float = 0.9995
    grad_clip: float = 1.0
    drop_path: float = 0.1
    balanced_loss: bool = True        # class-weighted loss for imbalanced data
    patience: int = 10
    workers: int = 4
    seed: int = 42
    amp: str = "auto"                 # auto | bf16 | fp16 | off
    compile: bool = False             # torch.compile
    resume: bool = False


def parse_args(argv=None) -> TrainConfig:
    ap = argparse.ArgumentParser(description="Vision Pro trainer (single GPU or torchrun DDP)")
    for f in fields(TrainConfig):
        name = "--" + f.name.replace("_", "-")
        if f.type is bool:
            ap.add_argument(name, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(name, type=f.type, default=f.default)
    return TrainConfig(**{k: v for k, v in vars(ap.parse_args(argv)).items()})
