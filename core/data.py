from pathlib import Path
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms as T

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def eval_tf(size):
    return T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size)])


def train_tf(size):
    return T.Compose([T.RandomResizedCrop(size, scale=(0.35, 1.0)), T.RandomHorizontalFlip(),
                      T.TrivialAugmentWide(), T.ToTensor(), T.Normalize(MEAN, STD),
                      T.RandomErasing(0.25)])


def val_tf(size):
    return T.Compose([eval_tf(size), T.ToTensor(), T.Normalize(MEAN, STD)])


def class_weights(targets, n):
    counts = torch.bincount(torch.tensor(targets), minlength=n).float().clamp(min=1)
    w = counts.sum() / (n * counts)
    return w / w.mean()


def build_loaders(cfg, rank=0, world=1):
    root = Path(cfg.data_dir)
    if not (root / "train").is_dir() or not (root / "val").is_dir():
        raise FileNotFoundError(f"Expected {root}/train/<class>/* and {root}/val/<class>/*")
    tr = datasets.ImageFolder(root / "train", train_tf(cfg.img_size))
    va = datasets.ImageFolder(root / "val", val_tf(cfg.img_size))
    te = datasets.ImageFolder(root / "test", val_tf(cfg.img_size)) if (root / "test").is_dir() else None
    if va.classes != tr.classes:
        raise ValueError("train/ and val/ must contain identical class folders")
    kw = dict(num_workers=cfg.workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=cfg.workers > 0)
    sampler = DistributedSampler(tr, world, rank, shuffle=True, seed=cfg.seed) if world > 1 else None
    train_dl = DataLoader(tr, cfg.batch_size, shuffle=sampler is None, sampler=sampler,
                          drop_last=True, **kw)
    eval_dl = lambda d: DataLoader(d, cfg.batch_size * 2, **kw)
    return train_dl, eval_dl(va), (eval_dl(te) if te else None), tr.classes, tr.targets
