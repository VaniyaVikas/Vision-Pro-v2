import argparse
from collections import defaultdict
from pathlib import Path
from torchvision import datasets

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data_cifar")
ap.add_argument("--per-class", type=int, default=500)
a = ap.parse_args()

tr = datasets.CIFAR10("cifar_raw", True, download=True)
te = datasets.CIFAR10("cifar_raw", False, download=True)

def dump(ds, split, n, skip=0):
    seen = defaultdict(int)
    for img, y in ds:
        c = ds.classes[y]
        seen[c] += 1
        if seen[c] <= skip or seen[c] > skip + n:
            continue
        p = Path(a.out) / split / c
        p.mkdir(parents=True, exist_ok=True)
        img.save(p / f"{seen[c]}.png")

dump(tr, "train", a.per_class)
dump(tr, "val", 100, skip=a.per_class)
dump(te, "test", 100)
print("done:", a.out)