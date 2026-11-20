"""Creates a tiny synthetic dataset (circle / square / triangle) so you can test the full pipeline in minutes."""
import argparse, random
from pathlib import Path
from PIL import Image, ImageDraw


def make(cls, size=128):
    img = Image.new("RGB", (size, size), tuple(random.randint(0, 90) for _ in range(3)))
    d, c = ImageDraw.Draw(img), tuple(random.randint(120, 255) for _ in range(3))
    cx, cy, r = random.randint(44, 84), random.randint(44, 84), random.randint(18, 34)
    if cls == "circle": d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
    elif cls == "square": d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=c)
    else: d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=c)
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data"); a = ap.parse_args()
    for split, n in (("train", 150), ("val", 40), ("test", 40)):
        for cls in ("circle", "square", "triangle"):
            p = Path(a.out) / split / cls; p.mkdir(parents=True, exist_ok=True)
            for i in range(n): make(cls).save(p / f"{i}.jpg", quality=92)
    print("demo data written to", a.out)
