
import argparse
import torch
import torch.nn as nn
from core.models import build_model


class Calibrated(nn.Module):
    def __init__(self, m, t):
        super().__init__(); self.m, self.t = m, t

    def forward(self, x):
        return (self.m(x) / self.t).softmax(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/exp1/model.pt"); ap.add_argument("--out", default="model.onnx")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True); c = ck["config"]
    m = build_model(c["model"], len(ck["classes"]), c["img_size"], c["patch"], False, 0.0)
    m.load_state_dict(ck["state_dict"])
    net = Calibrated(m.eval(), ck["temperature"]).eval()
    args = (torch.randn(1, 3, c["img_size"], c["img_size"]),)
    kw = dict(input_names=["image"], output_names=["probs"], dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(net, args, a.out, dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(net, args, a.out, **kw)
    print("exported", a.out)
