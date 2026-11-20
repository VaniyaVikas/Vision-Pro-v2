import torch
import torch.nn.functional as F
from core.models import cam_target


class GradCAM:
    """Grad-CAM for CNNs and ViTs. Not thread-safe: use one instance (and lock) per model copy."""
    def __init__(self, model):
        self.model = model
        layer, self.to_map = cam_target(model)
        self.out = self.grad = None
        layer.register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        if out.requires_grad:
            self.out = out
            out.register_hook(lambda g: setattr(self, "grad", g))

    def __call__(self, x, class_idx=None, out_size=(224, 224)):
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            idx = int(logits.argmax(1)) if class_idx is None else class_idx
            logits[0, idx].backward()
        acts, grads = self.to_map(self.out.detach()), self.to_map(self.grad)
        cam = F.relu((grads.mean((2, 3), keepdim=True) * acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=out_size, mode="bilinear", align_corners=False)[0, 0]
        return logits.detach(), (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def colorize(cam):
    f = lambda k: (1.5 - (4 * cam - k).abs()).clamp(0, 1)
    return torch.stack([f(3), f(2), f(1)])
