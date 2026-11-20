import copy, csv, math, os, random
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F



def setup_dist():
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank, local = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=60))
    return rank, local, world, device


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)



class EMA:
    def __init__(self, model, decay):
        self.module, self.decay, self.n = copy.deepcopy(model).eval(), decay, 0
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)



def mix_batch(x, y, num_classes, mixup_alpha, cutmix_alpha):
    t = F.one_hot(y, num_classes).float()
    idx = torch.randperm(x.size(0), device=x.device)
    if random.random() < 0.5:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
        x = lam * x + (1 - lam) * x[idx]
    else:
        lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
        H, W = x.shape[2:]
        r = math.sqrt(1 - lam)
        ch, cw = int(H * r), int(W * r)
        cy, cx = random.randint(0, H - 1), random.randint(0, W - 1)
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
        x = x.clone(); x[:, :, y1:y2, x1:x2] = x[idx][:, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return x, lam * t + (1 - lam) * t[idx]


@torch.no_grad()
def collect_logits(model, loader, device, amp_dtype=None):
    model.eval(); outs, ys = [], []
    for x, y in loader:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            outs.append(model(x.to(device, non_blocking=True)).float().cpu())
        ys.append(y)
    return torch.cat(outs), torch.cat(ys)


def classification_metrics(logits, y, classes):
    n = len(classes)
    pred = logits.argmax(1)
    conf = torch.bincount(y * n + pred, minlength=n * n).view(n, n)
    tp = conf.diag().float(); prec = tp / conf.sum(0).clamp(min=1); rec = tp / conf.sum(1).clamp(min=1)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
    return {"acc": (pred == y).float().mean().item(), "macro_f1": f1.mean().item(),
            "nll": F.cross_entropy(logits, y).item(), "ece": ece(logits, y),
            "per_class": {c: {"recall": rec[i].item(), "f1": f1[i].item()} for i, c in enumerate(classes)},
            "confusion": conf.tolist()}


def ece(logits, y, bins=15):
    """Expected calibration error."""
    p = logits.softmax(1); conf, pred = p.max(1); ok = (pred == y).float()
    edges, total = torch.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean().item() * abs(ok[m].mean().item() - conf[m].mean().item())
    return total


def fit_temperature(logits, y):
    """Post-hoc temperature scaling on the validation set (makes confidences trustworthy)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(logits / log_t.exp(), y); loss.backward(); return loss
    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20))


class RunLogger:
    def __init__(self, out_dir, enabled=True):
        self.enabled, self.tb = enabled, None
        if not enabled: return
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.f = open(Path(out_dir) / "log.csv", "a", newline=""); self.w = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(Path(out_dir) / "tb"))
        except Exception:
            pass  # tensorboard not installed -> CSV only

    def log(self, step, **kv):
        if not self.enabled: return
        if self.w is None:
            self.w = csv.DictWriter(self.f, ["step"] + list(kv)); 
            if self.f.tell() == 0: self.w.writeheader()
        self.w.writerow({"step": step, **kv}); self.f.flush()
        if self.tb:
            for k, v in kv.items(): self.tb.add_scalar(k, v, step)
