"""Single-GPU:  python train.py --data-dir data --model resnet50 --pretrained
Multi-GPU:   torchrun --nproc_per_node=4 train.py --data-dir data --model vit_small
Features: DDP, bf16/fp16 AMP, torch.compile, grad accumulation, warmup+cosine LR, AdamW (no WD on norm/bias),
MixUp/CutMix + label smoothing, class-weighted loss, EMA, early stopping, resume, temperature calibration,
macro-F1 / ECE metrics, CSV + TensorBoard logs."""
import json, logging, math, time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from core.config import parse_args
from core.data import build_loaders, class_weights
from core.models import build_model
from core.utils import (EMA, RunLogger, classification_metrics, collect_logits, fit_temperature,
                        mix_batch, seed_all, setup_dist)

log = logging.getLogger("train")


def pick_amp(mode, device):
    if mode == "off" or device.type != "cuda":
        return None
    if mode in ("bf16", "fp16"):
        return torch.bfloat16 if mode == "bf16" else torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def main():
    cfg = parse_args()
    rank, local, world, device = setup_dist()
    is_main = rank == 0
    logging.basicConfig(level=logging.INFO if is_main else logging.WARNING,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    seed_all(cfg.seed + rank)
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    amp = pick_amp(cfg.amp, device)

    train_dl, val_dl, test_dl, classes, targets = build_loaders(cfg, rank, world)
    nc = len(classes)
    raw = build_model(cfg.model, nc, cfg.img_size, cfg.patch, cfg.pretrained, cfg.drop_path).to(device)
    ema = EMA(raw, cfg.ema_decay)
    model = torch.compile(raw) if cfg.compile else raw
    if world > 1:
        model = DDP(model, device_ids=[local] if device.type == "cuda" else None)
    log.info("model=%s params=%.1fM device=%s world=%d amp=%s classes=%d",
             cfg.model, sum(p.numel() for p in raw.parameters()) / 1e6, device, world, amp, nc)

    decay = [p for n, p in raw.named_parameters() if p.ndim > 1 and n not in ("pos", "cls")]
    no_decay = [p for n, p in raw.named_parameters() if p.ndim <= 1 or n in ("pos", "cls")]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)
    spe = max(1, len(train_dl) // cfg.accum_steps)
    total, warm = cfg.epochs * spe, max(1, cfg.warmup_epochs * spe)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    w = class_weights(targets, nc).to(device) if cfg.balanced_loss else torch.ones(nc, device=device)

    start, best, bad = 0, -1.0, 0
    if cfg.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        raw.load_state_dict(ck["model"]); ema.module.load_state_dict(ck["ema"]); ema.n = ck["ema_n"]
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"] + 1, ck["best"]
        log.info("resumed from epoch %d", start)

    logger = RunLogger(out, is_main)
    for ep in range(start, cfg.epochs):
        if world > 1:
            train_dl.sampler.set_epoch(ep)
        model.train(); t0, run, n = time.time(), 0.0, 0
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if torch.rand(1).item() < cfg.mix_prob:
                x, t = mix_batch(x, y, nc, cfg.mixup_alpha, cfg.cutmix_alpha)
            else:
                t = F.one_hot(y, nc).float()
            t = t * (1 - cfg.label_smoothing) + cfg.label_smoothing / nc
            step_now = (i + 1) % cfg.accum_steps == 0 or i + 1 == len(train_dl)
            ctx = model.no_sync() if (world > 1 and not step_now) else nullcontext()
            with ctx:
                with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                    outp = model(x)
                loss = -(t * F.log_softmax(outp.float(), 1) * w).sum(1).mean() / cfg.accum_steps
                scaler.scale(loss).backward()
            if step_now:
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                sched.step(); ema.update(raw)
            run += loss.item() * cfg.accum_steps; n += 1

        stop = torch.zeros(1, device=device)
        if is_main:
            lg, y = collect_logits(ema.module, val_dl, device, amp)
            m = classification_metrics(lg, y, classes)
            logger.log(ep, train_loss=run / n, val_acc=m["acc"], val_f1=m["macro_f1"],
                       val_nll=m["nll"], lr=opt.param_groups[0]["lr"], grad_norm=float(gn))
            log.info("ep %02d/%d | loss %.3f | val acc %.2f%% f1 %.3f ece %.3f | %.0fs",
                     ep + 1, cfg.epochs, run / n, m["acc"] * 100, m["macro_f1"], m["ece"], time.time() - t0)
            if m["macro_f1"] > best:
                best, bad = m["macro_f1"], 0
                torch.save({"state_dict": ema.module.state_dict(), "classes": list(classes),
                            "config": asdict(cfg)}, out / "best.pt")
            else:
                bad += 1
            torch.save({"model": raw.state_dict(), "ema": ema.module.state_dict(), "ema_n": ema.n,
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": ep, "best": best}, out / "last.pt")
            stop.fill_(float(bad >= cfg.patience))
        if world > 1:
            dist.broadcast(stop, 0)
        if stop.item():
            log.info("early stopping"); break

    if is_main:   
        ck = torch.load(out / "best.pt", map_location=device, weights_only=True)
        ema.module.load_state_dict(ck["state_dict"])
        lg, y = collect_logits(ema.module, val_dl, device, amp)
        T = fit_temperature(lg, y)
        before, after = classification_metrics(lg, y, classes), classification_metrics(lg / T, y, classes)
        report = {"temperature": T, "val": after, "val_ece_before_calibration": before["ece"]}
        if test_dl is not None:
            tl, ty = collect_logits(ema.module, test_dl, device, amp)
            report["test"] = classification_metrics(tl / T, ty, classes)
            log.info("TEST acc %.2f%% f1 %.3f ece %.3f", report["test"]["acc"] * 100,
                     report["test"]["macro_f1"], report["test"]["ece"])
        log.info("temperature %.3f | val ECE %.3f -> %.3f", T, before["ece"], after["ece"])
        torch.save({"state_dict": ck["state_dict"], "classes": ck["classes"], "config": ck["config"],
                    "temperature": T, "val_metrics": {"acc": after["acc"], "macro_f1": after["macro_f1"]}},
                   out / "model.pt")
        (out / "report.json").write_text(json.dumps(report, indent=2))
        log.info("saved %s", out / "model.pt")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
