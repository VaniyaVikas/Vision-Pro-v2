import asyncio, io, pathlib, sys
import pytest
import torch
import torch.nn as nn
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from core.gradcam import GradCAM
from core.models import ViT, build_model
from core.utils import EMA, ece, fit_temperature, mix_batch
from serve.batcher import MicroBatcher


def tiny():
    return ViT(5, img_size=32, patch=8, dim=32, depth=2, heads=2, drop_path=0.1)


def test_vit_forward():
    assert tiny()(torch.randn(3, 3, 32, 32)).shape == (3, 5)


def test_gradcam_vit_shape_and_range():
    _, cam = GradCAM(tiny().eval())(torch.randn(1, 3, 32, 32), out_size=(32, 32))
    assert cam.shape == (32, 32) and not torch.isnan(cam).any() and 0 <= cam.min() and cam.max() <= 1


def test_mix_targets_are_distributions():
    x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 5, (8,))
    xm, t = mix_batch(x, y, 5, 0.2, 1.0)
    assert xm.shape == x.shape and torch.allclose(t.sum(1), torch.ones(8))


def test_ema_moves_toward_model():
    m = nn.Linear(2, 2); ema = EMA(m, 0.9); before = ema.module.weight.clone()
    with torch.no_grad(): m.weight.add_(1.0)
    ema.update(m)
    assert not torch.equal(before, ema.module.weight)


def test_temperature_scaling_improves_overconfident_model():
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 5, (2000,), generator=g)
    pred = torch.where(torch.rand(2000, generator=g) < 0.7, y, torch.randint(0, 5, (2000,), generator=g))
    logits = torch.nn.functional.one_hot(pred, 5).float() * 12 + torch.randn(2000, 5, generator=g)
    T = fit_temperature(logits, y)
    assert T > 1.5 and ece(logits / T, y) < ece(logits, y)


def test_batcher_batches_and_preserves_results():
    async def go():
        b = MicroBatcher(lambda xs: xs * 2, max_batch=8, max_wait_ms=50); await b.start()
        xs = [torch.full((3,), float(i)) for i in range(6)]
        res = await asyncio.gather(*[b.submit(x) for x in xs]); await b.stop()
        assert all(torch.equal(o, x * 2) for (o, _), x in zip(res, xs))
        assert max(bs for _, bs in res) > 1
    asyncio.run(go())


def test_api_end_to_end(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from PIL import Image
    classes = ["a", "b", "c"]
    m = build_model("vit_tiny", 3, 32, 8, False, 0.0)
    ck = {"state_dict": m.state_dict(), "classes": classes, "temperature": 1.3,
          "config": {"model": "vit_tiny", "img_size": 32, "patch": 8},
          "val_metrics": {"acc": 0.9, "macro_f1": 0.9}}
    torch.save(ck, tmp_path / "model.pt")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.pt")); monkeypatch.setenv("DEVICE", "cpu")
    import serve.main as sm
    with TestClient(sm.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        buf = io.BytesIO(); Image.new("RGB", (80, 60), (200, 30, 30)).save(buf, "PNG")
        r = c.post("/api/predict?explain=true", files={"file": ("a.png", buf.getvalue(), "image/png")})
        j = r.json()
        assert r.status_code == 200 and j["label"] in classes and "heatmap" in j
        assert abs(sum(t["p"] for t in j["top"]) - 100) < 0.5
        assert c.post("/api/predict", files={"file": ("x.txt", b"nope", "text/plain")}).status_code == 400
        assert c.get("/metrics").json()["requests"] == 1
