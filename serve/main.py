import base64, collections, copy, io, logging, math, os, sys, threading, time
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from core.data import MEAN, STD, eval_tf
from core.gradcam import GradCAM, colorize
from core.models import build_model
from serve.batcher import MicroBatcher

log = logging.getLogger("serve"); logging.basicConfig(level=logging.INFO)
MAX_BYTES = 8 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 60_000_000
S, LOCK = {}, threading.Lock()
STATS = {"requests": 0, "errors": 0, "lat": collections.deque(maxlen=1000)}
MEAN_T, STD_T = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    path = Path(os.getenv("MODEL_PATH", ROOT / "runs" / "exp1" / "model.pt"))
    if path.exists():
        dev = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(path, map_location="cpu", weights_only=True); c = ck["config"]
        model = build_model(c["model"], len(ck["classes"]), c["img_size"], c["patch"], False, 0.0)
        model.load_state_dict(ck["state_dict"]); model.to(dev).eval()
        T = ck["temperature"]

        def run(xs):                                  # calibrated probabilities for a whole batch
            with torch.inference_mode():
                return (model(xs.to(dev)).float() / T).softmax(1).cpu()

        explain = copy.deepcopy(model)                 # own copy: Grad-CAM hooks keep state
        batcher = MicroBatcher(run, int(os.getenv("MAX_BATCH", 16)), float(os.getenv("MAX_WAIT_MS", 8)))
        await batcher.start()
        S.update(dev=dev, cam=GradCAM(explain), batcher=batcher, classes=ck["classes"], size=c["img_size"],
                 name=c["model"], T=T, f1=ck["val_metrics"]["macro_f1"], acc=ck["val_metrics"]["acc"],
                 floor=float(os.getenv("UNCERTAIN_BELOW", 0.6)))
        log.info("loaded %s (%d classes) on %s", c["model"], len(ck["classes"]), dev)
    else:
        S["dev"] = torch.device("cpu"); log.warning("no model at %s - run train.py first", path)
    yield
    if "batcher" in S:
        await S["batcher"].stop()
    S.clear()


app = FastAPI(title="Vision Pro", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "serve" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "serve" / "templates")


@app.middleware("http")
async def timing(request: Request, call_next):
    t = time.perf_counter(); resp = await call_next(request)
    resp.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t) * 1000:.1f}"
    return resp


def b64(img):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_image(raw):
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 8 MB.")
    try:
        img = Image.open(io.BytesIO(raw)); img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "That file is not a readable image. Use JPG, PNG or WebP.")
    return ImageOps.exif_transpose(img).convert("RGB")


def to_tensor(img):
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255


async def analyze(img, explain):
    if "batcher" not in S:
        raise HTTPException(503, "No trained model loaded. Run train.py first.")
    t0, size = time.perf_counter(), S["size"]
    view = eval_tf(size)(img)
    x = (to_tensor(view) - MEAN_T) / STD_T
    probs, bs = await S["batcher"].submit(x)
    k = min(5, len(probs)); top = probs.topk(k)
    ent = float(-(probs * probs.clamp_min(1e-9).log()).sum() / math.log(max(len(probs), 2)))
    res = {"label": S["classes"][int(top.indices[0])], "confidence": round(float(top.values[0]) * 100, 1),
           "uncertain": float(top.values[0]) < S["floor"], "entropy": round(ent, 3),
           "top": [{"label": S["classes"][int(i)], "p": round(float(p) * 100, 1)} for p, i in zip(top.values, top.indices)],
           "batch_size": bs, "image": b64(view)}
    if explain:
        def cam_job():
            with LOCK:
                return S["cam"](x[None].to(S["dev"]), int(top.indices[0]), (size, size))[1].cpu()
        heat = await run_in_threadpool(cam_job)
        ov = (0.55 * to_tensor(view) + 0.45 * colorize(heat)).clamp(0, 1)
        res["heatmap"] = b64(Image.fromarray((ov.permute(1, 2, 0) * 255).byte().numpy()))
    res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    STATS["requests"] += 1; STATS["lat"].append(res["latency_ms"])
    return res


def ctx(request, **kw):
    return {"request": request, "ready": "batcher" in S, "info": S, **kw}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", ctx(request))


@app.post("/", response_class=HTMLResponse)
async def predict_page(request: Request, file: UploadFile = File(...), explain: bool = Form(False)):
    try:
        res = await analyze(load_image(await file.read(MAX_BYTES + 1)), explain)
        return templates.TemplateResponse(request, "index.html", ctx(request, r=res))
    except HTTPException as e:
        STATS["errors"] += 1
        return templates.TemplateResponse(request, "index.html", ctx(request, error=e.detail), status_code=e.status_code)


@app.post("/api/predict")
async def predict_api(file: UploadFile = File(...), explain: bool = Query(False)):
    try:
        return JSONResponse(await analyze(load_image(await file.read(MAX_BYTES + 1)), explain))
    except HTTPException:
        STATS["errors"] += 1; raise


@app.get("/health")
async def health():
    return {"status": "ok" if "batcher" in S else "no_model"}


@app.get("/metrics")
async def metrics():
    lat = sorted(STATS["lat"])
    pct = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] if lat else None
    return {"requests": STATS["requests"], "errors": STATS["errors"], "latency_ms_p50": pct(0.5),
            "latency_ms_p95": pct(0.95), "mean_batch_size": S["batcher"].mean_batch if "batcher" in S else None,
            "model": S.get("name"), "device": str(S.get("dev")), "val_macro_f1": S.get("f1"), "temperature": S.get("T")}
