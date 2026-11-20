# Vision Pro v2 – production image-classification stack (PyTorch · FastAPI · CSS)

Works on **any** image dataset in `ImageFolder` layout:
```
data/train/<class>/*.jpg   data/val/<class>/*.jpg   data/test/<class>/*.jpg (optional)
```

## Quick start (no dataset needed – 3-minute smoke test)
```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q                                              # unit + API integration tests
python tools/make_demo_data.py
python train.py --model vit_tiny --img-size 64 --patch 8 --epochs 5 --workers 0 --batch-size 32
uvicorn serve.main:app --port 8000
```

## Real training
```bash
# small/medium dataset -> fine-tune (best accuracy)
python train.py --data-dir data --model resnet50 --pretrained --lr 3e-4 --epochs 20
# big dataset / multi-GPU -> ViT from scratch with DDP + torch.compile
torchrun --nproc_per_node=4 train.py --model vit_small --compile --epochs 100 --accum-steps 2
python train.py --resume            # continue after a crash
python export.py --ckpt runs/exp1/model.pt   # ONNX with calibration baked in
```
Outputs in `runs/exp1/`: `model.pt` (serving artifact), `best.pt`, `last.pt`, `report.json`
(macro-F1, per-class recall, confusion matrix, ECE, temperature), `log.csv` (+ TensorBoard if installed).

## Serve / deploy
`uvicorn serve.main:app` · `docker build -t vision-pro . && docker run -p 8000:8000 -v $PWD/runs:/app/runs vision-pro`

Env: `MODEL_PATH`, `DEVICE`, `MAX_BATCH` (16), `MAX_WAIT_MS` (8), `UNCERTAIN_BELOW` (0.6).
Endpoints: `/` UI · `POST /api/predict?explain=true` · `/health` · `/metrics` · `/docs`.

## What makes it production-grade
| Area | What's inside |
|---|---|
| Models | ViT written from scratch (fused SDPA attention, stochastic depth), or torchvision ResNet50 / ConvNeXt-T |
| Scale | DDP via torchrun, bf16/fp16 AMP, torch.compile, gradient accumulation, resume |
| Optimisation | AdamW (no WD on norm/bias), warmup+cosine, grad-clip, EMA, early stopping |
| Regularisation | MixUp/CutMix soft targets, label smoothing, TrivialAugment, RandomErasing, class-weighted loss |
| Trustworthiness | Temperature scaling, ECE, macro-F1, Grad-CAM for CNN **and** ViT, uncertainty flag |
| Serving | Async dynamic micro-batching, bounded uploads, decompression-bomb guard, /metrics (p50/p95, mean batch) |
| Engineering | pytest suite incl. API integration test, Dockerfile (non-root, healthcheck), GitHub Actions CI, ONNX export |
| Frontend | Server-rendered Jinja + pure CSS, no JavaScript |
