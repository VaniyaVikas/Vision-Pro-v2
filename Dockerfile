FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY core ./core
COPY serve ./serve
RUN useradd -m app && chown -R app /app
USER app
ENV MODEL_PATH=/app/runs/exp1/model.pt
EXPOSE 8000
HEALTHCHECK --interval=30s CMD python -c "import urllib.request as u;u.urlopen('http://localhost:8000/health')"
# one worker: the model lives in memory and dynamic batching works per process
CMD ["uvicorn", "serve.main:app", "--host", "0.0.0.0", "--port", "8000"]
