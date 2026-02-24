from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from PIL import Image
import io
import logging
import os
import time
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import base64
from pydantic import BaseModel

app = FastAPI()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("captcha-ocr")

BEARER_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIyNjY4MzQ1MDAwIiwibmFtZSI6IkZBU1QgWUFaSUxJTSIsImFkbWluIjp0cnVlLCJpYXQiOjE3NTkwNjU3MzEsImV4cCI6MTc1OTA2OTMzMX0.eEfrHkwm3tamPT1ozf3A09wE8KBQDTnKW7JPqjZUAi8"
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "16"))
_rate_limit_window = 60.0
_rate_limit_state = {}

MODEL_PATH = "zai-org/GLM-OCR"
PROMPT = "Text Recognition:"

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype="auto"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
logger.info("GLM-OCR model loaded on device=%s", device)


def _require_bearer(auth_header: str | None):
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth_header.split(" ", 1)[1].strip()
    if token != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def _check_rate_limit(client_ip: str):
    if RATE_LIMIT_PER_MIN <= 0:
        return
    now = time.time()
    window_start = now - _rate_limit_window
    timestamps = _rate_limit_state.get(client_ip, [])
    timestamps = [t for t in timestamps if t >= window_start]
    if len(timestamps) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    timestamps.append(now)
    _rate_limit_state[client_ip] = timestamps


def _preprocess_image(content: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(content)).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255))
    combined = Image.alpha_composite(background, image).convert("RGB")
    return combined


class Base64Request(BaseModel):
    image_base64: str


def _predict_single(image: Image.Image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    inputs.pop("token_type_ids", None)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    gen_ids = out[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return text

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html>
<html lang="tr">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>OCR | Fast Yazılım</title>
    <style>
      :root {
        --bg: #0c0f14;
        --fg: #e7edf3;
        --muted: #94a3b8;
        --accent: #22d3ee;
        --accent-2: #a78bfa;
        --card: #121721;
        --stroke: rgba(255,255,255,0.08);
      }
      [data-theme="light"] {
        --bg: #f7f8fb;
        --fg: #0b1020;
        --muted: #5b6472;
        --accent: #0ea5e9;
        --accent-2: #22c55e;
        --card: #ffffff;
        --stroke: rgba(2,8,23,0.12);
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Sora", "Segoe UI", Arial, sans-serif;
        background: var(--bg);
        color: var(--fg);
      }
      body::before {
        content: "";
        position: fixed;
        inset: 0;
        background:
          linear-gradient(180deg, rgba(255,255,255,0.04), transparent 40%),
          repeating-linear-gradient(90deg, rgba(255,255,255,0.02) 0 1px, transparent 1px 24px);
        pointer-events: none;
        z-index: -1;
      }
      .wrap { max-width: 1100px; margin: 28px auto; padding: 18px; }
      .hero { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 16px; }
      .title { font-size: 28px; font-weight: 700; letter-spacing: 0.3px; }
      .subtitle { color: var(--muted); }
      .toggle {
        background: var(--card);
        border: 1px solid var(--stroke);
        border-radius: 999px;
        padding: 6px;
        display: inline-flex;
        gap: 6px;
        align-items: center;
      }
      .toggle button {
        border: none;
        padding: 6px 10px;
        border-radius: 999px;
        background: transparent;
        color: var(--muted);
        cursor: pointer;
        font-weight: 600;
      }
      .toggle button.active { background: var(--accent); color: #041016; }
      .panel {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent 60%), var(--card);
        border: 1px solid var(--stroke);
        border-radius: 18px;
        padding: 18px;
      }
      .drop {
        border: 1px dashed var(--stroke);
        border-radius: 16px;
        padding: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        background: rgba(255,255,255,0.02);
      }
      .drop-left { display: flex; flex-direction: column; gap: 6px; }
      .drop-title { font-weight: 700; }
      .drop-sub { color: var(--muted); font-size: 14px; }
      .actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
      .btn {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: #031018;
        border: none;
        padding: 10px 16px;
        border-radius: 12px;
        font-weight: 700;
        cursor: pointer;
      }
      .btn:disabled { opacity: 0.6; cursor: not-allowed; }
      .token {
        flex: 1 1 260px;
        min-width: 220px;
        padding: 10px 12px;
        border-radius: 12px;
        border: 1px solid var(--stroke);
        background: transparent;
        color: var(--fg);
      }
      input[type=file] { display: none; }
      .file-label {
        border: 1px solid var(--stroke);
        padding: 10px 14px;
        border-radius: 12px;
        cursor: pointer;
        font-weight: 600;
        color: var(--fg);
        background: rgba(255,255,255,0.03);
      }
      .status { color: var(--muted); }
      .grid {
        margin-top: 16px;
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
        gap: 14px;
      }
      .tile {
        border: 1px solid var(--stroke);
        border-radius: 14px;
        padding: 10px;
        background: rgba(255,255,255,0.02);
      }
      .thumb {
        height: 160px;
        border-radius: 10px;
        overflow: hidden;
        display:flex;
        align-items:center;
        justify-content:center;
        background: rgba(255,255,255,0.04);
      }
      .thumb img { max-width: 100%; max-height: 100%; display:block; }
      .out {
        margin-top: 8px;
        font-size: 14px;
        color: var(--fg);
        white-space: pre-wrap;
        word-break: break-word;
      }
      .meta { margin-top: 6px; color: var(--muted); font-size: 12px; }
      .spinner {
        display:inline-block; width:14px; height:14px;
        border:2px solid rgba(255,255,255,0.3);
        border-top-color: var(--accent);
        border-radius:50%; animation: spin 1s linear infinite;
        vertical-align: middle; margin-right: 8px;
      }
      @keyframes spin { to { transform: rotate(360deg); } }
      @media (max-width: 720px) {
        .hero { flex-direction: column; align-items: flex-start; }
        .drop { flex-direction: column; align-items: flex-start; }
      }
    </style>
  </head>
  <body data-theme="dark">
    <div class="wrap">
      <div class="hero">
        <div>
          <div class="title">OCR | Fast Yazılım</div>
          <div class="subtitle">Birden fazla görsel yükleyebilirsiniz. (Max: 10)</div>
        </div>
        <div class="toggle" role="group" aria-label="Tema">
          <button id="darkBtn" class="active">🌙</button>
          <button id="lightBtn">🌖</button>
        </div>
      </div>
      <div class="panel">
        <div class="drop" id="drop">
          <div class="drop-left">
            <div class="drop-title">Gorselleri birak veya sec</div>
            <div class="drop-sub">PNG, JPG. Birden fazla dosya desteklenir.</div>
          </div>
          <div class="actions">
            <label class="file-label" for="files">Dosya Sec</label>
            <input id="files" type="file" accept="image/*" multiple />
            <input id="token" class="token" type="password" placeholder="Bearer token" />
            <button id="btn" class="btn">OKU</button>
            <span class="status" id="status"></span>
          </div>
        </div>
        <div class="grid" id="grid"></div>
      </div>
    </div>
    <script>
      const filesInput = document.getElementById('files');
      const grid = document.getElementById('grid');
      const status = document.getElementById('status');
      const btn = document.getElementById('btn');
      const drop = document.getElementById('drop');
      const darkBtn = document.getElementById('darkBtn');
      const lightBtn = document.getElementById('lightBtn');

      function setTheme(theme) {
        document.body.setAttribute('data-theme', theme);
        if (theme === 'dark') { darkBtn.classList.add('active'); lightBtn.classList.remove('active'); }
        else { lightBtn.classList.add('active'); darkBtn.classList.remove('active'); }
      }
      darkBtn.addEventListener('click', () => setTheme('dark'));
      lightBtn.addEventListener('click', () => setTheme('light'));

      function clearGrid() { grid.innerHTML = ''; }

      function addTile(previewUrl, text, seconds, name) {
        const tile = document.createElement('div');
        tile.className = 'tile';
        const thumb = document.createElement('div');
        thumb.className = 'thumb';
        const img = document.createElement('img');
        img.src = previewUrl;
        thumb.appendChild(img);
        const out = document.createElement('div');
        out.className = 'out';
        const timeText = (seconds !== undefined && seconds !== null) ? ` (${seconds}s)` : '';
        out.textContent = (text || '-') + timeText;
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.textContent = name || '';
        tile.appendChild(thumb);
        tile.appendChild(out);
        tile.appendChild(meta);
        grid.appendChild(tile);
      }

      function previewFiles(files) {
        clearGrid();
        for (const file of files) {
          const reader = new FileReader();
          reader.onload = (e) => addTile(e.target.result, '...', null, file.name);
          reader.readAsDataURL(file);
        }
      }

      filesInput.addEventListener('change', () => {
        const files = Array.from(filesInput.files || []);
        if (!files.length) return;
        previewFiles(files);
      });

      drop.addEventListener('dragover', (e) => {
        e.preventDefault();
        drop.style.borderColor = 'var(--accent)';
      });
      drop.addEventListener('dragleave', () => {
        drop.style.borderColor = 'var(--stroke)';
      });
      drop.addEventListener('drop', (e) => {
        e.preventDefault();
        drop.style.borderColor = 'var(--stroke)';
        if (e.dataTransfer.files && e.dataTransfer.files.length) {
          filesInput.files = e.dataTransfer.files;
          previewFiles(Array.from(e.dataTransfer.files));
        }
      });

      btn.addEventListener('click', async () => {
        const files = Array.from(filesInput.files || []);
        if (!files.length) { alert('Lutfen en az bir gorsel secin.'); return; }
        const token = document.getElementById('token').value.trim();
        if (!token) { alert('Bearer token gerekli.'); return; }
        status.innerHTML = '<span class="spinner"></span>Isleniyor...';
        btn.disabled = true;
        const form = new FormData();
        files.forEach(f => form.append('files', f));
        const res = await fetch('/predict_batch', { method: 'POST', body: form, headers: { 'Authorization': `Bearer ${token}` } });
        if (!res.ok) {
          status.textContent = 'Hata';
          btn.disabled = false;
          return;
        }
        const data = await res.json();
        const texts = data.texts || [];
        const secs = data.seconds || [];
        clearGrid();
        files.forEach((file, idx) => {
          const reader = new FileReader();
          reader.onload = (e) => addTile(e.target.result, texts[idx] || '-', secs[idx], file.name);
          reader.readAsDataURL(file);
        });
        status.textContent = '';
        btn.disabled = false;
      });
    </script>
  </body>
</html>
"""


@app.post("/predict")
async def predict(
    request: Request,
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    _require_bearer(authorization)
    _check_rate_limit(request.client.host)

    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            raise HTTPException(status_code=400, detail="Missing file in multipart form")
        if hasattr(upload, "read"):
            content = await upload.read()
        else:
            content = bytes(upload)
    else:
        try:
            payload = await request.json()
            b64 = payload.get("image_base64", "") or payload.get("image", "")
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            content = base64.b64decode(b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Provide multipart file or JSON image/image_base64")

    image = _preprocess_image(content)
    start = time.perf_counter()
    text = _predict_single(image)
    seconds = time.perf_counter() - start
    return {"text": text, "seconds": round(seconds, 3)}


@app.post("/predict_batch")
async def predict_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    _require_bearer(authorization)
    _check_rate_limit(request.client.host)
    if len(files) > MAX_BATCH:
        raise HTTPException(status_code=400, detail="Batch too large")

    texts = []
    seconds_list = []
    for f in files:
        content = await f.read()
        image = _preprocess_image(content)
        start = time.perf_counter()
        text = _predict_single(image)
        seconds = time.perf_counter() - start
        texts.append(text)
        seconds_list.append(round(seconds, 3))
    return {"texts": texts, "seconds": seconds_list}
