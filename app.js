const MODEL_URL = "model.onnx";
const META_URL = "model_meta.json";
const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

ort.env.wasm.wasmPaths = ORT_CDN;
ort.env.wasm.numThreads = 1;

const el = (id) => document.getElementById(id);
const drop = el("drop"), fileInput = el("file"), predictBtn = el("predict"),
      clearBtn = el("clear"), statusEl = el("status"), preview = el("preview"),
      previewImg = el("previewImg"), results = el("results"),
      verdict = el("verdict"), bars = el("bars");

let meta = null, session = null, currentFile = null, currentImage = null;

el("repo").href = `https://github.com/${location.hostname.split(".")[0]}/4b-chart-predictor`;

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("err", isError);
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} loading ${url}`);
  const total = Number(res.headers.get("content-length") || 0);
  if (!res.body || !total) return await res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received / total);
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { out.set(c, offset); offset += c.length; }
  return out.buffer;
}

async function init() {
  try {
    setStatus("Loading model metadata…");
    meta = await (await fetch(META_URL, { cache: "no-cache" })).json();

    setStatus("Downloading model (0%)…");
    const buf = await fetchWithProgress(MODEL_URL, (p) => {
      setStatus(`Downloading model (${Math.round(p * 100)}%)…`);
    });

    setStatus("Starting inference engine…");
    session = await ort.InferenceSession.create(buf, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    setStatus("Ready — choose a chart image.");
  } catch (err) {
    setStatus(`Could not load the model: ${err.message}`, true);
  }
}

function preprocess(img) {
  const size = meta.image_size;
  const canvas = document.createElement("canvas");
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, size, size);
  const { data } = ctx.getImageData(0, 0, size, size);
  const mean = meta.mean, std = meta.std;
  const out = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    out[p] = (data[i] / 255 - mean[0]) / std[0];
    out[plane + p] = (data[i + 1] / 255 - mean[1]) / std[1];
    out[2 * plane + p] = (data[i + 2] / 255 - mean[2]) / std[2];
  }
  return new ort.Tensor("float32", out, [1, 3, size, size]);
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function render(probs) {
  const entries = meta.classes.map((name, i) => [name, probs[i]])
    .sort((a, b) => b[1] - a[1]);
  const top = entries[0];
  verdict.innerHTML = `Model estimate: <b>${top[0]}</b> · ${(top[1] * 100).toFixed(1)}% confidence`;
  const colors = { up: "var(--up)", flat: "var(--flat)", down: "var(--down)" };
  bars.innerHTML = "";
  for (const [name, p] of entries) {
    const row = document.createElement("div");
    row.className = "bar";
    row.innerHTML = `<span class="label">${name}</span>
      <span class="track"><span class="fill" style="background:${colors[name] || "var(--accent)"}"></span></span>
      <span class="val">${(p * 100).toFixed(1)}%</span>`;
    bars.appendChild(row);
    requestAnimationFrame(() => { row.querySelector(".fill").style.width = `${p * 100}%`; });
  }
  results.style.display = "block";
}

async function predict() {
  if (!session || !currentImage) return;
  predictBtn.disabled = true;
  setStatus("Running inference…");
  try {
    const input = preprocess(currentImage);
    const out = await session.run({ input });
    render(softmax(Array.from(out.logits.data)));
    setStatus("Done.");
  } catch (err) {
    setStatus(`Inference failed: ${err.message}`, true);
  } finally {
    predictBtn.disabled = !currentFile || !session;
  }
}

function loadFile(file) {
  if (!file) return;
  currentFile = file;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    currentImage = img;
    previewImg.src = url;
    preview.style.display = "block";
    predictBtn.disabled = !session;
    clearBtn.disabled = false;
    results.style.display = "none";
    setStatus(session ? "Ready — press Predict." : "Waiting for the model to load…");
  };
  img.onerror = () => setStatus("That file is not a readable image.", true);
  img.src = url;
}

function clearAll() {
  currentFile = null; currentImage = null;
  fileInput.value = "";
  preview.style.display = "none";
  results.style.display = "none";
  predictBtn.disabled = true;
  clearBtn.disabled = true;
  setStatus(session ? "Ready — choose a chart image." : "Loading model…");
}

drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault(); drop.classList.remove("over");
  loadFile(e.dataTransfer.files?.[0]);
});
fileInput.addEventListener("change", () => loadFile(fileInput.files?.[0]));
predictBtn.addEventListener("click", predict);
clearBtn.addEventListener("click", clearAll);

init();
