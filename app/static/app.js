const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const fileName = document.getElementById("file-name");
const runBtn = document.getElementById("run-btn");
const busy = document.getElementById("busy");
const errorBox = document.getElementById("error-box");
const findings = document.getElementById("findings");
const predBadge = document.getElementById("pred-badge");
const copyBtn = document.getElementById("copy-btn");

let selectedFile = null;
let selectedSample = null;

function showError(message) {
  errorBox.hidden = !message;
  errorBox.textContent = message || "";
}

function setBusy(on) {
  busy.hidden = !on;
  runBtn.disabled = on;
}

function setSlot(name, b64) {
  const el = document.querySelector(`[data-slot="${name}"]`);
  el.replaceChildren();
  if (!b64) {
    el.classList.add("empty");
    el.textContent = name;
    return;
  }
  el.classList.remove("empty");
  const img = document.createElement("img");
  img.alt = name;
  img.src = `data:image/png;base64,${b64}`;
  el.appendChild(img);
}

function renderProbs(probabilities) {
  const root = document.getElementById("probs");
  root.replaceChildren();
  const heading = document.createElement("h3");
  heading.textContent = "Class scores";
  root.appendChild(heading);
  Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .forEach(([name, value]) => {
      const pct = Math.round(value * 1000) / 10;
      const row = document.createElement("div");
      row.className = "prob-row";
      const label = document.createElement("span");
      label.textContent = name;
      const bar = document.createElement("div");
      bar.className = "bar";
      const fill = document.createElement("span");
      fill.style.width = `${pct}%`;
      bar.appendChild(fill);
      const score = document.createElement("span");
      score.textContent = `${pct.toFixed(1)}%`;
      row.append(label, bar, score);
      root.appendChild(row);
    });
}

async function loadHealth() {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error("Could not load pipeline status");
  const data = await res.json();
  document.getElementById("device-label").textContent = data.device;
  const dual = data.checkpoints.classification.dual;
  const full = data.checkpoints.classification.full;
  const seg = Object.entries(data.checkpoints.segmentation)
    .filter(([, present]) => present)
    .map(([name]) => name);
  document.getElementById("status-label").textContent = seg.length
    ? `ready (${seg.join(", ")})`
    : "missing checkpoints";
  const box = document.getElementById("samples");
  box.replaceChildren();
  data.samples.forEach((sample) => {
    if (!sample.available) return;
    const btn = document.createElement("button");
    btn.type = "button";
    const img = document.createElement("img");
    img.src = `/api/samples/${encodeURIComponent(sample.id)}`;
    img.alt = "";
    const title = document.createElement("div");
    title.textContent = sample.label;
    const hint = document.createElement("small");
    hint.textContent = "sample";
    btn.append(img, title, hint);
    btn.addEventListener("click", () => {
      selectedFile = null;
      selectedSample = sample.id;
      fileInput.value = "";
      fileName.textContent = `Sample: ${sample.label}`;
      box.querySelectorAll("button").forEach((el) => el.classList.toggle("active", el === btn));
    });
    box.appendChild(btn);
  });
  const clsMode = document.getElementById("cls-mode");
  clsMode.querySelector('option[value="dual"]').disabled = !dual;
  clsMode.querySelector('option[value="full"]').disabled = !full;
  if (!full && dual) clsMode.value = "dual";
}

async function runPipeline() {
  showError("");
  if (!selectedFile && !selectedSample) {
    showError("Choose an MRI file or a sample slice first.");
    return;
  }
  const body = new FormData();
  body.set("seg_model", document.getElementById("seg-model").value);
  body.set("classifier_mode", document.getElementById("cls-mode").value);
  if (selectedFile) body.set("file", selectedFile);
  if (selectedSample && !selectedFile) body.set("sample_id", selectedSample);

  setBusy(true);
  try {
    const res = await fetch("/api/analyze", { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = data.detail;
      const message = typeof detail === "string" ? detail : "Pipeline failed";
      throw new Error(message);
    }
    document.getElementById("result-kicker").textContent = data.filename;
    document.getElementById("result-title").textContent =
      `${data.prediction} · ${(data.confidence * 100).toFixed(1)}% model score`;
    predBadge.hidden = false;
    predBadge.textContent = data.prediction;
    predBadge.classList.toggle("ok", data.prediction === "notumor");
    setSlot("original", data.images.original);
    setSlot("segmentation", data.images.segmentation);
    setSlot("crop", data.images.crop);
    setSlot("gradcam", data.images.gradcam);
    document.getElementById("explanation-title").textContent = data.explanation_method;
    renderProbs(data.probabilities);
    document.getElementById("report-text").textContent = data.report;
    const seg = data.model_details?.segmentation;
    const cls = data.model_details?.classification;
    const gate = data.classifier_mode === "dual" ? data.classifier_gate : null;
    const extras = [
      seg ? `seg=${seg.architecture} (${seg.checkpoint})` : `seg=${data.segmentation_model}`,
      cls ? `clf=${cls.architecture} (${cls.checkpoint})` : `clf=${data.classifier_mode}`,
      gate ? `gate=${gate.prediction} ${(gate.score * 100).toFixed(1)}%` : null,
      data.stage_iou == null ? null : `seg/attention overlap=${data.stage_iou.toFixed(3)}`,
      `mask=${(data.mask_fraction * 100).toFixed(2)}%`,
      ...(data.notes || []),
    ].filter(Boolean);
    document.getElementById("footnotes").textContent = extras.join(" · ");
    findings.hidden = false;
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    setBusy(false);
  }
}

["dragenter", "dragover"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  });
});
["dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  });
});
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) {
    selectedFile = file;
    selectedSample = null;
    fileName.textContent = file.name;
  }
});
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  selectedFile = file || null;
  selectedSample = null;
  fileName.textContent = file ? file.name : "No file selected";
});
runBtn.addEventListener("click", runPipeline);
copyBtn.addEventListener("click", async () => {
  const text = document.getElementById("report-text").textContent;
  try {
    await navigator.clipboard.writeText(text);
    copyBtn.textContent = "Copied";
  } catch {
    copyBtn.textContent = "Copy failed";
  }
  setTimeout(() => {
    copyBtn.textContent = "Copy";
  }, 1600);
});

loadHealth().catch((err) => showError(err.message));
