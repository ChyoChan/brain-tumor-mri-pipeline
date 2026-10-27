const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const runBtn = document.getElementById("run-btn");
const runLabel = document.getElementById("run-label");
const runHint = document.getElementById("run-hint");
const busy = document.getElementById("busy");
const errorBox = document.getElementById("error-box");
const findings = document.getElementById("findings");
const resultEmpty = document.getElementById("result-empty");
const resultState = document.getElementById("result-state");
const predBadge = document.getElementById("pred-badge");
const copyBtn = document.getElementById("copy-btn");
const focusFrame = document.getElementById("focus-frame");
const segSelect = document.getElementById("seg-model");
const clsSelect = document.getElementById("cls-mode");
const reportToggle = document.getElementById("generate-report");
const sampleBrowser = document.getElementById("sample-browser");
const browseBtn = document.getElementById("browse-samples");
const sampleClass = document.getElementById("sample-class");
const sampleSearch = document.getElementById("sample-search");
const browserStatus = document.getElementById("browser-status");
const browserGrid = document.getElementById("browser-grid");
const previousSamples = document.getElementById("previous-samples");
const nextSamples = document.getElementById("next-samples");

const classLabels = {
  glioma: "Glioma",
  meningioma: "Meningioma",
  pituitary: "Pituitary",
  notumor: "No tumor",
};
const views = {
  original: { title: "Original scan", description: "The MRI image supplied to the pipeline." },
  segmentation: { title: "Segmentation", description: "Highlighted pixels are the region predicted by the segmenter." },
  crop: { title: "Region crop", description: "The predicted region, cropped with surrounding context." },
  gradcam: { title: "Model attention", description: "Warmer colors indicate stronger model attention." },
};
const imageSources = new Map();
let selectedFile = null;
let selectedSample = null;
let activeView = "original";
let previewUrl = null;
let isBusy = false;
let pipelineReady = false;
let reportText = "";
let resultAvailable = false;
let copyTimer = null;
let galleryPage = 1;
let galleryPages = 0;
let galleryLoading = false;
let galleryController = null;
let searchTimer = null;

function showError(message) {
  errorBox.hidden = !message;
  errorBox.textContent = message || "";
}

function updateRunState() {
  const hasInput = Boolean(selectedFile || selectedSample);
  runBtn.disabled = isBusy || !hasInput || !pipelineReady;
  runLabel.textContent = isBusy ? "Analyzing…" : "Analyze scan";
  runHint.hidden = isBusy;
  runHint.textContent = !hasInput
    ? "Choose a scan to begin."
    : !pipelineReady
      ? "Waiting for the local pipeline."
      : "Your image stays on this computer.";
}

function setBusy(on) {
  isBusy = on;
  busy.hidden = !on;
  fileInput.disabled = on;
  segSelect.disabled = on;
  clsSelect.disabled = on;
  reportToggle.disabled = on;
  browseBtn.disabled = on;
  document.querySelectorAll("#samples button").forEach((button) => { button.disabled = on; });
  document.getElementById("reading-surface").setAttribute("aria-busy", String(on));
  resultState.textContent = on ? "Analyzing" : resultAvailable ? "Complete" : "Ready to analyze";
  updateRunState();
}

function setSlot(name, src) {
  const slot = document.querySelector(`[data-slot="${name}"]`);
  const button = document.querySelector(`[data-view="${name}"]`);
  slot.replaceChildren();
  slot.classList.toggle("empty", !src);
  button.disabled = !src;
  if (!src) {
    imageSources.delete(name);
    return;
  }
  imageSources.set(name, src);
  const img = document.createElement("img");
  img.alt = "";
  img.src = src;
  slot.appendChild(img);
}

function selectView(name) {
  const src = imageSources.get(name);
  if (!src) return;
  activeView = name;
  document.querySelectorAll("[data-view]").forEach((button) => {
    const active = button.dataset.view === name;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  document.getElementById("active-view-title").textContent = views[name].title;
  document.getElementById("view-description").textContent = views[name].description;
  document.getElementById("view-size").textContent = "";
  const img = document.createElement("img");
  img.alt = `${views[name].title} of the selected MRI`;
  img.addEventListener("load", () => {
    if (img.parentNode === focusFrame) {
      document.getElementById("view-size").textContent = `${img.naturalWidth} × ${img.naturalHeight} px`;
    }
  });
  img.addEventListener("error", () => {
    if (img.parentNode !== focusFrame) return;
    const message = document.createElement("p");
    message.className = "viewer-empty";
    message.textContent = "This image cannot be previewed in your browser. Analyze the scan to load its processed view.";
    focusFrame.replaceChildren(message);
  });
  img.src = src;
  focusFrame.replaceChildren(img);
}

function clearResults() {
  resultAvailable = false;
  findings.hidden = true;
  resultEmpty.hidden = false;
  resultState.textContent = "Ready to analyze";
  reportText = "";
  document.getElementById("report-text").replaceChildren();
  document.getElementById("copy-label").textContent = "Copy";
  clearTimeout(copyTimer);
  views.gradcam.title = "Model attention";
  views.crop.description = "The predicted region, cropped with surrounding context.";
  ["segmentation", "crop", "gradcam"].forEach((name) => setSlot(name, null));
  document.querySelector(".model-details").open = false;
  showError("");
}

function releasePreview() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
}

function selectSample(sample) {
  if (isBusy) return;
  releasePreview();
  selectedFile = null;
  selectedSample = sample.id;
  fileInput.value = "";
  clearResults();
  document.getElementById("result-kicker").textContent = sample.filename || `${sample.label} sample`;
  document.querySelectorAll("[data-sample-id]").forEach((item) => {
    const selected = item.dataset.sampleId === sample.id;
    item.classList.toggle("active", selected);
    item.setAttribute("aria-pressed", String(selected));
  });
  setSlot("original", `/api/samples/${encodeURIComponent(sample.id)}`);
  selectView("original");
  updateRunState();
}

function selectFile(file) {
  if (!file || isBusy) return;
  const supported = /\.(png|jpe?g|tiff?)$/i.test(file.name);
  if (!supported) {
    showError("Choose a PNG, JPG or TIFF image.");
    fileInput.value = "";
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showError("This image exceeds 20 MB. Choose a smaller image.");
    fileInput.value = "";
    return;
  }
  releasePreview();
  selectedFile = file;
  selectedSample = null;
  clearResults();
  document.querySelectorAll("[data-sample-id]").forEach((button) => {
    button.classList.remove("active");
    button.setAttribute("aria-pressed", "false");
  });
  document.getElementById("result-kicker").textContent = file.name;
  previewUrl = URL.createObjectURL(file);
  setSlot("original", previewUrl);
  selectView("original");
  updateRunState();
}

function renderProbs(probabilities) {
  const root = document.getElementById("probs");
  root.replaceChildren();
  const heading = document.createElement("h3");
  heading.textContent = "Class scores";
  root.appendChild(heading);
  Object.entries(probabilities).sort((a, b) => b[1] - a[1]).forEach(([name, value]) => {
    const pct = Math.max(0, Math.min(100, value * 100));
    const row = document.createElement("div");
    row.className = "prob-row";
    const label = document.createElement("span");
    label.textContent = classLabels[name] || name;
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.setAttribute("aria-hidden", "true");
    const fill = document.createElement("span");
    fill.style.width = `${pct}%`;
    bar.appendChild(fill);
    const score = document.createElement("span");
    score.textContent = `${pct.toFixed(1)}%`;
    row.append(label, bar, score);
    root.appendChild(row);
  });
}

function renderReport(text) {
  const root = document.getElementById("report-text");
  root.replaceChildren();
  reportText = typeof text === "string" ? text : "";
  const hasReport = Boolean(reportText);
  document.getElementById("report-panel").hidden = !hasReport;
  document.getElementById("report-skipped").hidden = hasReport;
  copyBtn.disabled = !hasReport;
  document.getElementById("copy-label").textContent = "Copy";
  clearTimeout(copyTimer);
  if (!hasReport) return;
  const sections = text.split(/(?:^|\n)\s*(Findings|Impression|Recommendation):\s*/i);
  if (sections[0].trim()) {
    const paragraph = document.createElement("p");
    paragraph.textContent = sections[0].trim();
    root.appendChild(paragraph);
  }
  for (let index = 1; index < sections.length; index += 2) {
    const heading = document.createElement("h4");
    heading.textContent = sections[index];
    const paragraph = document.createElement("p");
    paragraph.textContent = (sections[index + 1] || "").trim();
    root.append(heading, paragraph);
  }
}

function renderDetails(data) {
  const root = document.getElementById("analysis-details");
  root.replaceChildren();
  const seg = data.model_details?.segmentation;
  const cls = data.model_details?.classification;
  const entries = [
    ["Segmentation", seg ? `${seg.architecture} (${seg.checkpoint})` : data.segmentation_model],
    ["Classification", cls ? `${cls.architecture} (${cls.checkpoint})` : data.classifier_mode],
    ["Mask coverage", `${(data.mask_fraction * 100).toFixed(2)}% of image pixels`],
    ["Explanation", data.explanation_method || "Grad-CAM"],
    ["Region / attention overlap", data.stage_iou == null ? "Not available" : data.stage_iou.toFixed(3)],
  ];
  if (data.classifier_mode === "dual" && data.classifier_gate) {
    entries.push(["Full-image gate", `${classLabels[data.classifier_gate.prediction] || data.classifier_gate.prediction} (${(data.classifier_gate.score * 100).toFixed(1)}%)`]);
  }
  entries.push(["Report", data.report_generated === false ? "Skipped" : "Template summary"]);
  entries.forEach(([label, value]) => {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    term.textContent = label;
    const detail = document.createElement("dd");
    detail.textContent = value;
    row.append(term, detail);
    root.appendChild(row);
  });
  const notes = document.getElementById("analysis-notes");
  notes.replaceChildren();
  (data.notes || []).forEach((note) => {
    const paragraph = document.createElement("p");
    paragraph.textContent = note;
    notes.appendChild(paragraph);
  });
}

async function loadHealth() {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error("The local pipeline is unavailable. Restart the server and reload this page.");
  const data = await res.json();
  const devices = { mps: "Apple GPU", cuda: "NVIDIA GPU", cpu: "CPU" };
  document.getElementById("device-label").textContent = devices[data.device] || data.device;
  const classification = data.checkpoints.classification;
  Array.from(segSelect.options).forEach((option) => { option.disabled = !data.checkpoints.segmentation[option.value]; });
  Array.from(clsSelect.options).forEach((option) => { option.disabled = !classification[option.value]; });
  if (segSelect.selectedOptions[0].disabled) {
    const available = Array.from(segSelect.options).find((option) => !option.disabled);
    if (available) segSelect.value = available.value;
  }
  if (clsSelect.selectedOptions[0].disabled) {
    const available = Array.from(clsSelect.options).find((option) => !option.disabled);
    if (available) clsSelect.value = available.value;
  }
  pipelineReady = Object.values(data.checkpoints.segmentation).some(Boolean) && Object.values(classification).some(Boolean);
  document.getElementById("status-label").textContent = pipelineReady ? "Pipeline ready" : "Models missing";
  document.getElementById("connection").classList.toggle("ready", pipelineReady);
  const box = document.getElementById("samples");
  box.replaceChildren();
  const availableSamples = data.samples.filter((sample) => sample.available);
  availableSamples.forEach((sample) => {
    box.appendChild(createSampleButton(sample));
  });
  document.querySelector(".section-heading > span").textContent = `${availableSamples.length} classes`;
  if (!availableSamples.length) {
    const message = document.createElement("p");
    message.className = "muted";
    message.textContent = "Sample images are not installed. Upload an MRI to begin.";
    box.appendChild(message);
  }
  if (!pipelineReady) showError("The pipeline needs trained model weights. Add the checkpoints to the models folder and reload.");
  updateRunState();
}

function createSampleButton(sample, fromGallery = false) {
  const button = document.createElement("button");
  button.type = "button";
  button.dataset.sampleId = sample.id;
  button.setAttribute("aria-label", `Select ${sample.label} ${sample.filename || "sample"}`);
  button.setAttribute("aria-pressed", String(selectedSample === sample.id));
  button.classList.toggle("active", selectedSample === sample.id);
  const img = document.createElement("img");
  img.src = `/api/samples/${encodeURIComponent(sample.id)}`;
  img.alt = ""; // The button's name describes this selectable thumbnail.
  img.loading = "lazy";
  const title = document.createElement("span");
  title.className = "sample-name";
  const label = document.createElement("span");
  label.textContent = sample.label;
  const check = document.createElement("span");
  check.className = "sample-check";
  check.textContent = "✓";
  check.setAttribute("aria-hidden", "true");
  title.append(label, check);
  button.append(img, title);
  if (sample.filename) {
    const filename = document.createElement("span");
    filename.className = "sample-filename";
    filename.textContent = sample.filename;
    button.appendChild(filename);
  }
  button.addEventListener("click", () => {
    if (isBusy) return;
    selectSample(sample);
    if (fromGallery) sampleBrowser.close();
  });
  return button;
}

function updateGalleryButtons() {
  previousSamples.disabled = galleryLoading || galleryPage <= 1;
  nextSamples.disabled = galleryLoading || galleryPage >= galleryPages;
}

async function loadGallery(page = 1) {
  galleryController?.abort();
  const controller = new AbortController();
  galleryController = controller;
  galleryLoading = true;
  browserGrid.replaceChildren();
  browserGrid.setAttribute("aria-busy", "true");
  browserStatus.textContent = "Loading local images…";
  document.getElementById("sample-page").textContent = "";
  updateGalleryButtons();
  const params = new URLSearchParams({ page: String(page), page_size: "8", search: sampleSearch.value.trim() });
  if (sampleClass.value) params.set("class_name", sampleClass.value);
  try {
    const response = await fetch(`/api/samples?${params}`, { signal: controller.signal });
    if (!response.ok) throw new Error("Could not load local images. Check the server and try browsing again.");
    const data = await response.json();
    if (controller !== galleryController) return;
    galleryPage = data.page;
    galleryPages = data.pages;
    data.items.forEach((sample) => browserGrid.appendChild(createSampleButton(sample, true)));
    const first = data.total ? (data.page - 1) * data.page_size + 1 : 0;
    const last = data.total ? first + data.items.length - 1 : 0;
    browserStatus.textContent = data.total
      ? `Showing ${first}–${last} of ${data.total} local images.`
      : "No matching images. Change the filters or upload an MRI from your computer.";
    document.getElementById("sample-page").textContent = data.pages ? `Page ${data.page} of ${data.pages}` : "No images";
  } catch (error) {
    if (error.name === "AbortError" || controller !== galleryController) return;
    galleryPages = 0;
    browserStatus.textContent = error.message;
  } finally {
    if (controller === galleryController) {
      galleryLoading = false;
      browserGrid.setAttribute("aria-busy", "false");
      updateGalleryButtons();
    }
  }
}

async function runPipeline() {
  if (isBusy || !pipelineReady || (!selectedFile && !selectedSample)) return;
  showError("");
  const body = new FormData();
  body.set("seg_model", segSelect.value);
  body.set("classifier_mode", clsSelect.value);
  body.set("generate_report", String(reportToggle.checked));
  if (selectedFile) body.set("file", selectedFile);
  else body.set("sample_id", selectedSample);

  setBusy(true);
  try {
    const res = await fetch("/api/analyze", { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Analysis failed. Check the server output and try again.");
    document.getElementById("result-kicker").textContent = data.filename;
    document.getElementById("result-title").textContent = classLabels[data.prediction] || data.prediction;
    document.getElementById("model-score").textContent = `${(data.confidence * 100).toFixed(1)}%`;
    predBadge.hidden = false;
    predBadge.textContent = data.prediction === "notumor" ? "Predicted no-tumor class" : "Predicted tumor class";
    predBadge.classList.toggle("ok", data.prediction === "notumor");
    Object.entries(data.images).forEach(([name, b64]) => {
      if (views[name]) setSlot(name, b64 ? `data:image/png;base64,${b64}` : null);
    });
    views.gradcam.title = data.explanation_method || "Model attention";
    views.crop.description = data.crop_source === "full_image_gate"
      ? "The full image is used as the first view after the no-tumor gate."
      : data.crop_source === "not_used"
        ? "Region crop for inspection. The full-image classifier does not use this crop."
        : "The predicted region, cropped with surrounding context.";
    selectView(imageSources.has(activeView) ? activeView : "original");
    renderProbs(data.probabilities);
    renderReport(data.report);
    renderDetails(data);
    resultAvailable = true;
    findings.hidden = false;
    resultEmpty.hidden = true;
  } catch (err) {
    showError(err.message || "The server could not be reached. Restart it and try again.");
  } finally {
    setBusy(false);
  }
}

document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => selectView(button.dataset.view));
});
["dragenter", "dragover"].forEach((event) => {
  dropzone.addEventListener(event, (e) => {
    e.preventDefault();
    if (!isBusy) dropzone.classList.add("drag");
  });
});
["dragleave", "drop"].forEach((event) => {
  dropzone.addEventListener(event, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); });
});
dropzone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));
runBtn.addEventListener("click", runPipeline);
browseBtn.addEventListener("click", () => {
  if (isBusy) return;
  clearTimeout(searchTimer);
  sampleBrowser.showModal();
  loadGallery(galleryPage);
});
document.getElementById("close-browser").addEventListener("click", () => sampleBrowser.close());
sampleBrowser.addEventListener("close", () => {
  clearTimeout(searchTimer);
  galleryController?.abort();
  galleryController = null;
});
sampleClass.addEventListener("change", () => {
  clearTimeout(searchTimer);
  loadGallery(1);
});
sampleSearch.addEventListener("input", () => {
  clearTimeout(searchTimer);
  galleryController?.abort();
  galleryController = null;
  // Remove stale selectable results while a new search is being debounced.
  browserGrid.replaceChildren();
  galleryLoading = true;
  browserStatus.textContent = "Searching local images…";
  updateGalleryButtons();
  searchTimer = setTimeout(() => loadGallery(1), 250);
});
previousSamples.addEventListener("click", () => loadGallery(galleryPage - 1));
nextSamples.addEventListener("click", () => loadGallery(galleryPage + 1));
clsSelect.addEventListener("change", () => {
  document.getElementById("classifier-hint").textContent = clsSelect.value === "full"
    ? "Full image is the recommended baseline."
    : "Experimental: combines the region crop and full image with a no-tumor gate.";
});
copyBtn.addEventListener("click", async () => {
  if (!reportText) return;
  try {
    await navigator.clipboard.writeText(reportText);
    document.getElementById("copy-label").textContent = "Copied";
  } catch {
    document.getElementById("copy-label").textContent = "Copy failed";
  }
  clearTimeout(copyTimer);
  copyTimer = setTimeout(() => { document.getElementById("copy-label").textContent = "Copy"; }, 1800);
});
function setExpanded(expanded) {
  document.getElementById("reading-surface").classList.toggle("expanded", expanded);
  const button = document.getElementById("expand-btn");
  button.setAttribute("aria-expanded", String(expanded));
  button.setAttribute("aria-label", expanded ? "Collapse image viewer" : "Expand image viewer");
  document.querySelectorAll(".app-header, .scan-tray, .result-panel, .app-footer").forEach((element) => {
    element.inert = expanded;
  });
  document.body.style.overflow = expanded ? "hidden" : "";
  button.focus();
}
document.getElementById("expand-btn").addEventListener("click", () => {
  setExpanded(!document.getElementById("reading-surface").classList.contains("expanded"));
});
document.addEventListener("keydown", (event) => {
  const surface = document.getElementById("reading-surface");
  if (!surface.classList.contains("expanded")) return;
  if (event.key === "Escape") setExpanded(false);
  if (event.key === "Tab") {
    const buttons = Array.from(surface.querySelectorAll("button:not(:disabled)"));
    const first = buttons[0];
    const last = buttons[buttons.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }
});
window.addEventListener("pagehide", releasePreview);

loadHealth().catch((err) => {
  showError(err.message);
  document.getElementById("status-label").textContent = "Server unavailable";
  document.getElementById("connection").classList.add("error-state");
  updateRunState();
});
