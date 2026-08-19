// Planogrid dashboard — vanilla JS, no build step, no framework (Chart.js
// is the one CDN dependency, used for the radar comparison and the
// interactive per-epoch training curves).

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const samplesRow = document.getElementById("samplesRow");
const statusGrid = document.getElementById("statusGrid");

const resultsEmpty = document.getElementById("resultsEmpty");
const resultsLoading = document.getElementById("resultsLoading");
const resultsContent = document.getElementById("resultsContent");
const resultImage = document.getElementById("resultImage");
const confidenceList = document.getElementById("confidenceList");

const stageBackdrop = document.getElementById("stageBackdrop");
const stageItems = Array.from(document.querySelectorAll(".stage-item"));
const stageProgressFill = document.getElementById("stageProgressFill");

const insightsCard = document.getElementById("insightsCard");
const insightsNarrative = document.getElementById("insightsNarrative");
const insightsNumeric = document.getElementById("insightsNumeric");
const insights3d = document.getElementById("insights3d");

const zoomCard = document.getElementById("zoomCard");
const zoomRow = document.getElementById("zoomRow");
const lightboxBackdrop = document.getElementById("lightboxBackdrop");
const lightboxImage = document.getElementById("lightboxImage");
const lightboxCaption = document.getElementById("lightboxCaption");
const lightboxCrops = document.getElementById("lightboxCrops");
const lightboxClose = document.getElementById("lightboxClose");

const tileProduct = document.getElementById("tileProductValue");
const tileGap = document.getElementById("tileGapValue");
const tileTag = document.getElementById("tileTagValue");
const tileLatency = document.getElementById("tileLatencyValue");

const TASK_LABEL = { product: "Product detector", empty_gap: "Gap detector", price_tag: "Tag detector" };
const TASK_DIR = { product: "product", empty_gap: "gap", price_tag: "tag" };
const TASK_COLOR = { product: "#F59E0B", empty_gap: "#3B82F6", price_tag: "#14B8A6" };
const TASK_ORDER = ["product", "empty_gap", "price_tag"];

let radarChartInstance = null;
const curveChartInstances = {};
let lastImageBlob = null;

function showResultsState(state) {
  resultsEmpty.hidden = state !== "empty";
  resultsLoading.hidden = state !== "loading";
  resultsContent.hidden = state !== "content";
}

// Presentational only — the four stages are real pipeline stages (see
// inference/pipeline.py + CompositeDetector), but their pacing here is a
// fixed per-stage dwell, not a live progress feed from the backend (there's
// no streaming endpoint). The real /predict call runs concurrently; the
// stage sequence and the fetch are joined via Promise.all before revealing
// results, so this never claims a timing the model didn't actually take.
const STAGE_DELAY_MS = 480;

function resetStages() {
  stageItems.forEach((el) => el.classList.remove("is-active", "is-done"));
  stageProgressFill.style.width = "0%";
}

async function playStages() {
  resetStages();
  for (let i = 0; i < stageItems.length; i++) {
    stageItems[i].classList.add("is-active");
    stageProgressFill.style.width = `${((i + 1) / stageItems.length) * 100}%`;
    await new Promise((resolve) => setTimeout(resolve, STAGE_DELAY_MS));
    stageItems[i].classList.remove("is-active");
    stageItems[i].classList.add("is-done");
  }
}

function showStageModal() {
  stageBackdrop.hidden = false;
  requestAnimationFrame(() => stageBackdrop.classList.add("is-visible"));
}

function hideStageModal() {
  stageBackdrop.classList.remove("is-visible");
  setTimeout(() => {
    stageBackdrop.hidden = true;
  }, 300);
}

async function runDetection(fileOrBlob, filename) {
  showResultsState("loading");
  insightsCard.hidden = true;
  lastImageBlob = fileOrBlob;
  showStageModal();

  const formData = new FormData();
  formData.append("file", fileOrBlob, filename || "upload.jpg");

  try {
    const [response] = await Promise.all([
      fetch("/predict", { method: "POST", body: formData }),
      playStages(),
    ]);
    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }
    const data = await response.json();
    hideStageModal();
    renderResult(data);
  } catch (err) {
    hideStageModal();
    resultsEmpty.querySelector("p").textContent = `Detection failed: ${err.message}`;
    showResultsState("empty");
  }
}

function renderResult(data) {
  resultImage.src = data.annotated_image;
  showResultsState("content");

  const detections = data.analysis.detections || [];
  const summary = data.analysis.summary || {};
  const product = summary.product ?? 0;
  const gap = summary.empty_gap ?? 0;
  const tag = summary.price_tag ?? 0;

  tileProduct.textContent = product;
  tileGap.textContent = gap;
  tileTag.textContent = tag;
  tileLatency.textContent = `${data.latency_ms.toFixed(0)} ms`;

  renderConfidenceList(detections);
  renderInsights({ product, gap, tag, detections, latencyMs: data.latency_ms });
  renderDetectorZooms(detections);
}

function renderConfidenceList(detections) {
  confidenceList.innerHTML = "";
  if (!detections.length) {
    confidenceList.innerHTML = `<p class="status-pending-note">No detections above the confidence threshold.</p>`;
    return;
  }

  // A shelf photo can have 100+ products but only a handful of gaps/tags —
  // sorting by confidence alone lets the most common class swamp the list
  // with near-duplicate rows. Show a capped, balanced sample per class
  // instead, so every detected class is actually visible.
  const PER_CLASS_LIMIT = 4;
  const byClass = { empty_gap: [], price_tag: [], product: [] };
  detections.forEach((d) => {
    if (byClass[d.class_name]) byClass[d.class_name].push(d);
  });

  for (const cls of ["empty_gap", "price_tag", "product"]) {
    const all = byClass[cls];
    if (!all.length) continue;
    const top = [...all].sort((a, b) => b.confidence - a.confidence).slice(0, PER_CLASS_LIMIT);

    const header = document.createElement("div");
    header.className = "confidence-group-header";
    const label = cls.replace("_", " ");
    header.textContent =
      all.length > top.length
        ? `${label} - top ${top.length} of ${all.length}`
        : `${label} - ${all.length}`;
    confidenceList.appendChild(header);

    top.forEach((det) => {
      const color = TASK_COLOR[det.class_name] || "#9c9fa8";
      const pct = Math.round(det.confidence * 100);
      const row = document.createElement("div");
      row.className = "confidence-row";
      row.innerHTML = `
        <span class="conf-class">${det.class_name.replace("_", " ")}</span>
        <span class="confidence-bar-track"><span class="confidence-bar-fill" style="width:${pct}%;background:${color}"></span></span>
        <span class="conf-value">${pct}%</span>
      `;
      confidenceList.appendChild(row);
    });
  }
}

function avgConfidence(detections, className) {
  const vals = detections.filter((d) => d.class_name === className).map((d) => d.confidence);
  if (!vals.length) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

function renderInsights({ product, gap, tag, detections, latencyMs }) {
  const total = product + gap + tag;
  const gapPer100 = product > 0 ? (gap / product) * 100 : null;
  const tagCoveragePct = product > 0 ? Math.min(100, Math.round((tag / product) * 100)) : null;
  const overallConf = detections.length
    ? detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length
    : 0;

  const sentences = [];
  sentences.push(
    `This shelf photo shows ${product} product facing${product === 1 ? "" : "s"}, ` +
      (gap ? `${gap} empty gap${gap === 1 ? "" : "s"} flagged for restocking, ` : "no empty gaps detected, ") +
      (tag ? `and ${tag} price tag${tag === 1 ? "" : "s"} identified.` : "and no price tags identified.")
  );
  if (gapPer100 !== null && gap > 0) {
    sentences.push(
      `That works out to roughly ${gapPer100.toFixed(1)} gap${gapPer100.toFixed(1) === "1.0" ? "" : "s"} per 100 products, ` +
        `${gapPer100 > 8 ? "a compliance risk worth prioritizing on this shelf." : "within a normal restocking cadence."}`
    );
  }
  if (tagCoveragePct !== null) {
    sentences.push(
      `Tag coverage sits at about ${tagCoveragePct}% relative to detected facings, ` +
        `${tagCoveragePct < 60 ? "suggesting several facings may be missing a visible price tag." : "meaning most facings have a visible price tag."}`
    );
  }
  sentences.push(
    `Overall detection confidence averaged ${(overallConf * 100).toFixed(0)}% across ${total} detections, returned in ${latencyMs.toFixed(0)} ms.`
  );
  insightsNarrative.innerHTML = sentences.map((s) => `<p>${s}</p>`).join("");

  const stat = (label, value, color) =>
    `<div class="insight-stat"><span class="insight-stat-value" style="${color ? `color:${color}` : ""}">${value}</span><span class="insight-stat-label">${label}</span></div>`;
  insightsNumeric.innerHTML = [
    stat("Total detections", total),
    stat("Products", product, "#F59E0B"),
    stat("Gaps", gap, "#3B82F6"),
    stat("Tags", tag, "#14B8A6"),
    stat("Gap rate / 100 products", product ? gapPer100.toFixed(1) : "-"),
    stat("Tag coverage", tagCoveragePct === null ? "-" : `${tagCoveragePct}%`),
    stat("Avg confidence", `${(overallConf * 100).toFixed(0)}%`),
    stat("Inference time", `${latencyMs.toFixed(0)} ms`),
  ].join("");

  render3DBars({ product, gap, tag });
  insightsCard.hidden = false;
}

function render3DBars({ product, gap, tag }) {
  const items = [
    { label: "Product", value: product, color: "#F59E0B" },
    { label: "Gap", value: gap, color: "#3B82F6" },
    { label: "Tag", value: tag, color: "#14B8A6" },
  ];
  const max = Math.max(1, ...items.map((i) => i.value));
  const MAX_HEIGHT_PX = 150;

  insights3d.innerHTML = items
    .map((item) => {
      const h = Math.max(6, Math.round((item.value / max) * MAX_HEIGHT_PX));
      return `
        <div class="bar3d-col">
          <span class="bar3d-value">${item.value}</span>
          <div class="bar3d" style="--h:${h}px;--c:${item.color}">
            <div class="bar3d-top"></div>
            <div class="bar3d-front"></div>
            <div class="bar3d-side"></div>
          </div>
          <span class="bar3d-label">${item.label}</span>
        </div>`;
    })
    .join("");
}

// Per-detector "zoom" views — redraws the original upload on a canvas with
// only one class's boxes at a time (so a teammate can see exactly what one
// detector alone contributed, independent of the merged composite image),
// plus cropped + upscaled close-ups of its highest-confidence detections —
// tags in particular are a handful of pixels in a full shelf photo and are
// unreadable without a real per-box zoom, not just an isolated overlay.
const ZOOM_CLASSES = [
  { key: "product", label: "Product detector", color: "#F59E0B" },
  { key: "empty_gap", label: "Gap detector", color: "#3B82F6" },
  { key: "price_tag", label: "Tag detector", color: "#14B8A6" },
];
const CROP_COUNT = 6;
const CROP_MIN_OUTPUT_PX = 220;

function cropDetail(bitmap, bbox, padRatio = 0.3) {
  const { x1, y1, x2, y2 } = bbox;
  const w = Math.max(1, x2 - x1);
  const h = Math.max(1, y2 - y1);
  const padX = w * padRatio;
  const padY = h * padRatio;
  const sx = Math.max(0, x1 - padX);
  const sy = Math.max(0, y1 - padY);
  const sw = Math.min(bitmap.width - sx, w + padX * 2);
  const sh = Math.min(bitmap.height - sy, h + padY * 2);

  const scale = Math.max(1, CROP_MIN_OUTPUT_PX / Math.max(sw, sh));
  const outW = Math.round(sw * scale);
  const outH = Math.round(sh * scale);

  const canvas = document.createElement("canvas");
  canvas.width = outW;
  canvas.height = outH;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, outW, outH);
  return canvas.toDataURL("image/jpeg", 0.9);
}

async function renderDetectorZooms(detections) {
  if (!lastImageBlob) {
    zoomCard.hidden = true;
    return;
  }

  let bitmap;
  try {
    bitmap = await createImageBitmap(lastImageBlob);
  } catch {
    zoomCard.hidden = true;
    return;
  }

  zoomRow.innerHTML = "";
  ZOOM_CLASSES.forEach((cls) => {
    const dets = detections.filter((d) => d.class_name === cls.key);

    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(bitmap, 0, 0);
    ctx.strokeStyle = cls.color;
    ctx.lineWidth = Math.max(2, Math.round(bitmap.width / 650));
    dets.forEach((d) => {
      const { x1, y1, x2, y2 } = d.bbox;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    });
    const dataUrl = canvas.toDataURL("image/jpeg", 0.86);

    const topDets = [...dets].sort((a, b) => b.confidence - a.confidence).slice(0, CROP_COUNT);
    const crops = topDets.map((d) => ({ url: cropDetail(bitmap, d.bbox), confidence: d.confidence }));

    const panel = document.createElement("button");
    panel.type = "button";
    panel.className = "zoom-panel";
    panel.style.setProperty("--zc", cls.color);
    const countLabel = `${dets.length} detection${dets.length === 1 ? "" : "s"}`;
    panel.innerHTML = `
      <img src="${dataUrl}" alt="${cls.label} view" />
      <span class="zoom-panel-label">${cls.label} <b>${dets.length}</b></span>
    `;
    panel.addEventListener("click", () => openLightbox(dataUrl, `${cls.label} - ${countLabel}`, crops));
    zoomRow.appendChild(panel);
  });

  zoomCard.hidden = false;
}

function openLightbox(src, caption, crops = []) {
  lightboxImage.src = src;
  lightboxCaption.textContent = caption;
  if (crops.length) {
    lightboxCrops.innerHTML = crops
      .map(
        (c) => `
        <div class="lightbox-crop">
          <img src="${c.url}" alt="Close-up detection" />
          <span>${Math.round(c.confidence * 100)}%</span>
        </div>`
      )
      .join("");
    lightboxCrops.hidden = false;
  } else {
    lightboxCrops.innerHTML = "";
    lightboxCrops.hidden = true;
  }
  lightboxBackdrop.hidden = false;
  requestAnimationFrame(() => lightboxBackdrop.classList.add("is-visible"));
}

function closeLightbox() {
  lightboxBackdrop.classList.remove("is-visible");
  setTimeout(() => {
    lightboxBackdrop.hidden = true;
  }, 250);
}

lightboxClose.addEventListener("click", closeLightbox);
lightboxBackdrop.addEventListener("click", (e) => {
  if (e.target === lightboxBackdrop) closeLightbox();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !lightboxBackdrop.hidden) closeLightbox();
});

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) runDetection(file, file.name);
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("is-dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) runDetection(file, file.name);
});

async function loadSamples() {
  try {
    const response = await fetch("/samples");
    const paths = await response.json();
    samplesRow.innerHTML = "";
    paths.slice(0, 8).forEach((path) => {
      const img = document.createElement("img");
      img.src = path;
      img.className = "sample-thumb";
      img.alt = "Sample shelf photo";
      img.addEventListener("click", async () => {
        const blob = await (await fetch(path)).blob();
        runDetection(blob, path.split("/").pop());
      });
      samplesRow.appendChild(img);
    });
  } catch {
    samplesRow.innerHTML = "";
  }
}

function chartTextColors() {
  return { text: "#f2f1ed", muted: "#9c9fa8", grid: "#1c2029" };
}

function renderRadarChart(byTask) {
  const canvas = document.getElementById("radarChart");
  const { muted, text } = chartTextColors();
  const labels = ["Precision", "Recall", "mAP50", "mAP50-95", "F1"];
  const datasets = [];

  TASK_ORDER.forEach((task) => {
    const info = byTask[task];
    if (!info || !info.ready || !info.metrics) return;
    const m = info.metrics;
    const f1 = m.f1 ?? (2 * m.precision * m.recall) / (m.precision + m.recall || 1);
    datasets.push({
      label: TASK_LABEL[task],
      data: [m.precision, m.recall, m.map50, m.map50_95, f1],
      borderColor: TASK_COLOR[task],
      backgroundColor: `${TASK_COLOR[task]}26`,
      pointBackgroundColor: TASK_COLOR[task],
      borderWidth: 2,
    });
  });

  if (radarChartInstance) radarChartInstance.destroy();
  if (!datasets.length) return;

  radarChartInstance = new Chart(canvas, {
    type: "radar",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0,
          max: 1,
          angleLines: { color: "#262b35" },
          grid: { color: "#262b35" },
          pointLabels: { color: muted, font: { size: 11 } },
          ticks: { display: false, backdropColor: "transparent" },
        },
      },
      plugins: {
        legend: { position: "bottom", labels: { color: text, boxWidth: 12, font: { size: 11 } } },
      },
    },
  });
}

async function renderTrainingCurve(canvas, detectorDir) {
  const response = await fetch(`/training-history/${detectorDir}`);
  const history = await response.json();
  if (!history.length) return;

  const { muted, grid: gridColor } = chartTextColors();

  if (curveChartInstances[detectorDir]) curveChartInstances[detectorDir].destroy();
  curveChartInstances[detectorDir] = new Chart(canvas, {
    type: "line",
    data: {
      labels: history.map((h) => h.epoch),
      datasets: [
        {
          label: "mAP50",
          data: history.map((h) => h.map50),
          borderColor: "#F59E0B",
          borderWidth: 1.6,
          pointRadius: 0,
          tension: 0.2,
        },
        {
          label: "mAP50-95",
          data: history.map((h) => h.map50_95),
          borderColor: "#EF4444",
          borderWidth: 1.6,
          pointRadius: 0,
          tension: 0.2,
        },
        {
          label: "Precision",
          data: history.map((h) => h.precision),
          borderColor: "#14B8A6",
          borderWidth: 1,
          pointRadius: 0,
          tension: 0.2,
          hidden: true,
        },
        {
          label: "Recall",
          data: history.map((h) => h.recall),
          borderColor: "#3B82F6",
          borderWidth: 1,
          pointRadius: 0,
          tension: 0.2,
          hidden: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { ticks: { color: muted, maxTicksLimit: 6, font: { size: 9 } }, grid: { color: gridColor } },
        y: { min: 0, max: 1, ticks: { color: muted, font: { size: 9 } }, grid: { color: gridColor } },
      },
      plugins: {
        legend: { labels: { color: muted, boxWidth: 10, font: { size: 9 } } },
      },
    },
  });
}

async function loadModelInfo() {
  try {
    const response = await fetch("/model-info");
    const data = await response.json();
    const byTask = Object.fromEntries(data.detectors.map((d) => [d.task, d]));

    renderRadarChart(byTask);

    statusGrid.innerHTML = "";
    TASK_ORDER.forEach((task) => {
      const info = byTask[task];
      if (!info) return;

      const card = document.createElement("div");
      card.className = "status-card";

      const chipClass = info.ready ? "is-ready" : "is-pending";
      const chipText = info.ready ? "Ready" : "Training…";
      const detectorDir = TASK_DIR[task];

      let metricsHtml = `<p class="status-pending-note">Checkpoint not available yet, trains automatically overnight.</p>`;
      let chartHtml = "";
      if (info.ready && info.metrics) {
        const m = info.metrics;
        metricsHtml = `
          <div class="status-metrics">
            <span>mAP50 <b>${m.map50?.toFixed(3) ?? "-"}</b></span>
            <span>mAP50-95 <b>${m.map50_95?.toFixed(3) ?? "-"}</b></span>
            <span>Precision <b>${m.precision?.toFixed(3) ?? "-"}</b></span>
            <span>Recall <b>${m.recall?.toFixed(3) ?? "-"}</b></span>
          </div>`;
        chartHtml = `<div class="chart-wrap chart-wrap--curve"><canvas id="curve-${detectorDir}"></canvas></div>`;
        if (info.training_curve_url) {
          chartHtml += `<a class="status-pending-note" href="${info.training_curve_url}" target="_blank" rel="noopener">Download static chart (PNG)</a>`;
        }
      }

      card.innerHTML = `
        <div class="status-head">
          <span class="status-name">${TASK_LABEL[task]}</span>
          <span class="status-chip ${chipClass}">${chipText}</span>
        </div>
        ${metricsHtml}
        ${chartHtml}
      `;
      statusGrid.appendChild(card);

      if (info.ready) {
        const canvas = card.querySelector(`#curve-${detectorDir}`);
        if (canvas) renderTrainingCurve(canvas, detectorDir);
      }
    });
  } catch {
    // Backend not reachable yet — leave the grid as-is, next poll retries.
  }
}

loadSamples();
loadModelInfo();
setInterval(loadModelInfo, 30000);
