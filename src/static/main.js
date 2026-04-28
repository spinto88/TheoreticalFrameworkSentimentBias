const fileInput = document.getElementById("fileInput");
const dropZone  = document.getElementById("dropZone");
const runBtn    = document.getElementById("runBtn");

// ── File selection ──────────────────────────────────────────────
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  document.getElementById("fileName").textContent = file.name;
  document.getElementById("dropText").innerHTML =
    `<strong>${file.name}</strong> ready to analyse`;
  dropZone.classList.add("has-file");
  runBtn.disabled = false;
  clearError();
}

// ── Drag & drop ─────────────────────────────────────────────────
dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

["dragleave", "dragend"].forEach(evt =>
  dropZone.addEventListener(evt, () => dropZone.classList.remove("drag-over"))
);

dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    setFile(file);
  }
});

// ── UI helpers ──────────────────────────────────────────────────
function showError(msg) {
  const banner = document.getElementById("errorBanner");
  banner.textContent = msg;
  banner.style.display = "block";
}

function clearError() {
  document.getElementById("errorBanner").style.display = "none";
}

function toggleRaw() {
  const el = document.getElementById("rawOutput");
  el.style.display = el.style.display === "block" ? "none" : "block";
}

// ── Processing time estimate ─────────────────────────────────────
// Rough heuristic for L-BFGS-B with analytical gradient.
function estimateSeconds(data, nDims) {
  const m = new Set(data.map(r => r.outlet)).size;
  const k = new Set(data.map(r => r.subject)).size;
  return Math.max(1, Math.round(0.001 * m * k * (m + 2 * k) * nDims));
}

let _timerInterval = null;
let _startTime     = null;

function startProgress(estimateSec) {
  _startTime = Date.now();
  const section = document.getElementById("progressSection");
  section.style.display = "block";
  document.getElementById("progressEstimate").textContent =
    `~${estimateSec}s estimated`;

  _timerInterval = setInterval(() => {
    const elapsed = (Date.now() - _startTime) / 1000;
    document.getElementById("progressElapsed").textContent =
      `${elapsed.toFixed(1)}s elapsed`;
    const pct = Math.min(94, (elapsed / estimateSec) * 100);
    document.getElementById("progressFill").style.width = pct + "%";
  }, 100);
}

function stopProgress() {
  clearInterval(_timerInterval);
  _timerInterval = null;
  const elapsed = ((Date.now() - _startTime) / 1000).toFixed(1);
  document.getElementById("progressElapsed").textContent =
    `Completed in ${elapsed}s`;
  document.getElementById("progressEstimate").textContent = "";
  document.getElementById("progressFill").style.width = "100%";
  setTimeout(() => {
    document.getElementById("progressSection").style.display = "none";
  }, 2500);
}

// ── Analysis ────────────────────────────────────────────────────
async function sendData() {
  clearError();

  const file  = fileInput.files[0];
  const isCSV = file.name.toLowerCase().endsWith(".csv");

  let jsonData;
  try {
    const text = await file.text();
    jsonData = isCSV ? parseCSV(text) : JSON.parse(text);
  } catch (e) {
    showError(`Could not parse the file — ${e.message}`);
    return;
  }

  const nDims = parseInt(document.getElementById("nDimensionsSelect").value, 10);
  jsonData.n_dimensions = nDims;

  runBtn.disabled    = true;
  runBtn.textContent = "Running…";
  startProgress(estimateSeconds(jsonData.data, nDims));

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(jsonData)
    });

    if (!response.ok) {
      const err = await response.text();
      showError(`Server error ${response.status}: ${err}`);
      return;
    }

    const result = await response.json();

    document.getElementById("rawOutput").textContent = JSON.stringify(result, null, 2);
    document.getElementById("results").style.display = "block";

    renderAllCharts(result);
  } catch (e) {
    showError(`Request failed: ${e.message}`);
  } finally {
    stopProgress();
    runBtn.disabled    = false;
    runBtn.textContent = "Run Analysis";
  }
}
