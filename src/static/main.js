const fileInput = document.getElementById("fileInput");
const dropZone  = document.getElementById("dropZone");
const runBtn    = document.getElementById("runBtn");

let _currentDim = 1;
let _fileType   = "input"; // "input" | "results"

function setDim(d) {
  _currentDim = d;
  document.querySelectorAll(".dim-pill").forEach(btn => {
    btn.classList.toggle("active", parseInt(btn.dataset.dim) === d);
  });
}

// ── File selection ──────────────────────────────────────────────
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

async function setFile(file) {
  clearError();

  // Detect whether this is a pre-computed results JSON
  _fileType = "input";
  if (!file.name.toLowerCase().endsWith(".csv")) {
    try {
      const text = await file.text();
      const obj  = JSON.parse(text);
      if (obj.outlets && obj.subjects && Array.isArray(obj.outlets)) {
        _fileType = "results";
      }
    } catch {
      // treat as input data
    }
  }

  if (_fileType === "results") {
    document.getElementById("dropText").innerHTML =
      `<strong>${file.name}</strong> — results file detected`;
    runBtn.textContent = "Load Results";
  } else {
    document.getElementById("dropText").innerHTML =
      `<strong>${file.name}</strong> ready to analyse`;
    runBtn.textContent = "Run Analysis";
  }

  document.getElementById("fileName").textContent = file.name;
  dropZone.classList.add("has-file");
  runBtn.disabled = false;
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
// Rough heuristic based on differential_evolution cost: O(m · k · (m + 2k))
// with popsize=25 and maxiter=1000.
function estimateSeconds(data) {
  const m = new Set(data.map(r => r.outlet)).size;
  const k = new Set(data.map(r => r.subject)).size;
  return Math.max(2, Math.round(0.005 * m * k * (m + 2 * k)));
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

  let parsed;
  try {
    const text = await file.text();
    parsed = isCSV ? parseCSV(text) : JSON.parse(text);
  } catch (e) {
    showError(`Could not parse the file — ${e.message}`);
    return;
  }

  // ── Direct results load ────────────────────────────────────────
  if (_fileType === "results") {
    const nDims = parsed.outlets?.[0]?.z?.length ?? 1;
    setDim(nDims);
    document.getElementById("rawOutput").textContent = JSON.stringify(parsed, null, 2);
    document.getElementById("results").style.display = "block";
    renderAllCharts(parsed, nDims);
    return;
  }

  // ── Run analysis via API ───────────────────────────────────────
  // Always use the UI-selected dimension; strip any n_dimensions from the file.
  const jsonData = { data: parsed.data, n_dimensions: _currentDim };

  runBtn.disabled    = true;
  runBtn.textContent = "Running…";
  startProgress(estimateSeconds(jsonData.data));

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

    renderAllCharts(result, _currentDim);
  } catch (e) {
    showError(`Request failed: ${e.message}`);
  } finally {
    stopProgress();
    runBtn.disabled    = false;
    runBtn.textContent = "Run Analysis";
  }
}
