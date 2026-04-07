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
    // transfer to the real input so sendData() can read it
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

// ── Analysis ────────────────────────────────────────────────────
async function sendData() {
  clearError();

  let jsonData;
  try {
    const text = await fileInput.files[0].text();
    jsonData = JSON.parse(text);
  } catch {
    showError("Could not parse the file — make sure it is valid JSON.");
    return;
  }

  runBtn.disabled = true;
  runBtn.textContent = "Running…";

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
    runBtn.disabled = false;
    runBtn.textContent = "Run Analysis";
  }
}
