// Chart instances indexed by dimension (z and a per dim; b always single)
let chartZ = [], chartA = [], chartB = null;

function exportAllChartsSVG() {
  const nDims = chartZ.length;
  chartZ.forEach((c, d) => {
    const sfx = nDims > 1 ? `-dim${d + 1}` : "";
    exportChartSVG(c,        `outlet-bias-z${sfx}`);
    exportChartSVG(chartA[d], `subject-discrimination-a${sfx}`);
  });
  if (chartB) exportChartSVG(chartB, "subject-baseline-b");
}

function exportChartSVGById(canvasId) {
  const all = [...chartZ, ...chartA, chartB].filter(Boolean);
  const ref = all.find(c => c.canvas && c.canvas.id === canvasId);
  if (ref) exportChartSVG(ref, canvasId);
}

function sortByValue(labels, values) {
  const idx = [...Array(labels.length).keys()].sort((a, b) => values[a] - values[b]);
  return { labels: idx.map(i => labels[i]), values: idx.map(i => values[i]) };
}

function makeChartCardHTML(canvasId, title, badge, badgeClass, desc, height) {
  return `
    <div class="chart-card">
      <div class="chart-header">
        <div class="chart-header-left">
          <span class="chart-title">${title}</span>
          <span class="chart-badge ${badgeClass}">${badge}</span>
        </div>
        <div class="export-group">
          <button class="btn-export" onclick="exportChartPNG('${canvasId}','${canvasId}')">PNG</button>
          <button class="btn-export" onclick="exportChartSVGById('${canvasId}')">SVG</button>
        </div>
      </div>
      <div class="chart-wrap" style="height:${height || 300}px"><canvas id="${canvasId}"></canvas></div>
    </div>`;
}

function renderAllCharts(data) {
  const nDims = data.outlets[0].z.length;

  // Destroy old instances before clearing the DOM
  [...chartZ, ...chartA].forEach(c => c && c.destroy());
  if (chartB) chartB.destroy();
  chartZ = []; chartA = []; chartB = null;

  // Build the full HTML string first, then set innerHTML once
  let html = "";
  for (let d = 0; d < nDims; d++) {
    const dimSuffix = nDims > 1 ? ` — Dim ${d + 1}` : "";
    const idSuffix  = nDims > 1 ? `_dim${d + 1}` : "";
    const dLabel    = nDims > 1 ? String(d + 1) : "";
    html += makeChartCardHTML(
      `chartZ${idSuffix}`,
      `Outlet Bias${dimSuffix}`, `z${dLabel}`, "badge-z",
      "Latent bias score per outlet along this dimension.",
      380
    );
    html += makeChartCardHTML(
      `chartA${idSuffix}`,
      `Subject Discrimination${dimSuffix}`, `a${dLabel}`, "badge-a",
      "Discrimination parameter per subject along this dimension."
    );
  }
  html += makeChartCardHTML(
    "chartB",
    "Subject Baseline", "b", "badge-b",
    "Scalar baseline sentiment per subject, shared across all dimensions."
  );

  document.getElementById("chartsContainer").innerHTML = html;

  // Now create the Chart.js instances (canvas elements are in the DOM)
  for (let d = 0; d < nDims; d++) {
    const idSuffix = nDims > 1 ? `_dim${d + 1}` : "";
    const dLabel   = nDims > 1 ? String(d + 1) : "";

    const { labels: zL, values: zV } = sortByValue(
      data.outlets.map(o => o.outlet),
      data.outlets.map(o => o.z[d])
    );
    chartZ[d] = createBarChart(
      `chartZ${idSuffix}`, null, zL, zV,
      `Outlet Bias (z${dLabel})`, "z",
      { x: "Media Outlet", y: `Bias Score (z${dLabel})` },
      { maxRotation: 90, minRotation: 90, autoSkip: false }
    );

    const { labels: aL, values: aV } = sortByValue(
      data.subjects.map(s => s.subject),
      data.subjects.map(s => s.a[d])
    );
    chartA[d] = createBarChart(
      `chartA${idSuffix}`, null, aL, aV,
      `Discrimination (a${dLabel})`, "a",
      { x: "Subject", y: `Discrimination (a${dLabel})` }
    );
  }

  // b is always a scalar — one chart regardless of n_dims
  const { labels: bL, values: bV } = sortByValue(
    data.subjects.map(s => s.subject),
    data.subjects.map(s => s.b)
  );
  chartB = createBarChart(
    "chartB", null, bL, bV,
    "Baseline (b)", "b",
    { x: "Subject", y: "Baseline Sentiment (b)" }
  );
}
