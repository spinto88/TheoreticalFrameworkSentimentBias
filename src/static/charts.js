// Per-dimension chart instances (arrays indexed by dimension 0, 1, …)
let chartZ = [], chartA = [], chartB = [];

function exportAllChartsSVG() {
  const nDims = chartZ.length;
  chartZ.forEach((c, d) => {
    const sfx = nDims > 1 ? `-dim${d + 1}` : "";
    exportChartSVG(c,        `outlet-bias-z${sfx}`);
    exportChartSVG(chartA[d], `subject-discrimination-a${sfx}`);
    exportChartSVG(chartB[d], `subject-baseline-b${sfx}`);
  });
}

function exportChartSVGById(canvasId) {
  const all = [...chartZ, ...chartA, ...chartB];
  const ref = all.find(c => c && c.canvas && c.canvas.id === canvasId);
  if (ref) exportChartSVG(ref, canvasId);
}

function sortByValue(labels, values) {
  const idx = [...Array(labels.length).keys()].sort((a, b) => values[a] - values[b]);
  return { labels: idx.map(i => labels[i]), values: idx.map(i => values[i]) };
}

const CHART_META = {
  z: {
    title: "Outlet Bias",
    badge: "z",
    badgeClass: "badge-z",
    desc: "Latent bias score for each outlet along this dimension. Only the product z·a carries directional meaning.",
    height: 380,
    xTickOptions: { maxRotation: 90, minRotation: 90, autoSkip: false },
  },
  a: {
    title: "Subject Discrimination",
    badge: "a",
    badgeClass: "badge-a",
    desc: "Discrimination parameter for each subject along this dimension. Larger absolute values amplify the outlet bias effect.",
    height: 300,
    xTickOptions: {},
  },
  b: {
    title: "Subject Baseline",
    badge: "b",
    badgeClass: "badge-b",
    desc: "Baseline sentiment for each subject along this dimension, independent of outlet bias.",
    height: 300,
    xTickOptions: {},
  },
};

function makeChartCardHTML(canvasId, meta, dimSuffix) {
  const title = meta.title + dimSuffix;
  const badge = meta.badge + (dimSuffix ? dimSuffix.replace(" — Dim ", "") : "");
  return `
    <div class="chart-card">
      <div class="chart-header">
        <div class="chart-header-left">
          <span class="chart-title">${title}</span>
          <span class="chart-badge ${meta.badgeClass}">${badge}</span>
        </div>
        <div class="export-group">
          <button class="btn-export" onclick="exportChartPNG('${canvasId}','${canvasId}')">PNG</button>
          <button class="btn-export" onclick="exportChartSVGById('${canvasId}')">SVG</button>
        </div>
      </div>
      <p class="chart-desc">${meta.desc}</p>
      <div class="chart-wrap" style="height:${meta.height}px"><canvas id="${canvasId}"></canvas></div>
    </div>`;
}

function renderAllCharts(data) {
  const nDims = data.outlets[0].z.length;

  // Destroy old instances
  [...chartZ, ...chartA, ...chartB].forEach(c => c && c.destroy());
  chartZ = []; chartA = []; chartB = [];

  const container = document.getElementById("chartsContainer");
  container.innerHTML = "";

  for (let d = 0; d < nDims; d++) {
    const dimSuffix = nDims > 1 ? ` — Dim ${d + 1}` : "";
    const idSuffix  = nDims > 1 ? `_dim${d + 1}` : "";

    const zId = `chartZ${idSuffix}`;
    const aId = `chartA${idSuffix}`;
    const bId = `chartB${idSuffix}`;

    container.innerHTML +=
      makeChartCardHTML(zId, CHART_META.z, dimSuffix) +
      makeChartCardHTML(aId, CHART_META.a, dimSuffix) +
      makeChartCardHTML(bId, CHART_META.b, dimSuffix);
  }

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
      CHART_META.z.xTickOptions
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

    const { labels: bL, values: bV } = sortByValue(
      data.subjects.map(s => s.subject),
      data.subjects.map(s => s.b[d])
    );
    chartB[d] = createBarChart(
      `chartB${idSuffix}`, null, bL, bV,
      `Baseline (b${dLabel})`, "b",
      { x: "Subject", y: `Baseline Sentiment (b${dLabel})` }
    );
  }
}
