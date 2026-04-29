// Chart instances — null until first render, then reused via destroy+recreate
let chartZ1 = null, chartA1 = null;
let chart2DScatter = null;
let chartB = null;

function sortByValue(labels, values) {
  const idx = [...Array(labels.length).keys()].sort((a, b) => values[a] - values[b]);
  return { labels: idx.map(i => labels[i]), values: idx.map(i => values[i]) };
}

function exportAllChartsSVG() {
  const scatter2DVisible = document.getElementById("card2DScatter").style.display !== "none";
  if (chartZ1) exportChartSVG(chartZ1, "outlet-bias-z1");
  if (chartA1) exportChartSVG(chartA1, "subject-discrimination-a1");
  if (scatter2DVisible) exportChartPNG("chart2DScatter", "latent-space-2d");
  if (chartB)  exportChartSVG(chartB,  "subject-baseline-b");
}

function renderAllCharts(data) {
  const nDims = data.outlets[0].z.length;
  const outletNames  = data.outlets.map(o => o.outlet);
  const subjectNames = data.subjects.map(s => s.subject);

  const show1D = nDims === 1;
  const show2D = nDims > 1;

  document.getElementById("cardZ1").style.display       = show1D ? "" : "none";
  document.getElementById("cardA1").style.display       = show1D ? "" : "none";
  document.getElementById("card2DScatter").style.display = show2D ? "" : "none";
  document.getElementById("cardZ2").style.display       = "none";
  document.getElementById("cardA2").style.display       = "none";

  if (show1D) {
    document.getElementById("titleZ1").textContent = "Outlet Bias";
    document.getElementById("badgeZ1").textContent = "z";
    document.getElementById("titleA1").textContent = "Subject Discrimination";
    document.getElementById("badgeA1").textContent = "a";

    if (chart2DScatter) { chart2DScatter.destroy(); chart2DScatter = null; }

    const { labels: zL1, values: zV1 } = sortByValue(outletNames, data.outlets.map(o => o.z[0]));
    chartZ1 = createBarChart(
      "chartZ1", chartZ1, zL1, zV1,
      "Outlet Bias (z)", "z",
      { x: "Media Outlet", y: "z" },
      { maxRotation: 90, minRotation: 90, autoSkip: false }
    );

    const { labels: aL1, values: aV1 } = sortByValue(subjectNames, data.subjects.map(s => s.a[0]));
    chartA1 = createBarChart(
      "chartA1", chartA1, aL1, aV1,
      "Discrimination (a)", "a",
      { x: "Subject", y: "a" }
    );
  }

  if (show2D) {
    if (chartZ1) { chartZ1.destroy(); chartZ1 = null; }
    if (chartA1) { chartA1.destroy(); chartA1 = null; }

    chart2DScatter = createScatterChart(
      "chart2DScatter", chart2DScatter,
      outletNames, data.outlets.map(o => o.z),
      subjectNames, data.subjects.map(s => s.a)
    );
  }

  // Baseline — always scalar, always a bar chart
  const { labels: bL, values: bV } = sortByValue(subjectNames, data.subjects.map(s => s.b));
  chartB = createBarChart(
    "chartB", chartB, bL, bV,
    "Baseline (b)", "b",
    { x: "Subject", y: "b" }
  );
}

// Maps a point (x,y) in [-2,2]² to a color that fades from neutral gray
// at the origin toward a distinct vertex color as it approaches each quadrant corner.
// Q1(+,+)=red  Q2(-,+)=indigo  Q3(-,-)=teal  Q4(+,-)=amber
function quadrantColor(x, y, alpha) {
  const LIMIT = 2;
  const nx = x / LIMIT;
  const ny = y / LIMIT;

  const vertices = [
    { vx:  1, vy:  1, r: 239, g:  68, b:  68 },
    { vx: -1, vy:  1, r:  99, g: 102, b: 241 },
    { vx: -1, vy: -1, r:  16, g: 185, b: 129 },
    { vx:  1, vy: -1, r: 245, g: 158, b:  11 },
  ];

  const nr = 160, ng = 160, nb = 160;

  const t = Math.min(Math.sqrt(nx * nx + ny * ny) / Math.SQRT2, 1);

  const weights = vertices.map(v => Math.max(0, nx * v.vx) * Math.max(0, ny * v.vy));
  const totalW  = weights.reduce((s, w) => s + w, 0);

  let cr = nr, cg = ng, cb = nb;
  if (totalW > 1e-9) {
    cr = weights.reduce((s, w, i) => s + w * vertices[i].r, 0) / totalW;
    cg = weights.reduce((s, w, i) => s + w * vertices[i].g, 0) / totalW;
    cb = weights.reduce((s, w, i) => s + w * vertices[i].b, 0) / totalW;
  }

  return `rgba(${Math.round(nr + t*(cr-nr))},${Math.round(ng + t*(cg-ng))},${Math.round(nb + t*(cb-nb))},${alpha})`;
}

function createScatterChart(canvasId, chartRef, outletNames, zVecs, subjectNames, aVecs) {
  if (chartRef) chartRef.destroy();

  const outletDataset = {
    label: "Outlets (z)",
    data: outletNames.map((name, i) => ({ x: zVecs[i][0], y: zVecs[i][1], name })),
    backgroundColor: zVecs.map(z => quadrantColor(z[0], z[1], 0.80)),
    borderColor:     zVecs.map(z => quadrantColor(z[0], z[1], 1.00)),
    borderWidth: 1.5,
    pointStyle: "circle",
    pointRadius: 9,
    pointHoverRadius: 12,
  };

  const subjectDataset = {
    label: "Subjects (a)",
    data: subjectNames.map((name, i) => ({ x: aVecs[i][0], y: aVecs[i][1], name })),
    backgroundColor: aVecs.map(a => quadrantColor(a[0], a[1], 0.80)),
    borderColor:     aVecs.map(a => quadrantColor(a[0], a[1], 1.00)),
    borderWidth: 1.5,
    pointStyle: "rect",
    pointRadius: 9,
    pointHoverRadius: 12,
  };

  const axisTitleStyle = {
    display: true,
    color: "#374151",
    font: { size: 13, weight: "600", family: "'Inter', system-ui, sans-serif" },
  };

  return new Chart(document.getElementById(canvasId), {
    type: "scatter",
    data: { datasets: [outletDataset, subjectDataset] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 500, easing: "easeOutQuart" },
      plugins: {
        legend: {
          display: true,
          labels: {
            color: "#374151",
            font: { size: 13 },
            usePointStyle: true,
            pointStyleWidth: 14,
          }
        },
        tooltip: {
          backgroundColor: "#1e293b",
          borderColor: "#334155",
          borderWidth: 1,
          titleColor: "#f1f5f9",
          bodyColor: "#94a3b8",
          padding: 13,
          titleFont: { size: 13, weight: "600" },
          bodyFont: { size: 13 },
          callbacks: {
            title: items => items[0].raw.name,
            label: ctx => {
              const p = ctx.datasetIndex === 0 ? "z" : "a";
              return `  ${p}₁ = ${ctx.parsed.x.toFixed(4)},  ${p}₂ = ${ctx.parsed.y.toFixed(4)}`;
            }
          }
        }
      },
      scales: {
        x: {
          title: { ...axisTitleStyle, text: "Dimension 1" },
          grid:  { color: "rgba(0,0,0,0.06)" },
          border: { color: "#d1d5db" },
          ticks: { color: "#4b5563", font: { size: 13 } },
          min: -2.0,
          max:  2.0,
        },
        y: {
          title: { ...axisTitleStyle, text: "Dimension 2" },
          grid:  { color: "rgba(0,0,0,0.06)" },
          border: { color: "#d1d5db" },
          ticks: { color: "#4b5563", font: { size: 13 } },
          min: -2.0,
          max:  2.0,
        }
      }
    }
  });
}
