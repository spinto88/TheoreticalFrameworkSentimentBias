// Chart instances — null until first render, then reused via destroy+recreate
// on the same (static) canvas elements. Canvas elements never leave the DOM,
// so Chart.js can always clean up its ResizeObserver synchronously.
let chartZ1 = null, chartA1 = null;
let chartZ2 = null, chartA2 = null;
let chartB  = null;

function sortByValue(labels, values) {
  const idx = [...Array(labels.length).keys()].sort((a, b) => values[a] - values[b]);
  return { labels: idx.map(i => labels[i]), values: idx.map(i => values[i]) };
}

function exportAllChartsSVG() {
  const dim2Visible = document.getElementById("cardZ2").style.display !== "none";
  if (chartZ1) exportChartSVG(chartZ1, "outlet-bias-z1");
  if (chartA1) exportChartSVG(chartA1, "subject-discrimination-a1");
  if (dim2Visible && chartZ2) exportChartSVG(chartZ2, "outlet-bias-z2");
  if (dim2Visible && chartA2) exportChartSVG(chartA2, "subject-discrimination-a2");
  if (chartB)  exportChartSVG(chartB,  "subject-baseline-b");
}

function renderAllCharts(data) {
  const nDims = data.outlets[0].z.length;
  const outletNames  = data.outlets.map(o => o.outlet);
  const subjectNames = data.subjects.map(s => s.subject);

  // Update dim-1 card titles to include "Dim 1" suffix only when 2D
  const sfx = nDims > 1 ? " — Dim 1" : "";
  document.getElementById("titleZ1").textContent = "Outlet Bias" + sfx;
  document.getElementById("badgeZ1").textContent = nDims > 1 ? "z₁" : "z";
  document.getElementById("titleA1").textContent = "Subject Discrimination" + sfx;
  document.getElementById("badgeA1").textContent = nDims > 1 ? "a₁" : "a";

  // Show or hide dim-2 cards
  const show2 = nDims > 1;
  document.getElementById("cardZ2").style.display = show2 ? "" : "none";
  document.getElementById("cardA2").style.display = show2 ? "" : "none";

  // --- Dim 1 charts ---
  // Passing the existing chart reference causes createBarChart to destroy it
  // properly (canvas still in DOM) before creating a fresh one on the same canvas.
  const { labels: zL1, values: zV1 } = sortByValue(outletNames, data.outlets.map(o => o.z[0]));
  chartZ1 = createBarChart(
    "chartZ1", chartZ1, zL1, zV1,
    nDims > 1 ? "Outlet Bias (z₁)" : "Outlet Bias (z)", "z",
    { x: "Media Outlet", y: nDims > 1 ? "z₁" : "z" },
    { maxRotation: 90, minRotation: 90, autoSkip: false }
  );

  const { labels: aL1, values: aV1 } = sortByValue(subjectNames, data.subjects.map(s => s.a[0]));
  chartA1 = createBarChart(
    "chartA1", chartA1, aL1, aV1,
    nDims > 1 ? "Discrimination (a₁)" : "Discrimination (a)", "a",
    { x: "Subject", y: nDims > 1 ? "a₁" : "a" }
  );

  // --- Dim 2 charts ---
  if (show2) {
    const { labels: zL2, values: zV2 } = sortByValue(outletNames, data.outlets.map(o => o.z[1]));
    chartZ2 = createBarChart(
      "chartZ2", chartZ2, zL2, zV2,
      "Outlet Bias (z₂)", "z",
      { x: "Media Outlet", y: "z₂" },
      { maxRotation: 90, minRotation: 90, autoSkip: false }
    );

    const { labels: aL2, values: aV2 } = sortByValue(subjectNames, data.subjects.map(s => s.a[1]));
    chartA2 = createBarChart(
      "chartA2", chartA2, aL2, aV2,
      "Discrimination (a₂)", "a",
      { x: "Subject", y: "a₂" }
    );
  }

  // --- Baseline chart (always scalar) ---
  const { labels: bL, values: bV } = sortByValue(subjectNames, data.subjects.map(s => s.b));
  chartB = createBarChart(
    "chartB", chartB, bL, bV,
    "Baseline (b)", "b",
    { x: "Subject", y: "b" }
  );
}
