let chartZ, chartA, chartB;

function exportAllChartsSVG() {
  const charts = [
    [() => chartZ, "outlet-bias-z"],
    [() => chartA, "subject-discrimination-a"],
    [() => chartB, "subject-baseline-b"],
  ];
  charts.forEach(([getRef, name], i) =>
    setTimeout(() => exportChartSVG(getRef(), name), i * 250)
  );
}

// Sort labels and values together ascending by value (most negative → most positive).
function sortByValue(labels, values) {
  const idx = [...Array(labels.length).keys()].sort((a, b) => values[a] - values[b]);
  return {
    labels: idx.map(i => labels[i]),
    values: idx.map(i => values[i]),
  };
}

function renderAllCharts(data) {
  renderChartZ(data.outlets);
  renderChartA(data.subjects);
  renderChartB(data.subjects);
}

function renderChartZ(outlets) {
  let labels = outlets.map(o => o.outlet);
  let values = outlets.map(o => o.z);
  ({ labels, values } = sortByValue(labels, values));
  chartZ = createBarChart("chartZ", chartZ, labels, values, "Outlet Bias (z)", "z",
    { x: "Media Outlet", y: "Bias Score (z)" });
}

function renderChartA(subjects) {
  let labels = subjects.map(s => s.subject);
  let values = subjects.map(s => s.a);
  ({ labels, values } = sortByValue(labels, values));
  chartA = createBarChart("chartA", chartA, labels, values, "Discrimination (a)", "a",
    { x: "Subject", y: "Discrimination (a)" });
}

function renderChartB(subjects) {
  let labels = subjects.map(s => s.subject);
  let values = subjects.map(s => s.b);
  ({ labels, values } = sortByValue(labels, values));
  chartB = createBarChart("chartB", chartB, labels, values, "Baseline (b)", "b",
    { x: "Subject", y: "Baseline Sentiment (b)" });
}
