let chartZ, chartA, chartB;

function renderAllCharts(data) {
  renderChartZ(data.outlets);
  renderChartA(data.subjects);
  renderChartB(data.subjects);
}

function renderChartZ(outlets) {
  const labels = outlets.map(o => o.outlet);
  const values = outlets.map(o => o.z);
  chartZ = createBarChart("chartZ", chartZ, labels, values, "Outlet Bias (z)", "z",
    { x: "Media Outlet", y: "Bias Score (z)" });
}

function renderChartA(subjects) {
  const labels = subjects.map(s => s.subject);
  const values = subjects.map(s => s.a);
  chartA = createBarChart("chartA", chartA, labels, values, "Discrimination (a)", "a",
    { x: "Subject", y: "Discrimination (a)" });
}

function renderChartB(subjects) {
  const labels = subjects.map(s => s.subject);
  const values = subjects.map(s => s.b);
  chartB = createBarChart("chartB", chartB, labels, values, "Baseline (b)", "b",
    { x: "Subject", y: "Baseline Sentiment (b)" });
}
