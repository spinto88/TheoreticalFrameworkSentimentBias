let outletChart, subjectChart, biasChart;

function renderAllCharts(data) {
  renderOutletChart(data.outlets);
  renderSubjectChart(data.subjects);
  renderBiasChart(data.subjects);
}

// ---- OUTLETS ----
function renderOutletChart(outlets) {
  const labels = outlets.map(o => o.outlet);
  const values = outlets.map(o => o.z);

  outletChart = createBarChart("outletChart", outletChart, labels, values, "Outlet Score (z)");
}

// ---- SUBJECTS ----
function renderSubjectChart(subjects) {
  const labels = subjects.map(s => s.subject);
  const values = subjects.map(s => s.a);

  subjectChart = createBarChart("subjectChart", subjectChart, labels, values, "Subject Score (a)");
}

// ---- BIAS ----
function renderBiasChart(subjects) {
  const labels = subjects.map(s => s.subject);
  const values = subjects.map(s => s.a - s.b);

  biasChart = createBarChart("biasChart", biasChart, labels, values, "Bias (a - b)");
}
