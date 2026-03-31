function createBarChart(canvasId, chartRef, labels, values, label) {

  if (chartRef) {
    chartRef.destroy();
  }

  return new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: label,
        data: values
      }]
    }
  });
}
