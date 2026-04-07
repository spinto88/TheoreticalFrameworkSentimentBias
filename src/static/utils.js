const COLORS = {
  z: { pos: "rgba(99,102,241,0.85)",  neg: "rgba(239,68,68,0.75)",  border_pos: "#6366f1", border_neg: "#ef4444" },
  a: { pos: "rgba(16,185,129,0.85)",  neg: "rgba(239,68,68,0.75)",  border_pos: "#10b981", border_neg: "#ef4444" },
  b: { pos: "rgba(245,158,11,0.85)",  neg: "rgba(239,68,68,0.75)",  border_pos: "#f59e0b", border_neg: "#ef4444" },
};

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 500, easing: "easeOutQuart" },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: "#1a1d27",
      borderColor: "#2a2d3a",
      borderWidth: 1,
      titleColor: "#e2e4ed",
      bodyColor: "#9ca3af",
      padding: 12,
      callbacks: {
        label: ctx => `  ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`
      }
    }
  },
  scales: {
    x: {
      grid: { color: "rgba(255,255,255,0.04)" },
      ticks: { color: "#6b7280", font: { size: 12 } },
      border: { color: "#2a2d3a" }
    },
    y: {
      grid: { color: "rgba(255,255,255,0.06)" },
      ticks: { color: "#6b7280", font: { size: 11 }, maxTicksLimit: 7 },
      border: { color: "#2a2d3a" }
    }
  }
};

function barColors(values, palette) {
  return values.map(v =>
    v >= 0 ? palette.pos : palette.neg
  );
}

function borderColors(values, palette) {
  return values.map(v =>
    v >= 0 ? palette.border_pos : palette.border_neg
  );
}

function createBarChart(canvasId, chartRef, labels, values, label, paletteKey) {
  if (chartRef) chartRef.destroy();

  const palette = COLORS[paletteKey] || COLORS.z;

  return new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label,
        data: values,
        backgroundColor: barColors(values, palette),
        borderColor: borderColors(values, palette),
        borderWidth: 1.5,
        borderRadius: 6,
        borderSkipped: false,
      }]
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        ...CHART_DEFAULTS.scales,
        y: {
          ...CHART_DEFAULTS.scales.y,
          afterDataLimits(axis) {
            const pad = (axis.max - axis.min) * 0.12 || 0.5;
            axis.max += pad;
            axis.min -= pad;
          }
        }
      }
    }
  });
}
