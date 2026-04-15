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

function renderAllCharts(data, nDims = 1) {
  renderChartZ(data.outlets, nDims);
  renderChartA(data.subjects, nDims);
  renderChartB(data.subjects);
}

function resetChartZoom(chart) {
  if (chart) chart.resetZoom();
}

function renderChartZ(outlets, nDims) {
  const wrapEl      = document.getElementById("chartZ").parentElement;
  const resetBtn    = document.getElementById("resetZoomZ");
  if (nDims === 2) {
    wrapEl.style.height = "420px";
    resetBtn.style.display = "inline-flex";
    document.getElementById("descZ").innerHTML =
      "Each outlet is positioned in the 2D latent bias space. " +
      "The axes represent the two bias dimensions <em>z</em><sub>1</sub> and <em>z</em><sub>2</sub>. " +
      "The dot product <em>z\u00b7a</em> with a subject\u2019s discrimination vector determines the direction and strength of sentiment reinforcement. " +
      "<span style=\"color:var(--muted);font-size:0.75rem\">Scroll to zoom \u00b7 drag to pan \u00b7 crosshair to reset.</span>";
    const points = outlets.map(o => ({ x: o.z[0], y: o.z[1], label: o.outlet }));
    chartZ = createScatterChart("chartZ", chartZ, points, "z",
      { x: "z\u2081 (first dimension)", y: "z\u2082 (second dimension)" });
  } else {
    wrapEl.style.height = "380px";
    resetBtn.style.display = "none";
    document.getElementById("descZ").innerHTML =
      "Latent score for each media outlet. The sign of <em>z</em> has no standalone meaning \u2014 what matters is its product with the subject\u2019s discrimination parameter <em>a</em>: if <em>z\u00b7a &gt; 0</em> the outlet reinforces positive sentiment toward that subject; if <em>z\u00b7a &lt; 0</em> it reinforces negative sentiment.";
    let labels = outlets.map(o => o.outlet);
    let values = outlets.map(o => o.z[0]);
    ({ labels, values } = sortByValue(labels, values));
    chartZ = createBarChart("chartZ", chartZ, labels, values, "Outlet Bias (z)", "z",
      { x: "Media Outlet", y: "Bias Score (z)" },
      { maxRotation: 90, minRotation: 90, autoSkip: false });
  }
}

function renderChartA(subjects, nDims) {
  const wrapEl   = document.getElementById("chartA").parentElement;
  const resetBtn = document.getElementById("resetZoomA");
  if (nDims === 2) {
    wrapEl.style.height = "420px";
    resetBtn.style.display = "inline-flex";
    document.getElementById("descA").innerHTML =
      "Each subject is positioned in the 2D discrimination space. " +
      "The axes represent the two discrimination dimensions <em>a</em><sub>1</sub> and <em>a</em><sub>2</sub>. " +
      "The dot product <em>z\u00b7a</em> with each outlet\u2019s bias vector determines the log-odds contribution of outlet bias for this subject. " +
      "<span style=\"color:var(--muted);font-size:0.75rem\">Scroll to zoom \u00b7 drag to pan \u00b7 crosshair to reset.</span>";
    const points = subjects.map(s => ({ x: s.a[0], y: s.a[1], label: s.subject }));
    chartA = createScatterChart("chartA", chartA, points, "a",
      { x: "a\u2081 (first dimension)", y: "a\u2082 (second dimension)" });
  } else {
    wrapEl.style.height = "300px";
    resetBtn.style.display = "none";
    document.getElementById("descA").innerHTML =
      "Discrimination parameter for each subject. Like <em>z</em>, the sign of <em>a</em> is only meaningful in combination with an outlet\u2019s score: the product <em>z\u00b7a</em> determines the direction of reinforcement \u2014 positive means the outlet tends to cover the subject favourably, negative means unfavourably. Larger absolute values of <em>a</em> amplify this effect.";
    let labels = subjects.map(s => s.subject);
    let values = subjects.map(s => s.a[0]);
    ({ labels, values } = sortByValue(labels, values));
    chartA = createBarChart("chartA", chartA, labels, values, "Discrimination (a)", "a",
      { x: "Subject", y: "Discrimination (a)" });
  }
}

function renderChartB(subjects) {
  let labels = subjects.map(s => s.subject);
  let values = subjects.map(s => s.b);
  ({ labels, values } = sortByValue(labels, values));
  chartB = createBarChart("chartB", chartB, labels, values, "Baseline (b)", "b",
    { x: "Subject", y: "Baseline Sentiment (b)" });
}

// ── Scatter chart factory ────────────────────────────────────────
function createScatterChart(canvasId, chartRef, points, paletteKey, axisLabels) {
  if (chartRef) chartRef.destroy();

  const palette = COLORS[paletteKey] || COLORS.z;

  const axisTitleStyle = {
    display: true,
    color: "#374151",
    font: { size: 13, weight: "600", family: "'Inter', system-ui, sans-serif" },
  };

  // Inline plugin: draws dashed zero lines and point labels
  const overlayPlugin = {
    id: `overlay_${canvasId}`,
    afterDatasetsDraw(chart) {
      const ctx = chart.ctx;
      const { left, right, top, bottom } = chart.chartArea;
      const xScale = chart.scales.x;
      const yScale = chart.scales.y;

      // Dashed zero lines
      ctx.save();
      ctx.strokeStyle = "rgba(156,163,175,0.75)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      if (xScale.min <= 0 && xScale.max >= 0) {
        const x0 = xScale.getPixelForValue(0);
        ctx.beginPath(); ctx.moveTo(x0, top); ctx.lineTo(x0, bottom); ctx.stroke();
      }
      if (yScale.min <= 0 && yScale.max >= 0) {
        const y0 = yScale.getPixelForValue(0);
        ctx.beginPath(); ctx.moveTo(left, y0); ctx.lineTo(right, y0); ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.restore();

      // Point labels
      const meta = chart.getDatasetMeta(0);
      chart.data.datasets[0].data.forEach((pt, i) => {
        const el = meta.data[i];
        if (!el) return;
        const pos = el.getProps(["x", "y"], true);
        ctx.save();
        ctx.fillStyle = "#1f2937";
        ctx.font = "600 12px Inter, system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(pt.label, pos.x + 10, pos.y - 5);
        ctx.restore();
      });
    }
  };

  return new Chart(document.getElementById(canvasId), {
    type: "scatter",
    data: {
      datasets: [{
        label: axisLabels?.title || "",
        data: points,
        backgroundColor: palette.pos,
        borderColor: palette.border_pos,
        borderWidth: 1.5,
        pointRadius: 8,
        pointHoverRadius: 10,
      }]
    },
    plugins: [overlayPlugin],
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 500, easing: "easeOutQuart" },
      layout: { padding: { right: 80, top: 20 } },
      plugins: {
        legend: { display: false },
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
            title: items => items[0].raw.label,
            label: ctx => {
              const pt = ctx.raw;
              return `  (${pt.x.toFixed(4)}, ${pt.y.toFixed(4)})`;
            }
          }
        },
        zoom: {
          zoom: {
            wheel: { enabled: true, speed: 0.08 },
            pinch: { enabled: true },
            mode: "xy",
          },
          pan: {
            enabled: true,
            mode: "xy",
          },
        },
      },
      scales: {
        x: {
          grid:   { color: "rgba(0,0,0,0.06)" },
          border: { color: "#d1d5db" },
          ticks:  { color: "#4b5563", font: { size: 13 }, maxTicksLimit: 7, padding: 6 },
          title:  { ...axisTitleStyle, text: axisLabels?.x ?? "" },
          afterDataLimits(axis) {
            const pad = (axis.max - axis.min) * 0.2 || 0.8;
            axis.max += pad; axis.min -= pad;
          }
        },
        y: {
          grid:   { color: "rgba(0,0,0,0.06)" },
          border: { color: "#d1d5db" },
          ticks:  { color: "#4b5563", font: { size: 13 }, maxTicksLimit: 7, padding: 6 },
          title:  { ...axisTitleStyle, text: axisLabels?.y ?? "" },
          afterDataLimits(axis) {
            const pad = (axis.max - axis.min) * 0.2 || 0.8;
            axis.max += pad; axis.min -= pad;
          }
        }
      }
    }
  });
}
