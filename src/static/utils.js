// ── PNG export ──────────────────────────────────────────────────
// Exports at 2× the canvas pixel size for crisp print/presentation quality.
function exportChartPNG(canvasId, filename) {
  const src   = document.getElementById(canvasId);
  const SCALE = 2;
  const out   = document.createElement("canvas");
  out.width   = src.width  * SCALE;
  out.height  = src.height * SCALE;
  const ctx   = out.getContext("2d");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.drawImage(src, 0, 0, out.width, out.height);
  const link  = document.createElement("a");
  link.download = filename + ".png";
  link.href   = out.toDataURL("image/png");
  link.click();
}

function exportAllCharts() {
  const charts = [
    ["chartZ", "outlet-bias-z"],
    ["chartA", "subject-discrimination-a"],
    ["chartB", "subject-baseline-b"],
  ];
  charts.forEach(([id, name], i) =>
    setTimeout(() => exportChartPNG(id, name), i * 250)
  );
}

// ── CSV parser ──────────────────────────────────────────────────
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) throw new Error("CSV must have a header row and at least one data row.");

  const headers = lines[0].split(",").map(h => h.trim());

  const REQUIRED = ["outlet", "subject", "mention_type", "amount_of_mentions"];
  for (const col of REQUIRED) {
    if (!headers.includes(col))
      throw new Error(`Missing required CSV column: "${col}"`);
  }

  const data = lines.slice(1).map((line, idx) => {
    const values = line.split(",").map(v => v.trim());
    if (values.length !== headers.length)
      throw new Error(`Row ${idx + 2} has ${values.length} columns, expected ${headers.length}.`);

    const row = {};
    headers.forEach((h, i) => { row[h] = values[i]; });

    const amount = parseInt(row.amount_of_mentions, 10);
    if (isNaN(amount))
      throw new Error(`Row ${idx + 2}: "amount_of_mentions" must be an integer, got "${row.amount_of_mentions}".`);
    row.amount_of_mentions = amount;

    return row;
  });

  return { data };
}

// ── Chart colours ───────────────────────────────────────────────
const COLORS = {
  z: { pos: "rgba(99,102,241,0.80)",  neg: "rgba(239,68,68,0.70)",  border_pos: "#6366f1", border_neg: "#ef4444" },
  a: { pos: "rgba(16,185,129,0.80)",  neg: "rgba(239,68,68,0.70)",  border_pos: "#10b981", border_neg: "#ef4444" },
  b: { pos: "rgba(245,158,11,0.80)",  neg: "rgba(239,68,68,0.70)",  border_pos: "#f59e0b", border_neg: "#ef4444" },
};

function barColors(values, palette) {
  return values.map(v => v >= 0 ? palette.pos : palette.neg);
}

function borderColors(values, palette) {
  return values.map(v => v >= 0 ? palette.border_pos : palette.border_neg);
}

// ── Chart factory ───────────────────────────────────────────────
// axisLabels: { x: string, y: string }
function createBarChart(canvasId, chartRef, labels, values, label, paletteKey, axisLabels) {
  if (chartRef) chartRef.destroy();

  const palette = COLORS[paletteKey] || COLORS.z;

  const axisTitleStyle = {
    display: true,
    color: "#374151",
    font: { size: 13, weight: "600", family: "'Inter', system-ui, sans-serif" },
  };

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
        borderRadius: 7,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 500, easing: "easeOutQuart" },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1e293b",
          borderColor: "#334155",
          borderWidth: 1,
          titleColor: "#f1f5f9",
          bodyColor:  "#94a3b8",
          padding: 13,
          titleFont: { size: 13, weight: "600" },
          bodyFont:  { size: 13 },
          callbacks: {
            label: ctx => `  ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`
          }
        }
      },
      scales: {
        x: {
          grid:   { color: "rgba(0,0,0,0.06)" },
          border: { color: "#d1d5db" },
          ticks:  { color: "#4b5563", font: { size: 14 }, padding: 6 },
          title:  { ...axisTitleStyle, text: axisLabels?.x ?? "" },
        },
        y: {
          grid:   { color: "rgba(0,0,0,0.06)" },
          border: { color: "#d1d5db" },
          ticks:  { color: "#4b5563", font: { size: 13 }, maxTicksLimit: 7, padding: 6 },
          title:  { ...axisTitleStyle, text: axisLabels?.y ?? "" },
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
