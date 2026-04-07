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

// ── SVG export ──────────────────────────────────────────────────
// Builds a true vector SVG from the Chart.js data model.
// No rasterisation — scales infinitely without quality loss.
function buildBarChartSVG(chart, W, H) {
  const PAD    = { top: 30, right: 40, bottom: 72, left: 72 };
  const plotW  = W - PAD.left - PAD.right;
  const plotH  = H - PAD.top  - PAD.bottom;
  const FONT   = "Inter, system-ui, -apple-system, sans-serif";

  const labels   = chart.data.labels;
  const values   = chart.data.datasets[0].data;
  const bgColors = chart.data.datasets[0].backgroundColor;
  const xTitle   = chart.options.scales.x.title.text || "";
  const yTitle   = chart.options.scales.y.title.text || "";
  const n        = values.length;

  // ── Y scale ──────────────────────────────────────────────────
  const rawMin = Math.min(...values, 0);
  const rawMax = Math.max(...values, 0);
  const pad    = (rawMax - rawMin) * 0.12 || 0.5;
  const yMin   = rawMin - pad;
  const yMax   = rawMax + pad;
  const yRange = yMax - yMin;

  const yPos = v => PAD.top + plotH * (1 - (v - yMin) / yRange);
  const zeroY = yPos(0);

  // ── Nice Y ticks ──────────────────────────────────────────────
  function niceStep(range, maxTicks) {
    const rough = range / maxTicks;
    const mag   = Math.pow(10, Math.floor(Math.log10(rough)));
    const norm  = rough / mag;
    const step  = [1, 2, 2.5, 5, 10].find(s => s >= norm) * mag;
    return step;
  }

  const step      = niceStep(yRange, 7);
  const tickStart = Math.ceil(yMin / step) * step;
  const ticks     = [];
  for (let t = tickStart; t <= yMax + step * 0.01; t += step)
    ticks.push(parseFloat(t.toPrecision(10)));

  // ── Helpers ───────────────────────────────────────────────────
  const f  = n => n.toFixed(1);
  const esc = s => String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const tickLabel = v => {
    if (Math.abs(v) < 1e-9) return "0";
    const s = v.toPrecision(4);
    return parseFloat(s).toString();
  };

  const parts = [];

  // Background
  parts.push(`<rect width="${W}" height="${H}" fill="#ffffff" rx="8"/>`);

  // Grid + Y tick labels
  for (const t of ticks) {
    const y = yPos(t);
    if (y < PAD.top - 1 || y > PAD.top + plotH + 1) continue;
    parts.push(`<line x1="${PAD.left}" y1="${f(y)}" x2="${PAD.left + plotW}" y2="${f(y)}" stroke="#e5e7eb" stroke-width="1"/>`);
    parts.push(`<text x="${PAD.left - 10}" y="${f(y + 4.5)}" text-anchor="end" fill="#4b5563" font-size="12" font-family="${FONT}">${esc(tickLabel(t))}</text>`);
  }

  // Zero line
  if (zeroY >= PAD.top && zeroY <= PAD.top + plotH)
    parts.push(`<line x1="${PAD.left}" y1="${f(zeroY)}" x2="${PAD.left + plotW}" y2="${f(zeroY)}" stroke="#9ca3af" stroke-width="1.5"/>`);

  // Axis borders
  parts.push(`<line x1="${PAD.left}" y1="${PAD.top}" x2="${PAD.left}" y2="${PAD.top + plotH}" stroke="#d1d5db" stroke-width="1.5"/>`);
  parts.push(`<line x1="${PAD.left}" y1="${PAD.top + plotH}" x2="${PAD.left + plotW}" y2="${PAD.top + plotH}" stroke="#d1d5db" stroke-width="1.5"/>`);

  // Bars + X labels
  const groupW = plotW / n;
  const barW   = Math.max(groupW * 0.55, 4);

  for (let i = 0; i < n; i++) {
    const cx      = PAD.left + groupW * (i + 0.5);
    const x       = cx - barW / 2;
    const v       = values[i];
    const barTop  = Math.min(yPos(v), zeroY);
    const barH    = Math.max(Math.abs(yPos(v) - zeroY), 1);
    const fill    = Array.isArray(bgColors) ? bgColors[i] : bgColors;
    const r       = Math.min(5, barH / 2);

    parts.push(`<rect x="${f(x)}" y="${f(barTop)}" width="${f(barW)}" height="${f(barH)}" rx="${r}" ry="${r}" fill="${esc(fill)}"/>`);
    parts.push(`<text x="${f(cx)}" y="${f(PAD.top + plotH + 18)}" text-anchor="middle" fill="#4b5563" font-size="13" font-family="${FONT}">${esc(labels[i])}</text>`);
  }

  // Axis titles
  if (xTitle)
    parts.push(`<text x="${f(PAD.left + plotW / 2)}" y="${H - 10}" text-anchor="middle" fill="#374151" font-size="13" font-weight="600" font-family="${FONT}">${esc(xTitle)}</text>`);

  if (yTitle)
    parts.push(`<text transform="translate(15,${f(PAD.top + plotH / 2)}) rotate(-90)" text-anchor="middle" fill="#374151" font-size="13" font-weight="600" font-family="${FONT}">${esc(yTitle)}</text>`);

  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`,
    ...parts,
    `</svg>`
  ].join("\n");
}

function exportChartSVG(chartRef, filename) {
  const svg  = buildBarChartSVG(chartRef, 900, 380);
  const blob = new Blob([svg], { type: "image/svg+xml" });
  const link = document.createElement("a");
  link.download = filename + ".svg";
  link.href = URL.createObjectURL(blob);
  link.click();
  URL.revokeObjectURL(link.href);
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
