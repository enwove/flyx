// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Charts on a canvas, at the device's pixels: a sparkline, and series over the last minutes with an axis (or one at
// each side) and a tooltip.  Colours come from the page's tokens.
import { html, useEffect, useRef, useState } from "./html.js";
import { fmt, clock } from "./format.js";
import { t } from "./i18n.js";

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function setup(canvas) {
  const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, r.width, r.height);
  return { g, w: r.width, h: r.height };
}

function drawSpark(canvas, values) {
  if (!canvas.offsetParent) return;
  const { g, w, h } = setup(canvas);
  if (!values.length) return;
  const max = Math.max(1e-9, ...values);
  const n = values.length, step = n > 1 ? w / (n - 1) : 0;
  const pts = values.map((v, i) => [n > 1 ? i * step : w, h - 1.5 - (Math.max(0, v) / max) * (h - 4)]);
  g.beginPath(); g.moveTo(pts[0][0], h);
  for (const [x, y] of pts) g.lineTo(x, y);
  g.lineTo(pts[pts.length - 1][0], h); g.closePath();
  g.fillStyle = css("--chart-fill"); g.fill();
  g.beginPath();
  pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
  g.strokeStyle = css("--chart"); g.lineWidth = 1.5; g.lineJoin = "round"; g.stroke();
}

function niceStep(v) {  // 1, 2, 2.5 or 5 times a power of ten, four of which reach v
  const raw = v / 4, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
}

// An axis of four steps for the series on one side: to `o.top`, else to the highest sample in sight (at least
// `o.minTop`).
function axisOf(samples, series, side, o, now, span) {
  let top = o.top || 0;
  if (!top)
    for (const s of samples)
      if (now - s.t <= span) for (const se of series) if ((se.axis || "left") === side) top = Math.max(top, s[se.key] || 0);
  const step = niceStep(Math.max(top, o.minTop || 0) || 1);
  const lead = step / Math.pow(10, Math.floor(Math.log10(step)));
  const decimals = Math.max(0, -Math.floor(Math.log10(step)) + (lead === 2.5 ? 1 : 0));
  const label = (v) => v >= 10000 ? fmt(v / 1000, v % 1000 ? 1 : 0) + "k" : fmt(v, decimals) + (o.unit || "");
  return { top: 4 * step, step, label };
}

// Series over the last `span` seconds of samples [{t, key...}], a y axis to `top` (0: the data's), labelled.  With
// `opts.right` ({top, minTop, unit} as for the left), the series marked `axis: "right"` are drawn to a second axis at
// the right, its steps on the same grid lines.  Returns where the samples sit, for the tooltip.
function drawChart(canvas, samples, series, opts) {
  if (!canvas.offsetParent) return null;
  const { g, w, h } = setup(canvas);
  const twin = !!opts.right;
  const now = samples.length ? samples[samples.length - 1].t : Date.now() / 1000, span = opts.span;
  const axes = { left: axisOf(samples, series, "left", opts, now, span) };
  if (twin) axes.right = axisOf(samples, series, "right", opts.right, now, span);
  g.font = "10.5px " + css("--mono");
  // Each axis as wide as its widest label (and 8px from the plot).
  const widest = (ax) => Math.ceil(Math.max(...[0, 1, 2, 3, 4].map((i) => g.measureText(ax.label(ax.step * i)).width)));
  const left = Math.max(36, widest(axes.left) + 12), right = twin ? Math.max(40, widest(axes.right) + 12) : 4;
  const topPad = 8, bottom = 18, pw = w - left - right, ph = h - topPad - bottom;
  g.fillStyle = css("--faint");
  g.strokeStyle = css("--border"); g.lineWidth = 1;
  g.setLineDash([2, 3]);
  for (let i = 1; i <= 4; ++i) {
    const y = Math.round(topPad + ph - (i / 4) * ph) + 0.5;
    g.beginPath(); g.moveTo(left, y); g.lineTo(w - right, y); g.stroke();
  }
  g.setLineDash([]);
  g.beginPath(); g.moveTo(left, Math.round(topPad + ph) + 0.5); g.lineTo(w - right, Math.round(topPad + ph) + 0.5); g.stroke();
  g.textBaseline = "middle";
  for (let i = 0; i <= 4; ++i) {
    const y = topPad + ph - (i / 4) * ph;
    g.textAlign = "right"; g.fillText(axes.left.label(axes.left.step * i), left - 8, y);
    if (twin) { g.textAlign = "left"; g.fillText(axes.right.label(axes.right.step * i), w - right + 8, y); }
  }
  g.textAlign = "left"; g.textBaseline = "alphabetic";
  g.fillText(t("-" + Math.round(span / 60) + " min"), left, h - 3);
  g.textAlign = "right"; g.fillText(t("now"), w - right, h - 3);
  const xOf = (s) => left + pw - ((now - s) / span) * pw;
  for (const se of series) {
    const top = axes[se.axis || "left"].top;
    const yOf = (v) => topPad + ph - (Math.max(0, Math.min(top, v)) / top) * ph;
    const pts = samples.filter((s) => now - s.t <= span && s[se.key] !== null && s[se.key] !== undefined);
    // Runs of samples a second or so apart: a second without a value (no round to average) is a gap, not a line.
    const runs = [];
    for (const p of pts) {
      if (!runs.length || p.t - runs[runs.length - 1].at(-1).t > 1.5) runs.push([]);
      runs[runs.length - 1].push(p);
    }
    for (const run of runs) {
      if (run.length === 1) {  // a lone second: a dot
        g.beginPath(); g.arc(xOf(run[0].t), yOf(run[0][se.key]), 2.25, 0, 2 * Math.PI);
        g.fillStyle = css(se.color); g.fill();
        continue;
      }
      if (se.fill) {
        g.beginPath();
        g.moveTo(xOf(run[0].t), topPad + ph);
        for (const p of run) g.lineTo(xOf(p.t), yOf(p[se.key]));
        g.lineTo(xOf(run[run.length - 1].t), topPad + ph);
        g.closePath();
        g.fillStyle = css("--chart-fill"); g.fill();
      }
      g.beginPath();
      run.forEach((p, i) => (i ? g.lineTo(xOf(p.t), yOf(p[se.key])) : g.moveTo(xOf(p.t), yOf(p[se.key]))));
      g.strokeStyle = css(se.color); g.lineWidth = 1.75; g.lineJoin = "round"; g.stroke();
    }
  }
  return { left, pw, now, span, samples };
}

// A sparkline of the values, redrawn on every render.
export function Spark({ values, cls, label }) {
  const ref = useRef(null);
  useEffect(() => { if (ref.current) drawSpark(ref.current, values); });
  return html`<canvas class=${cls} ref=${ref} aria-label=${label}></canvas>`;
}

// A chart with its tooltip: `describe` writes a sample for the tooltip.  The tooltip sits in the nearest .pane.
export function Chart({ samples, series, opts, describe, label }) {
  const ref = useRef(null), geo = useRef(null);
  const [tip, setTip] = useState(null);
  useEffect(() => { if (ref.current) geo.current = drawChart(ref.current, samples, series, opts); });
  const move = (e) => {
    const g = geo.current, canvas = ref.current;
    if (!g || !g.samples.length) return;
    const r = canvas.getBoundingClientRect(), x = e.clientX - r.left;
    const at = g.now - ((g.left + g.pw - x) / g.pw) * g.span;
    let best = null;
    for (const s of g.samples) if (!best || Math.abs(s.t - at) < Math.abs(best.t - at)) best = s;
    if (!best || Math.abs(best.t - at) > 3) { setTip(null); return; }
    const card = canvas.closest(".pane").getBoundingClientRect();
    setTip({ text: clock(best.t) + "  " + describe(best), x: e.clientX - card.left, y: e.clientY - card.top, w: card.width });
  };
  return html`
    <canvas ref=${ref} role="img" aria-label=${label} onMouseMove=${move} onMouseLeave=${() => setTip(null)}></canvas>
    <div class="tip" style=${tip ? { display: "block", left: Math.min(tip.w - 160, Math.max(8, tip.x + 12)) + "px", top: tip.y - 32 + "px" } : {}}>
      ${tip ? tip.text : ""}</div>`;
}
