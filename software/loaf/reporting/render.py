"""Renders a ReportData into a single self-contained HTML file.

No external dependencies (CDN scripts, fonts, etc.) - everything a training
report needs (a small SVG charting engine, styling, and the run's data) is
inlined, so the file can be opened straight from a `runs/<name>/` directory,
copied to a laptop, or emailed, without a server or an internet connection.
"""

from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from typing import Any

from loaf.reporting.collect import IndexData, ReportData, RunSummary

_SEVERITY_LABEL = {
    "good": "Good",
    "info": "Info",
    "warning": "Watch",
    "critical": "Critical",
}


def _json_safe(obj: Any) -> Any:
    """Replace NaN/Infinity with null - JSON.parse() rejects the bare tokens
    Python's json module otherwise emits for them."""
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        if value != value:  # NaN
            return "–"
        return f"{value:.{digits}f}"
    return escape(str(value))


def _hyperparam_rows(hyperparams: dict[str, Any]) -> str:
    rows = []
    for key, value in hyperparams.items():
        if value is None:
            continue
        if isinstance(value, list):
            value_str = ", ".join(str(v) for v in value)
        elif isinstance(value, float):
            value_str = f"{value:g}"
        else:
            value_str = str(value)
        rows.append(
            f"<tr><th scope='row'>{escape(key)}</th><td>{escape(value_str)}</td></tr>"
        )
    return "\n".join(rows)


def _training_curve_table(training_curve: list[dict[str, float]]) -> str:
    if not training_curve:
        return "<p class='muted'>No train_log.csv in this run directory.</p>"
    header = "<tr><th scope='col'>Epoch</th><th scope='col'>Train loss</th>" \
        "<th scope='col'>Val loss</th><th scope='col'>Val MAE</th>" \
        "<th scope='col'>Val RMSE</th><th scope='col'>Val skill</th></tr>"
    body_rows = "\n".join(
        f"<tr><td>{r['epoch']}</td><td>{_fmt(r['train_loss'])}</td>"
        f"<td>{_fmt(r['val_loss'])}</td><td>{_fmt(r['val_mae'])}</td>"
        f"<td>{_fmt(r['val_rmse'])}</td><td>{_fmt(r['val_skill'])}</td></tr>"
        for r in training_curve
    )
    return f"<table class='data-table'><thead>{header}</thead><tbody>{body_rows}</tbody></table>"


def _per_horizon_table(per_horizon: dict[str, list[dict[str, float]]]) -> str:
    if not per_horizon:
        return "<p class='muted'>Per-horizon breakdown unavailable (see note above).</p>"
    header = "<tr><th scope='col'>Variable</th><th scope='col'>Lead time</th>" \
        "<th scope='col'>RMSE</th><th scope='col'>MAE</th><th scope='col'>Skill</th></tr>"
    body_rows = []
    for var, rows in per_horizon.items():
        for r in rows:
            body_rows.append(
                f"<tr><td>{escape(var)}</td><td>{r['lead_hr']}h</td>"
                f"<td>{_fmt(r['rmse'])}</td><td>{_fmt(r['mae'])}</td>"
                f"<td>{_fmt(r['skill'])}</td></tr>"
            )
    return f"<table class='data-table'><thead>{header}</thead><tbody>{''.join(body_rows)}</tbody></table>"


def _residual_stats_table(residuals: dict[str, dict[str, Any]]) -> str:
    if not residuals:
        return ""
    header = "<tr><th scope='col'>Variable</th><th scope='col'>Mean error (bias)</th>" \
        "<th scope='col'>Std dev</th></tr>"
    body_rows = "\n".join(
        f"<tr><td>{escape(var)}</td><td>{_fmt(stats['mean'])}</td><td>{_fmt(stats['std'])}</td></tr>"
        for var, stats in residuals.items()
    )
    return f"<table class='data-table'><thead>{header}</thead><tbody>{body_rows}</tbody></table>"


def _notes_html(notes: list[dict[str, str]]) -> str:
    items = []
    for note in notes:
        severity = note.get("severity", "info")
        label = _SEVERITY_LABEL.get(severity, "Info")
        items.append(
            f"<li class='note note-{escape(severity)}'>"
            f"<span class='note-chip'>{escape(label)}</span>"
            f"<span class='note-text'>{escape(note['text'])}</span></li>"
        )
    return f"<ul class='notes-list'>{''.join(items)}</ul>"


def _stat_tiles(data: ReportData) -> str:
    metrics = data.final_metrics
    skill = metrics.get("skill")
    skill_class = ""
    if skill is not None and skill == skill:
        skill_class = "stat-good" if skill >= 0.1 else ("stat-critical" if skill < 0 else "stat-warn")

    tiles = [
        ("Best epoch", data.best_epoch if data.best_epoch is not None else "–", ""),
        ("Val RMSE", _fmt(metrics.get("rmse")), ""),
        ("Val MAE", _fmt(metrics.get("mae")), ""),
        ("Val skill vs persistence", _fmt(metrics.get("skill")), skill_class),
        ("Parameters", f"{data.hyperparams.get('n_params'):,}" if data.hyperparams.get("n_params") else "–", ""),
        (
            "Train / val samples",
            f"{data.hyperparams.get('n_train_samples', '–')} / "
            f"{data.hyperparams.get('n_val_samples', '–')}",
            "",
        ),
    ]
    tiles_html = "\n".join(
        f"<div class='stat-tile {cls}'><div class='stat-label'>{escape(label)}</div>"
        f"<div class='stat-value'>{escape(str(value))}</div></div>"
        for label, value, cls in tiles
    )
    return f"<div class='stat-grid'>{tiles_html}</div>"


_PAGE_TEMPLATE = """\
<title>%%TITLE%%</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7;
    --surface-1: #fcfcfb;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --grid-line: #e1e0d9;
    --axis-line: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --series-1: #2a78d6;
    --series-2: #eb6834;
    --good: #0ca30c;
    --warning: #fab219;
    --serious: #ec835a;
    --critical: #d03b3b;
    --success-text: #006300;
    --tooltip-bg: #0b0b0b;
    --tooltip-fg: #ffffff;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --page: #0d0d0d;
      --surface-1: #1a1a19;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --grid-line: #2c2c2a;
      --axis-line: #383835;
      --border: rgba(255,255,255,0.10);
      --series-1: #3987e5;
      --series-2: #d95926;
      --critical: #e66767;
      --success-text: #0ca30c;
      --tooltip-bg: #fcfcfb;
      --tooltip-fg: #0b0b0b;
    }
  }
  * { box-sizing: border-box; }
  body {
    background: var(--page);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    margin: 0;
    padding: 24px 16px 64px;
  }
  .wrap { max-width: 1080px; margin: 0 auto; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  h2 { font-size: 1.1rem; margin: 0 0 12px; }
  .meta { color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 24px; }
  .meta code {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 4px; padding: 1px 5px; font-size: 0.82em;
  }
  .badge {
    display: inline-block; background: var(--series-1); color: #fff;
    border-radius: 999px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.02em; margin-left: 8px;
  }
  .card {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px; margin-bottom: 20px;
  }
  .stat-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
  }
  .stat-tile {
    border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px;
    background: var(--page);
  }
  .stat-label { color: var(--text-secondary); font-size: 0.78rem; margin-bottom: 4px; }
  .stat-value { font-size: 1.5rem; font-weight: 600; }
  .stat-good .stat-value { color: var(--good); }
  .stat-warn .stat-value { color: var(--warning); }
  .stat-critical .stat-value { color: var(--critical); }
  .notes-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 10px; }
  .note { display: flex; align-items: flex-start; gap: 10px; font-size: 0.92rem; line-height: 1.45; }
  .note-chip {
    flex: 0 0 auto; font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.03em; padding: 2px 8px; border-radius: 999px; margin-top: 1px;
    color: #fff;
  }
  .note-good .note-chip { background: var(--good); }
  .note-info .note-chip { background: var(--text-muted); }
  .note-warning .note-chip { background: var(--serious); }
  .note-critical .note-chip { background: var(--critical); }
  .note-text { color: var(--text-secondary); }
  .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }
  .chart-title { font-size: 0.95rem; font-weight: 600; margin: 0 0 2px; }
  .chart-sub { font-size: 0.78rem; color: var(--text-muted); margin: 0 0 10px; }
  .chart-svg-wrap { position: relative; width: 100%; overflow-x: auto; }
  svg.chart { width: 100%; height: auto; display: block; }
  .axis-label { fill: var(--text-muted); font-size: 10px; }
  .tick-label { fill: var(--text-muted); font-size: 10px; }
  .gridline { stroke: var(--grid-line); stroke-width: 1; }
  .axis-line { stroke: var(--axis-line); stroke-width: 1; }
  .ref-line { stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 3 3; }
  .crosshair { stroke: var(--text-muted); stroke-width: 1; pointer-events: none; }
  .legend { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 8px; font-size: 0.8rem; }
  .legend-item { display: flex; align-items: center; gap: 6px; color: var(--text-secondary); }
  .legend-swatch { width: 14px; height: 2px; border-radius: 1px; }
  .chart-tooltip {
    position: absolute; pointer-events: none; background: var(--tooltip-bg); color: var(--tooltip-fg);
    border-radius: 6px; padding: 6px 10px; font-size: 0.78rem; line-height: 1.4;
    opacity: 0; transform: translate(-50%, -110%); transition: opacity 0.08s ease; white-space: nowrap;
    z-index: 5;
  }
  .chart-tooltip.visible { opacity: 1; }
  .chart-tooltip .tt-row { display: flex; align-items: center; gap: 6px; }
  .chart-tooltip .tt-key { display: inline-block; width: 10px; height: 2px; }
  .chart-tooltip .tt-value { font-weight: 700; }
  .chart-tooltip .tt-name { opacity: 0.8; }
  details.table-toggle { margin-top: 10px; }
  details.table-toggle > summary {
    cursor: pointer; font-size: 0.8rem; color: var(--text-secondary); user-select: none;
  }
  table.data-table {
    width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.8rem;
    font-variant-numeric: tabular-nums;
  }
  table.data-table th, table.data-table td {
    text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--grid-line);
  }
  table.data-table thead th { color: var(--text-muted); font-weight: 600; }
  table.hyperparam-table th { width: 220px; color: var(--text-secondary); font-weight: 500; }
  .muted { color: var(--text-muted); font-size: 0.85rem; }
  .error-banner {
    border: 1px solid var(--critical); border-radius: 8px; padding: 12px 16px;
    color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 20px;
  }
  .error-banner strong { color: var(--critical); }
  footer { color: var(--text-muted); font-size: 0.78rem; margin-top: 24px; }
</style>
<div class="wrap">
  <header>
    <h1>LOAF training report<span class="badge">%%MODEL_TYPE%%</span></h1>
    <div class="meta">
      Run <code>%%RUN_NAME%%</code> &middot; generated %%GENERATED_AT%% &middot;
      checkpoint <code>%%CHECKPOINT_PATH%%</code>
    </div>
  </header>

  %%ERROR_BANNER%%

  <section class="card">
    <h2>Summary</h2>
    %%STAT_TILES%%
  </section>

  <section class="card">
    <h2>Fine-tuning notes</h2>
    %%NOTES%%
  </section>

  <section class="card">
    <h2>Training curves</h2>
    <div class="charts-grid">
      <div>
        <p class="chart-title">Loss</p>
        <p class="chart-sub">Train vs. validation loss (MSE) per epoch - the gap between them is the overfitting signal.</p>
        <div class="chart-svg-wrap" id="chart-loss"></div>
      </div>
      <div>
        <p class="chart-title">Validation error</p>
        <p class="chart-sub">MAE and RMSE in physical units, per epoch.</p>
        <div class="chart-svg-wrap" id="chart-error"></div>
      </div>
      <div>
        <p class="chart-title">Skill vs. persistence</p>
        <p class="chart-sub">1 &minus; (model MSE / naive "no change" MSE). Above 0 means the model beats persistence.</p>
        <div class="chart-svg-wrap" id="chart-skill"></div>
      </div>
    </div>
    <details class="table-toggle">
      <summary>View training curve as a table</summary>
      %%TRAINING_TABLE%%
    </details>
  </section>

  %%HORIZON_SECTION%%

  %%SCATTER_SECTION%%

  %%RESIDUAL_SECTION%%

  <section class="card">
    <h2>Run configuration</h2>
    <table class="data-table hyperparam-table">
      <tbody>
        %%HYPERPARAM_ROWS%%
      </tbody>
    </table>
  </section>

  <footer>Generated by <code>loaf-report</code> (loaf.reporting). Re-run it after retraining to refresh this file.</footer>
</div>

<script id="report-data" type="application/json">%%REPORT_JSON%%</script>
<script>
%%CHART_JS%%
</script>
"""

_CHART_JS = r"""
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("report-data").textContent);
  const NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs) {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }

  function fmt(v, digits) {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    digits = digits === undefined ? 2 : digits;
    return Number(v).toFixed(digits);
  }

  function niceTicks(min, max, count) {
    if (min === max) { min -= 1; max += 1; }
    const span = max - min;
    const rawStep = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const norm = rawStep / mag;
    const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
    const start = Math.ceil(min / step) * step;
    const ticks = [];
    for (let v = start; v <= max + step * 1e-6; v += step) {
      ticks.push(Math.round(v / step) * step);
    }
    if (ticks.length === 0) ticks.push(min, max);
    return ticks;
  }

  function scale(domain, range) {
    const span = domain[1] - domain[0];
    const m = span === 0 ? 0 : (range[1] - range[0]) / span;
    return function (v) { return range[0] + (v - domain[0]) * m; };
  }

  function pad(domain, frac) {
    const span = domain[1] - domain[0] || Math.abs(domain[0]) || 1;
    return [domain[0] - span * frac, domain[1] + span * frac];
  }

  const MARGIN = { top: 10, right: 16, bottom: 28, left: 46 };
  const WIDTH = 480;
  const HEIGHT = 220;

  function makeTooltip(wrap) {
    const tip = document.createElement("div");
    tip.className = "chart-tooltip";
    wrap.appendChild(tip);
    return tip;
  }

  function positionTooltip(tip, wrap, px, py) {
    const bounds = wrap.getBoundingClientRect();
    const scaleX = bounds.width / WIDTH;
    tip.style.left = (px * scaleX) + "px";
    tip.style.top = (py * (bounds.height / HEIGHT)) + "px";
  }

  // ---- Line chart: one or more series sharing an x axis (epoch or lead_hr). ----
  function lineChart(containerId, series, opts) {
    const container = document.getElementById(containerId);
    if (!container) return;
    if (!series.length || !series[0].points.length) {
      container.innerHTML = "<p class='muted'>No data.</p>";
      return;
    }
    opts = opts || {};
    const plotW = WIDTH - MARGIN.left - MARGIN.right;
    const plotH = HEIGHT - MARGIN.top - MARGIN.bottom;

    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    series.forEach(function (s) {
      s.points.forEach(function (p) {
        if (p[0] < xMin) xMin = p[0];
        if (p[0] > xMax) xMax = p[0];
        if (p[1] !== null && !Number.isNaN(p[1])) {
          if (p[1] < yMin) yMin = p[1];
          if (p[1] > yMax) yMax = p[1];
        }
      });
    });
    if (opts.includeZero) { yMin = Math.min(yMin, 0); yMax = Math.max(yMax, 0); }
    const yDomain = pad([yMin, yMax], 0.1);
    const xScale = scale([xMin, xMax], [0, plotW]);
    const yScale = scale(yDomain, [plotH, 0]);

    const svg = el("svg", { class: "chart", viewBox: "0 0 " + WIDTH + " " + HEIGHT });
    const plot = el("g", { transform: "translate(" + MARGIN.left + "," + MARGIN.top + ")" });
    svg.appendChild(plot);

    const yTicks = niceTicks(yDomain[0], yDomain[1], 4);
    yTicks.forEach(function (t) {
      const y = yScale(t);
      plot.appendChild(el("line", { class: "gridline", x1: 0, x2: plotW, y1: y, y2: y }));
      const label = el("text", { class: "tick-label", x: -6, y: y + 3, "text-anchor": "end" });
      label.textContent = opts.yTickFormat ? opts.yTickFormat(t) : fmt(t, 1);
      plot.appendChild(label);
    });

    const xTicks = niceTicks(xMin, xMax, 5);
    xTicks.forEach(function (t) {
      const x = xScale(t);
      const label = el("text", { class: "tick-label", x: x, y: plotH + 16, "text-anchor": "middle" });
      label.textContent = opts.xTickFormat ? opts.xTickFormat(t) : String(Math.round(t));
      plot.appendChild(label);
    });
    plot.appendChild(el("line", { class: "axis-line", x1: 0, x2: plotW, y1: plotH, y2: plotH }));

    if (opts.includeZero && yDomain[0] < 0) {
      plot.appendChild(el("line", { class: "ref-line", x1: 0, x2: plotW, y1: yScale(0), y2: yScale(0) }));
    }

    series.forEach(function (s) {
      const pts = s.points.filter(function (p) { return p[1] !== null && !Number.isNaN(p[1]); });
      const d = pts.map(function (p, i) {
        return (i === 0 ? "M" : "L") + xScale(p[0]).toFixed(2) + "," + yScale(p[1]).toFixed(2);
      }).join(" ");
      plot.appendChild(el("path", { d: d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
      if (pts.length) {
        const last = pts[pts.length - 1];
        const dot = el("circle", { cx: xScale(last[0]), cy: yScale(last[1]), r: 4, fill: s.color, stroke: "var(--surface-1)", "stroke-width": 2 });
        plot.appendChild(dot);
      }
    });

    // Crosshair + shared tooltip, driven by nearest x.
    const hit = el("rect", { x: 0, y: 0, width: plotW, height: plotH, fill: "transparent", tabindex: "0", style: "outline:none;cursor:crosshair" });
    const crosshair = el("line", { class: "crosshair", x1: 0, x2: 0, y1: 0, y2: plotH, style: "display:none" });
    plot.appendChild(crosshair);
    plot.appendChild(hit);

    container.innerHTML = "";
    if (series.length > 1) {
      const legend = document.createElement("div");
      legend.className = "legend";
      series.forEach(function (s) {
        const item = document.createElement("span");
        item.className = "legend-item";
        const sw = document.createElement("span");
        sw.className = "legend-swatch";
        sw.style.background = s.color;
        const label = document.createElement("span");
        label.textContent = s.name;
        item.appendChild(sw);
        item.appendChild(label);
        legend.appendChild(item);
      });
      container.appendChild(legend);
    }
    container.appendChild(svg);
    const tip = makeTooltip(container);

    const xValues = series[0].points.map(function (p) { return p[0]; });
    function showAt(index) {
      const xv = xValues[index];
      const px = xScale(xv);
      crosshair.setAttribute("x1", px);
      crosshair.setAttribute("x2", px);
      crosshair.style.display = "block";
      let html = "<div class='tt-row'><span class='tt-value'>" + (opts.xTickFormat ? opts.xTickFormat(xv) : xv) + "</span></div>";
      series.forEach(function (s) {
        const p = s.points[index];
        if (!p) return;
        html += "<div class='tt-row'><span class='tt-key' style='background:" + s.color + "'></span>" +
          "<span class='tt-value'>" + fmt(p[1], opts.digits) + "</span>" +
          "<span class='tt-name'>" + s.name + "</span></div>";
      });
      tip.innerHTML = html;
      tip.classList.add("visible");
      positionTooltip(tip, container, MARGIN.left + px, MARGIN.top + plotH * 0.15);
    }
    function nearestIndex(px) {
      let best = 0, bestDist = Infinity;
      xValues.forEach(function (xv, i) {
        const d = Math.abs(xScale(xv) - px);
        if (d < bestDist) { bestDist = d; best = i; }
      });
      return best;
    }
    hit.addEventListener("pointermove", function (evt) {
      const rect = svg.getBoundingClientRect();
      const px = ((evt.clientX - rect.left) / rect.width) * WIDTH - MARGIN.left;
      showAt(nearestIndex(px));
    });
    hit.addEventListener("pointerleave", function () {
      crosshair.style.display = "none";
      tip.classList.remove("visible");
    });
    let focusIndex = xValues.length - 1;
    hit.addEventListener("focus", function () { showAt(focusIndex); });
    hit.addEventListener("blur", function () {
      crosshair.style.display = "none";
      tip.classList.remove("visible");
    });
    hit.addEventListener("keydown", function (evt) {
      if (evt.key === "ArrowLeft") { focusIndex = Math.max(0, focusIndex - 1); showAt(focusIndex); evt.preventDefault(); }
      else if (evt.key === "ArrowRight") { focusIndex = Math.min(xValues.length - 1, focusIndex + 1); showAt(focusIndex); evt.preventDefault(); }
    });
  }

  // ---- Scatter: actual (x) vs predicted (y), with a y=x reference line. ----
  function scatterChart(containerId, points, opts) {
    const container = document.getElementById(containerId);
    if (!container) return;
    if (!points.length) {
      container.innerHTML = "<p class='muted'>No sampled predictions.</p>";
      return;
    }
    opts = opts || {};
    const plotW = WIDTH - MARGIN.left - MARGIN.right;
    const plotH = HEIGHT - MARGIN.top - MARGIN.bottom;

    let lo = Infinity, hi = -Infinity;
    points.forEach(function (p) {
      lo = Math.min(lo, p[0], p[1]);
      hi = Math.max(hi, p[0], p[1]);
    });
    const domain = pad([lo, hi], 0.06);
    const xScale = scale(domain, [0, plotW]);
    const yScale = scale(domain, [plotH, 0]);

    const svg = el("svg", { class: "chart", viewBox: "0 0 " + WIDTH + " " + HEIGHT });
    const plot = el("g", { transform: "translate(" + MARGIN.left + "," + MARGIN.top + ")" });
    svg.appendChild(plot);

    const ticks = niceTicks(domain[0], domain[1], 4);
    ticks.forEach(function (t) {
      const y = yScale(t), x = xScale(t);
      plot.appendChild(el("line", { class: "gridline", x1: 0, x2: plotW, y1: y, y2: y }));
      const yl = el("text", { class: "tick-label", x: -6, y: y + 3, "text-anchor": "end" });
      yl.textContent = fmt(t, 1);
      plot.appendChild(yl);
      const xl = el("text", { class: "tick-label", x: x, y: plotH + 16, "text-anchor": "middle" });
      xl.textContent = fmt(t, 1);
      plot.appendChild(xl);
    });
    plot.appendChild(el("line", { class: "axis-line", x1: 0, x2: plotW, y1: plotH, y2: plotH }));
    plot.appendChild(el("line", { class: "ref-line", x1: xScale(domain[0]), x2: xScale(domain[1]), y1: yScale(domain[0]), y2: yScale(domain[1]) }));

    const dots = [];
    points.forEach(function (p) {
      const cx = xScale(p[0]), cy = yScale(p[1]);
      const dot = el("circle", { cx: cx, cy: cy, r: 2.6, fill: opts.color || "var(--series-1)", opacity: 0.4 });
      plot.appendChild(dot);
      dots.push({ x: cx, y: cy, actual: p[0], predicted: p[1], node: dot });
    });

    container.innerHTML = "";
    container.appendChild(svg);
    const tip = makeTooltip(container);
    const hit = el("rect", { x: 0, y: 0, width: plotW, height: plotH, fill: "transparent", style: "cursor:crosshair" });
    plot.appendChild(hit);
    let highlighted = null;

    hit.addEventListener("pointermove", function (evt) {
      const rect = svg.getBoundingClientRect();
      const px = ((evt.clientX - rect.left) / rect.width) * WIDTH - MARGIN.left;
      const py = ((evt.clientY - rect.top) / rect.height) * HEIGHT - MARGIN.top;
      let best = null, bestDist = 400; // 20px radius, squared
      dots.forEach(function (d) {
        const dist = (d.x - px) * (d.x - px) + (d.y - py) * (d.y - py);
        if (dist < bestDist) { bestDist = dist; best = d; }
      });
      if (highlighted && highlighted !== best) {
        highlighted.node.setAttribute("r", 2.6);
        highlighted.node.setAttribute("opacity", 0.4);
      }
      if (best) {
        best.node.setAttribute("r", 5);
        best.node.setAttribute("opacity", 1);
        highlighted = best;
        const err = best.predicted - best.actual;
        tip.innerHTML =
          "<div class='tt-row'><span class='tt-value'>" + fmt(best.actual) + "</span><span class='tt-name'>actual</span></div>" +
          "<div class='tt-row'><span class='tt-value'>" + fmt(best.predicted) + "</span><span class='tt-name'>predicted</span></div>" +
          "<div class='tt-row'><span class='tt-value'>" + (err >= 0 ? "+" : "") + fmt(err) + "</span><span class='tt-name'>error</span></div>";
        tip.classList.add("visible");
        positionTooltip(tip, container, MARGIN.left + best.x, MARGIN.top + best.y);
      } else {
        tip.classList.remove("visible");
      }
    });
    hit.addEventListener("pointerleave", function () {
      if (highlighted) { highlighted.node.setAttribute("r", 2.6); highlighted.node.setAttribute("opacity", 0.4); highlighted = null; }
      tip.classList.remove("visible");
    });

    const n = points.length;
    const rmse = Math.sqrt(points.reduce(function (acc, p) { return acc + (p[1] - p[0]) * (p[1] - p[0]); }, 0) / n);
    const stats = document.createElement("p");
    stats.className = "chart-sub";
    stats.style.marginTop = "6px";
    stats.textContent = "n=" + n + " sampled points, RMSE " + fmt(rmse) + " (dashed line = perfect prediction)";
    container.appendChild(stats);
  }

  // ---- Histogram: prediction residuals (pred - actual). ----
  function histChart(containerId, hist, opts) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const edges = hist.bin_edges, counts = hist.counts;
    if (!edges.length) {
      container.innerHTML = "<p class='muted'>No sampled predictions.</p>";
      return;
    }
    opts = opts || {};
    const plotW = WIDTH - MARGIN.left - MARGIN.right;
    const plotH = HEIGHT - MARGIN.top - MARGIN.bottom;

    const xDomain = [edges[0], edges[edges.length - 1]];
    const yMax = Math.max.apply(null, counts);
    const xScale = scale(xDomain, [0, plotW]);
    const yScale = scale([0, yMax], [plotH, 0]);

    const svg = el("svg", { class: "chart", viewBox: "0 0 " + WIDTH + " " + HEIGHT });
    const plot = el("g", { transform: "translate(" + MARGIN.left + "," + MARGIN.top + ")" });
    svg.appendChild(plot);

    niceTicks(0, yMax, 4).forEach(function (t) {
      const y = yScale(t);
      plot.appendChild(el("line", { class: "gridline", x1: 0, x2: plotW, y1: y, y2: y }));
      const label = el("text", { class: "tick-label", x: -6, y: y + 3, "text-anchor": "end" });
      label.textContent = String(Math.round(t));
      plot.appendChild(label);
    });
    niceTicks(xDomain[0], xDomain[1], 5).forEach(function (t) {
      const x = xScale(t);
      const label = el("text", { class: "tick-label", x: x, y: plotH + 16, "text-anchor": "middle" });
      label.textContent = fmt(t, 1);
      plot.appendChild(label);
    });
    plot.appendChild(el("line", { class: "axis-line", x1: 0, x2: plotW, y1: plotH, y2: plotH }));
    if (xDomain[0] < 0 && xDomain[1] > 0) {
      plot.appendChild(el("line", { class: "ref-line", x1: xScale(0), x2: xScale(0), y1: 0, y2: plotH }));
    }

    const bars = [];
    const gap = 1;
    for (let i = 0; i < counts.length; i++) {
      const x0 = xScale(edges[i]), x1 = xScale(edges[i + 1]);
      const barH = plotH - yScale(counts[i]);
      const rect = el("rect", {
        x: x0 + gap / 2, y: yScale(counts[i]), width: Math.max(0, x1 - x0 - gap), height: barH,
        fill: opts.color || "var(--series-1)", rx: 2,
      });
      plot.appendChild(rect);
      bars.push({ node: rect, lo: edges[i], hi: edges[i + 1], count: counts[i] });
    }
    if (typeof hist.mean === "number") {
      plot.appendChild(el("line", { class: "ref-line", stroke: "var(--series-2)", x1: xScale(hist.mean), x2: xScale(hist.mean), y1: 0, y2: plotH }));
    }

    container.innerHTML = "";
    container.appendChild(svg);
    const tip = makeTooltip(container);
    bars.forEach(function (b) {
      b.node.addEventListener("pointermove", function (evt) {
        const rect = svg.getBoundingClientRect();
        const px = ((evt.clientX - rect.left) / rect.width) * WIDTH - MARGIN.left;
        tip.innerHTML = "<div class='tt-row'><span class='tt-value'>" + b.count + "</span><span class='tt-name'>samples, [" + fmt(b.lo, 2) + ", " + fmt(b.hi, 2) + ")</span></div>";
        tip.classList.add("visible");
        positionTooltip(tip, container, MARGIN.left + px, MARGIN.top + plotH * 0.2);
        b.node.setAttribute("opacity", 0.75);
      });
      b.node.addEventListener("pointerleave", function () {
        tip.classList.remove("visible");
        b.node.setAttribute("opacity", 1);
      });
    });
  }

  // ---- Wire up the report's charts from the embedded data. ----
  const SERIES_COLOR = ["var(--series-1)", "var(--series-2)"];

  if (DATA.training_curve.length) {
    lineChart("chart-loss", [
      { name: "train loss", color: SERIES_COLOR[0], points: DATA.training_curve.map(function (r) { return [r.epoch, r.train_loss]; }) },
      { name: "val loss", color: SERIES_COLOR[1], points: DATA.training_curve.map(function (r) { return [r.epoch, r.val_loss]; }) },
    ], { xTickFormat: function (v) { return "e" + Math.round(v); }, digits: 3 });

    lineChart("chart-error", [
      { name: "val MAE", color: SERIES_COLOR[0], points: DATA.training_curve.map(function (r) { return [r.epoch, r.val_mae]; }) },
      { name: "val RMSE", color: SERIES_COLOR[1], points: DATA.training_curve.map(function (r) { return [r.epoch, r.val_rmse]; }) },
    ], { xTickFormat: function (v) { return "e" + Math.round(v); }, digits: 3 });

    lineChart("chart-skill", [
      { name: "skill", color: SERIES_COLOR[0], points: DATA.training_curve.map(function (r) { return [r.epoch, r.val_skill]; }) },
    ], { xTickFormat: function (v) { return "e" + Math.round(v); }, includeZero: true, digits: 3 });
  }

  const vars = DATA.target_vars || [];
  if (Object.keys(DATA.per_horizon).length) {
    lineChart("chart-horizon-rmse", vars.map(function (v, i) {
      return { name: v, color: SERIES_COLOR[i % 2], points: (DATA.per_horizon[v] || []).map(function (r) { return [r.lead_hr, r.rmse]; }) };
    }), { xTickFormat: function (v) { return v + "h"; }, digits: 3 });

    lineChart("chart-horizon-skill", vars.map(function (v, i) {
      return { name: v, color: SERIES_COLOR[i % 2], points: (DATA.per_horizon[v] || []).map(function (r) { return [r.lead_hr, r.skill]; }) };
    }), { xTickFormat: function (v) { return v + "h"; }, includeZero: true, digits: 3 });
  }

  vars.forEach(function (v) {
    const pts = DATA.scatter[v];
    if (pts && pts.length) scatterChart("chart-scatter-" + v, pts, {});
    const hist = DATA.residuals[v];
    if (hist && hist.bin_edges && hist.bin_edges.length) histChart("chart-residual-" + v, hist, {});
  });
})();
"""


def render_html(data: ReportData, output_path: str | Path) -> Path:
    """Render `data` to a standalone HTML file at `output_path`."""
    output_path = Path(output_path)

    if data.per_horizon:
        var_scatter_cards = "".join(
            f"<div><p class='chart-title'>{escape(var)}</p>"
            f"<p class='chart-sub'>Predicted vs. actual, sampled validation points.</p>"
            f"<div class='chart-svg-wrap' id='chart-scatter-{escape(var)}'></div></div>"
            for var in data.target_vars
        )
        residual_cards = "".join(
            f"<div><p class='chart-title'>{escape(var)}</p>"
            f"<p class='chart-sub'>Prediction error (predicted &minus; actual). Centered on 0 with a tight "
            f"spread is the goal; a shifted mean (orange line) means the model is biased.</p>"
            f"<div class='chart-svg-wrap' id='chart-residual-{escape(var)}'></div></div>"
            for var in data.target_vars
        )
        horizon_section = f"""
  <section class="card">
    <h2>Accuracy by forecast horizon</h2>
    <div class="charts-grid">
      <div>
        <p class="chart-title">RMSE by lead time</p>
        <p class="chart-sub">Where the model's error grows fastest as it forecasts further out.</p>
        <div class="chart-svg-wrap" id="chart-horizon-rmse"></div>
      </div>
      <div>
        <p class="chart-title">Skill by lead time</p>
        <p class="chart-sub">Skill vs. persistence at each horizon - persistence usually wins at short lead times.</p>
        <div class="chart-svg-wrap" id="chart-horizon-skill"></div>
      </div>
    </div>
    <details class="table-toggle">
      <summary>View per-horizon breakdown as a table</summary>
      {_per_horizon_table(data.per_horizon)}
    </details>
  </section>
"""
        scatter_section = f"""
  <section class="card">
    <h2>Predicted vs. actual</h2>
    <div class="charts-grid">{var_scatter_cards}</div>
  </section>
"""
        residual_section = f"""
  <section class="card">
    <h2>Residual distribution</h2>
    <div class="charts-grid">{residual_cards}</div>
    <details class="table-toggle">
      <summary>View residual stats as a table</summary>
      {_residual_stats_table(data.residuals)}
    </details>
  </section>
"""
    else:
        horizon_section = ""
        scatter_section = ""
        residual_section = ""

    error_banner = ""
    if data.inference_error:
        error_banner = (
            "<div class='error-banner'><strong>Per-horizon/scatter/residual sections skipped.</strong> "
            f"Re-running validation inference failed: {escape(data.inference_error)} "
            "Training-curve metrics above still come from train_log.csv.</div>"
        )

    payload = _json_safe(
        {
            "training_curve": data.training_curve,
            "target_vars": data.target_vars,
            "per_horizon": data.per_horizon,
            "scatter": data.scatter,
            "residuals": data.residuals,
        }
    )

    html = _PAGE_TEMPLATE
    html = html.replace("%%TITLE%%", escape(f"LOAF report — {data.run_name}"))
    html = html.replace("%%MODEL_TYPE%%", escape(data.model_type))
    html = html.replace("%%RUN_NAME%%", escape(data.run_name))
    html = html.replace("%%GENERATED_AT%%", escape(data.generated_at))
    html = html.replace("%%CHECKPOINT_PATH%%", escape(data.checkpoint_path))
    html = html.replace("%%ERROR_BANNER%%", error_banner)
    html = html.replace("%%STAT_TILES%%", _stat_tiles(data))
    html = html.replace("%%NOTES%%", _notes_html(data.notes))
    html = html.replace("%%TRAINING_TABLE%%", _training_curve_table(data.training_curve))
    html = html.replace("%%HORIZON_SECTION%%", horizon_section)
    html = html.replace("%%SCATTER_SECTION%%", scatter_section)
    html = html.replace("%%RESIDUAL_SECTION%%", residual_section)
    html = html.replace("%%HYPERPARAM_ROWS%%", _hyperparam_rows(data.hyperparams))
    # Escape "</" so a stray "</script>"-like substring in the data (e.g. an
    # unusual variable name) can't break out of the inline <script> block.
    report_json = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    html = html.replace("%%REPORT_JSON%%", report_json)
    html = html.replace("%%CHART_JS%%", _CHART_JS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path


# ---------------------------------------------------------------------------
# Cross-run "master report" - a fast comparison across every run under a
# runs/ directory. Self-contained the same way render_html()'s single-run
# report is; keeps its own copy of the small SVG charting engine (plus a
# categorical bar chart it needs that the single-run report doesn't) rather
# than sharing code with _CHART_JS, so each generated HTML file stays a
# fully independent, offline-readable artifact.
# ---------------------------------------------------------------------------

_CELL_CLASS = {"good": "cell-good", "warn": "cell-warn", "critical": "cell-critical"}


def _skill_class(skill: float | None) -> str:
    if skill is None or skill != skill:  # None or NaN
        return ""
    if skill < 0:
        return _CELL_CLASS["critical"]
    if skill < 0.1:
        return _CELL_CLASS["warn"]
    return _CELL_CLASS["good"]


def _index_stat_tiles(runs: list[RunSummary]) -> str:
    valid = [r for r in runs if not r.error]
    scored = [r for r in valid if r.final_metrics.get("skill") == r.final_metrics.get("skill")]

    tiles: list[tuple[str, str, str]] = [("Total runs", str(len(runs)), "")]
    if scored:
        best = max(scored, key=lambda r: r.final_metrics["skill"])
        best_skill = best.final_metrics["skill"]
        tiles.append(
            (
                "Best skill so far",
                f"{best_skill:.3f} ({best.display_date or best.run_name})",
                _skill_class(best_skill).replace("cell-", "stat-"),
            )
        )
    if valid:
        latest = valid[-1]
        tiles.append(("Most recent run", latest.display_date or latest.run_name, ""))
        latest_skill = latest.final_metrics.get("skill")
        if latest_skill is not None and latest_skill == latest_skill:
            tiles.append(
                (
                    "Most recent skill",
                    f"{latest_skill:.3f}",
                    _skill_class(latest_skill).replace("cell-", "stat-"),
                )
            )
    tiles_html = "\n".join(
        f"<div class='stat-tile {cls}'><div class='stat-label'>{escape(label)}</div>"
        f"<div class='stat-value'>{escape(value)}</div></div>"
        for label, value, cls in tiles
    )
    return f"<div class='stat-grid'>{tiles_html}</div>"


def _index_table(runs: list[RunSummary], output_path: Path) -> str:
    if not runs:
        return "<p class='muted'>No runs found under this runs/ directory.</p>"
    header = (
        "<tr><th scope='col'>Run</th><th scope='col'>Model</th>"
        "<th scope='col'>Epochs (best/trained)</th><th scope='col'>Val RMSE</th>"
        "<th scope='col'>Val MAE</th><th scope='col'>Val skill</th><th scope='col'>Report</th></tr>"
    )
    rows = []
    for run in reversed(runs):  # newest first
        run_label = (
            f"{escape(run.display_date or run.run_name)}"
            f"<br><span class='muted'>{escape(run.run_name)}</span>"
        )
        if run.has_report:
            rel_path = os.path.relpath(Path(run.run_dir) / "report.html", output_path.parent)
            report_cell = f"<a href='{escape(rel_path)}'>View report &rarr;</a>"
        else:
            report_cell = "<span class='muted'>not generated</span>"

        if run.error:
            rows.append(
                f"<tr><td>{run_label}</td>"
                f"<td colspan='5' class='muted'>Couldn't read this run: {escape(run.error)}</td>"
                f"<td>{report_cell}</td></tr>"
            )
            continue

        metrics = run.final_metrics
        rows.append(
            f"<tr><td>{run_label}</td>"
            f"<td>{escape(run.model_type or '–')}</td>"
            f"<td>{run.best_epoch if run.best_epoch is not None else '–'} / {run.epochs_trained}</td>"
            f"<td>{_fmt(metrics.get('rmse'))}</td>"
            f"<td>{_fmt(metrics.get('mae'))}</td>"
            f"<td class='{_skill_class(metrics.get('skill'))}'>{_fmt(metrics.get('skill'))}</td>"
            f"<td>{report_cell}</td></tr>"
        )
    return f"<table class='data-table'><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"


def _index_errors_banner(runs: list[RunSummary]) -> str:
    errored = [r for r in runs if r.error]
    if not errored:
        return ""
    items = "".join(f"<li>{escape(r.run_name)}: {escape(r.error)}</li>" for r in errored)
    return (
        f"<div class='error-banner'><strong>{len(errored)} run(s) couldn't be read</strong> "
        f"and are excluded from the charts below (still listed in the table).<ul>{items}</ul></div>"
    )


_INDEX_PAGE_TEMPLATE = """\
<title>%%TITLE%%</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7;
    --surface-1: #fcfcfb;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --grid-line: #e1e0d9;
    --axis-line: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
    --series-5: #e87ba4; --series-6: #008300; --series-7: #4a3aa7; --series-8: #e34948;
    --good: #0ca30c;
    --warning: #fab219;
    --serious: #ec835a;
    --critical: #d03b3b;
    --tooltip-bg: #0b0b0b;
    --tooltip-fg: #ffffff;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --page: #0d0d0d;
      --surface-1: #1a1a19;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --grid-line: #2c2c2a;
      --axis-line: #383835;
      --border: rgba(255,255,255,0.10);
      --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
      --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
      --critical: #e66767;
      --tooltip-bg: #fcfcfb;
      --tooltip-fg: #0b0b0b;
    }
  }
  * { box-sizing: border-box; }
  body {
    background: var(--page); color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    margin: 0; padding: 24px 16px 64px;
  }
  .wrap { max-width: 1080px; margin: 0 auto; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  h2 { font-size: 1.1rem; margin: 0 0 12px; }
  .meta { color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 24px; }
  .meta code {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 4px; padding: 1px 5px; font-size: 0.82em;
  }
  .card {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px; margin-bottom: 20px;
  }
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
  .stat-tile { border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; background: var(--page); }
  .stat-label { color: var(--text-secondary); font-size: 0.78rem; margin-bottom: 4px; }
  .stat-value { font-size: 1.5rem; font-weight: 600; }
  .stat-good .stat-value { color: var(--good); }
  .stat-warn .stat-value { color: var(--warning); }
  .stat-critical .stat-value { color: var(--critical); }
  .cell-good { color: var(--good); }
  .cell-warn { color: var(--warning); }
  .cell-critical { color: var(--critical); }
  .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }
  .chart-title { font-size: 0.95rem; font-weight: 600; margin: 0 0 2px; }
  .chart-sub { font-size: 0.78rem; color: var(--text-muted); margin: 0 0 10px; }
  .chart-svg-wrap { position: relative; width: 100%; overflow-x: auto; }
  svg.chart { width: 100%; height: auto; display: block; }
  .tick-label { fill: var(--text-muted); font-size: 10px; }
  .gridline { stroke: var(--grid-line); stroke-width: 1; }
  .axis-line { stroke: var(--axis-line); stroke-width: 1; }
  .crosshair { stroke: var(--text-muted); stroke-width: 1; pointer-events: none; }
  .legend { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 8px; font-size: 0.8rem; }
  .legend-item { display: flex; align-items: center; gap: 6px; color: var(--text-secondary); }
  .legend-swatch { width: 14px; height: 2px; border-radius: 1px; }
  .chart-tooltip {
    position: absolute; pointer-events: none; background: var(--tooltip-bg); color: var(--tooltip-fg);
    border-radius: 6px; padding: 6px 10px; font-size: 0.78rem; line-height: 1.4;
    opacity: 0; transform: translate(-50%, -110%); transition: opacity 0.08s ease; white-space: nowrap;
    z-index: 5;
  }
  .chart-tooltip.visible { opacity: 1; }
  .chart-tooltip .tt-row { display: flex; align-items: center; gap: 6px; }
  .chart-tooltip .tt-key { display: inline-block; width: 10px; height: 2px; }
  .chart-tooltip .tt-value { font-weight: 700; }
  .chart-tooltip .tt-name { opacity: 0.8; }
  .table-scroll { overflow-x: auto; }
  table.data-table {
    width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.85rem;
    font-variant-numeric: tabular-nums;
  }
  table.data-table th, table.data-table td {
    text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--grid-line); white-space: nowrap;
  }
  table.data-table thead th { color: var(--text-muted); font-weight: 600; }
  table.data-table a { color: var(--series-1); }
  .muted { color: var(--text-muted); font-size: 0.85rem; }
  .error-banner {
    border: 1px solid var(--critical); border-radius: 8px; padding: 12px 16px;
    color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 20px;
  }
  .error-banner strong { color: var(--critical); }
  .error-banner ul { margin: 8px 0 0; padding-left: 18px; }
  footer { color: var(--text-muted); font-size: 0.78rem; margin-top: 24px; }
</style>
<div class="wrap">
  <header>
    <h1>LOAF run summary</h1>
    <div class="meta">
      %%RUN_COUNT%% run(s) under <code>%%RUNS_DIR%%</code> &middot; generated %%GENERATED_AT%%
    </div>
  </header>

  %%ERROR_BANNER%%

  <section class="card">
    <h2>Overview</h2>
    %%STAT_TILES%%
  </section>

  <section class="card">
    <h2>Progress across runs</h2>
    <div class="charts-grid">
      <div>
        <p class="chart-title">Validation loss</p>
        <p class="chart-sub">Overlaid per-epoch validation loss for the most recent runs - is each retrain actually improving?</p>
        <div class="chart-svg-wrap" id="chart-loss-overlay"></div>
      </div>
      <div>
        <p class="chart-title">Best validation RMSE, by run</p>
        <p class="chart-sub">Lower is better. One bar per run, in training order.</p>
        <div class="chart-svg-wrap" id="chart-rmse-by-run"></div>
      </div>
      <div>
        <p class="chart-title">Best skill vs. persistence, by run</p>
        <p class="chart-sub">Above 0 means that run's model beat a naive "no change" forecast.</p>
        <div class="chart-svg-wrap" id="chart-skill-by-run"></div>
      </div>
    </div>
  </section>

  <section class="card">
    <h2>All runs</h2>
    <div class="table-scroll">
      %%RUN_TABLE%%
    </div>
  </section>

  <footer>Generated by <code>loaf-report-summary</code> (loaf.reporting). Re-run it after new training runs to refresh this file.</footer>
</div>

<script id="index-data" type="application/json">%%INDEX_JSON%%</script>
<script>
%%INDEX_CHART_JS%%
</script>
"""

_INDEX_CHART_JS = r"""
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("index-data").textContent);
  const NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs) {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }

  function fmt(v, digits) {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    digits = digits === undefined ? 2 : digits;
    return Number(v).toFixed(digits);
  }

  function niceTicks(min, max, count) {
    if (min === max) { min -= 1; max += 1; }
    const span = max - min;
    const rawStep = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const norm = rawStep / mag;
    const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
    const start = Math.ceil(min / step) * step;
    const ticks = [];
    for (let v = start; v <= max + step * 1e-6; v += step) ticks.push(Math.round(v / step) * step);
    if (ticks.length === 0) ticks.push(min, max);
    return ticks;
  }

  function scale(domain, range) {
    const span = domain[1] - domain[0];
    const m = span === 0 ? 0 : (range[1] - range[0]) / span;
    return function (v) { return range[0] + (v - domain[0]) * m; };
  }

  function pad(domain, frac) {
    const span = domain[1] - domain[0] || Math.abs(domain[0]) || 1;
    return [domain[0] - span * frac, domain[1] + span * frac];
  }

  const MARGIN = { top: 10, right: 16, bottom: 28, left: 46 };
  const WIDTH = 480;
  const HEIGHT = 220;

  function makeTooltip(wrap) {
    const tip = document.createElement("div");
    tip.className = "chart-tooltip";
    wrap.appendChild(tip);
    return tip;
  }

  function positionTooltip(tip, wrap, px, py) {
    const bounds = wrap.getBoundingClientRect();
    tip.style.left = (px * (bounds.width / WIDTH)) + "px";
    tip.style.top = (py * (bounds.height / HEIGHT)) + "px";
  }

  // ---- Line chart: one or more series sharing an x axis (epoch). ----
  function lineChart(containerId, series, opts) {
    const container = document.getElementById(containerId);
    if (!container) return;
    if (!series.length || !series.some(function (s) { return s.points.length; })) {
      container.innerHTML = "<p class='muted'>No data.</p>";
      return;
    }
    opts = opts || {};
    const plotW = WIDTH - MARGIN.left - MARGIN.right;
    const plotH = HEIGHT - MARGIN.top - MARGIN.bottom;

    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    series.forEach(function (s) {
      s.points.forEach(function (p) {
        if (p[0] < xMin) xMin = p[0];
        if (p[0] > xMax) xMax = p[0];
        if (p[1] !== null && !Number.isNaN(p[1])) {
          if (p[1] < yMin) yMin = p[1];
          if (p[1] > yMax) yMax = p[1];
        }
      });
    });
    const yDomain = pad([yMin, yMax], 0.1);
    const xScale = scale([xMin, xMax], [0, plotW]);
    const yScale = scale(yDomain, [plotH, 0]);

    const svg = el("svg", { class: "chart", viewBox: "0 0 " + WIDTH + " " + HEIGHT });
    const plot = el("g", { transform: "translate(" + MARGIN.left + "," + MARGIN.top + ")" });
    svg.appendChild(plot);

    niceTicks(yDomain[0], yDomain[1], 4).forEach(function (t) {
      const y = yScale(t);
      plot.appendChild(el("line", { class: "gridline", x1: 0, x2: plotW, y1: y, y2: y }));
      const label = el("text", { class: "tick-label", x: -6, y: y + 3, "text-anchor": "end" });
      label.textContent = fmt(t, 2);
      plot.appendChild(label);
    });
    niceTicks(xMin, xMax, 5).forEach(function (t) {
      const x = xScale(t);
      const label = el("text", { class: "tick-label", x: x, y: plotH + 16, "text-anchor": "middle" });
      label.textContent = opts.xTickFormat ? opts.xTickFormat(t) : String(Math.round(t));
      plot.appendChild(label);
    });
    plot.appendChild(el("line", { class: "axis-line", x1: 0, x2: plotW, y1: plotH, y2: plotH }));

    series.forEach(function (s) {
      const pts = s.points.filter(function (p) { return p[1] !== null && !Number.isNaN(p[1]); });
      const d = pts.map(function (p, i) {
        return (i === 0 ? "M" : "L") + xScale(p[0]).toFixed(2) + "," + yScale(p[1]).toFixed(2);
      }).join(" ");
      plot.appendChild(el("path", { d: d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
      if (pts.length) {
        const last = pts[pts.length - 1];
        plot.appendChild(el("circle", { cx: xScale(last[0]), cy: yScale(last[1]), r: 4, fill: s.color, stroke: "var(--surface-1)", "stroke-width": 2 }));
      }
    });

    const hit = el("rect", { x: 0, y: 0, width: plotW, height: plotH, fill: "transparent", style: "cursor:crosshair" });
    const crosshair = el("line", { class: "crosshair", x1: 0, x2: 0, y1: 0, y2: plotH, style: "display:none" });
    plot.appendChild(crosshair);
    plot.appendChild(hit);

    container.innerHTML = "";
    const legend = document.createElement("div");
    legend.className = "legend";
    series.forEach(function (s) {
      const item = document.createElement("span");
      item.className = "legend-item";
      const sw = document.createElement("span");
      sw.className = "legend-swatch";
      sw.style.background = s.color;
      const label = document.createElement("span");
      label.textContent = s.name;
      item.appendChild(sw);
      item.appendChild(label);
      legend.appendChild(item);
    });
    container.appendChild(legend);
    container.appendChild(svg);
    const tip = makeTooltip(container);

    const xValues = [];
    series.forEach(function (s) { s.points.forEach(function (p) { if (xValues.indexOf(p[0]) === -1) xValues.push(p[0]); }); });
    xValues.sort(function (a, b) { return a - b; });

    function showAt(xv) {
      const px = xScale(xv);
      crosshair.setAttribute("x1", px);
      crosshair.setAttribute("x2", px);
      crosshair.style.display = "block";
      let html = "<div class='tt-row'><span class='tt-value'>" + (opts.xTickFormat ? opts.xTickFormat(xv) : xv) + "</span></div>";
      series.forEach(function (s) {
        const p = s.points.find(function (pt) { return pt[0] === xv; });
        if (!p) return;
        html += "<div class='tt-row'><span class='tt-key' style='background:" + s.color + "'></span>" +
          "<span class='tt-value'>" + fmt(p[1], opts.digits) + "</span>" +
          "<span class='tt-name'>" + s.name + "</span></div>";
      });
      tip.innerHTML = html;
      tip.classList.add("visible");
      positionTooltip(tip, container, MARGIN.left + px, MARGIN.top + plotH * 0.15);
    }
    function nearestX(px) {
      let best = xValues[0], bestDist = Infinity;
      xValues.forEach(function (xv) {
        const d = Math.abs(xScale(xv) - px);
        if (d < bestDist) { bestDist = d; best = xv; }
      });
      return best;
    }
    hit.addEventListener("pointermove", function (evt) {
      const rect = svg.getBoundingClientRect();
      const px = ((evt.clientX - rect.left) / rect.width) * WIDTH - MARGIN.left;
      showAt(nearestX(px));
    });
    hit.addEventListener("pointerleave", function () {
      crosshair.style.display = "none";
      tip.classList.remove("visible");
    });
  }

  // ---- Bar chart: one value per named category (run), growing from y=0. ----
  function barChart(containerId, items, opts) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const usable = items.filter(function (d) { return d.value !== null && d.value !== undefined && !Number.isNaN(d.value); });
    if (!usable.length) {
      container.innerHTML = "<p class='muted'>No data.</p>";
      return;
    }
    opts = opts || {};
    const plotW = WIDTH - MARGIN.left - MARGIN.right;
    const plotH = HEIGHT - MARGIN.top - MARGIN.bottom;

    const values = usable.map(function (d) { return d.value; });
    const yDomain = pad([Math.min(0, Math.min.apply(null, values)), Math.max(0, Math.max.apply(null, values))], 0.12);
    const yScale = scale(yDomain, [plotH, 0]);
    const band = plotW / items.length;
    const barW = Math.min(24, band * 0.55);

    const svg = el("svg", { class: "chart", viewBox: "0 0 " + WIDTH + " " + HEIGHT });
    const plot = el("g", { transform: "translate(" + MARGIN.left + "," + MARGIN.top + ")" });
    svg.appendChild(plot);

    niceTicks(yDomain[0], yDomain[1], 4).forEach(function (t) {
      const y = yScale(t);
      plot.appendChild(el("line", { class: "gridline", x1: 0, x2: plotW, y1: y, y2: y }));
      const label = el("text", { class: "tick-label", x: -6, y: y + 3, "text-anchor": "end" });
      label.textContent = opts.yTickFormat ? opts.yTickFormat(t) : fmt(t, 2);
      plot.appendChild(label);
    });
    const zeroY = yScale(0);
    plot.appendChild(el("line", { class: "axis-line", x1: 0, x2: plotW, y1: zeroY, y2: zeroY }));

    const bars = [];
    items.forEach(function (d, i) {
      const cx = band * i + band / 2;
      const xl = el("text", { class: "tick-label", x: cx, y: plotH + 16, "text-anchor": "middle" });
      xl.textContent = d.xTick !== undefined ? d.xTick : d.label;
      plot.appendChild(xl);
      if (d.value === null || d.value === undefined || Number.isNaN(d.value)) return;
      const y1 = yScale(d.value);
      const top = Math.min(zeroY, y1), h = Math.max(Math.abs(zeroY - y1), 1);
      const rect = el("rect", { x: cx - barW / 2, y: top, width: barW, height: h, fill: opts.color || "var(--series-1)", rx: 4 });
      plot.appendChild(rect);
      bars.push({ node: rect, label: d.label, value: d.value, x: cx });
    });

    container.innerHTML = "";
    container.appendChild(svg);
    const tip = makeTooltip(container);
    bars.forEach(function (b) {
      b.node.addEventListener("pointermove", function (evt) {
        const rect = svg.getBoundingClientRect();
        const px = ((evt.clientX - rect.left) / rect.width) * WIDTH - MARGIN.left;
        tip.innerHTML = "<div class='tt-row'><span class='tt-value'>" + fmt(b.value, opts.digits) + "</span></div>" +
          "<div class='tt-row'><span class='tt-name'>" + b.label + "</span></div>";
        tip.classList.add("visible");
        positionTooltip(tip, container, MARGIN.left + px, MARGIN.top + 8);
        b.node.setAttribute("opacity", 0.8);
      });
      b.node.addEventListener("pointerleave", function () {
        tip.classList.remove("visible");
        b.node.setAttribute("opacity", 1);
      });
    });
  }

  // ---- Wire up the index's charts from the embedded data. ----
  const CATEGORICAL = [
    "var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)",
    "var(--series-5)", "var(--series-6)", "var(--series-7)", "var(--series-8)",
  ];
  const runs = DATA.runs.filter(function (r) { return !r.error; });
  const overlayRuns = runs.slice(-8); // cap at 8: the validated CVD-safe categorical order

  lineChart(
    "chart-loss-overlay",
    overlayRuns
      .map(function (r, i) {
        return {
          name: r.label,
          color: CATEGORICAL[i % CATEGORICAL.length],
          points: r.training_curve.map(function (row) { return [row.epoch, row.val_loss]; }),
        };
      })
      .filter(function (s) { return s.points.length; }),
    { xTickFormat: function (v) { return "e" + Math.round(v); }, digits: 3 }
  );

  barChart(
    "chart-rmse-by-run",
    runs.map(function (r) { return { label: r.name, xTick: r.label, value: r.rmse }; }),
    { digits: 3 }
  );
  barChart(
    "chart-skill-by-run",
    runs.map(function (r) { return { label: r.name, xTick: r.label, value: r.skill }; }),
    { digits: 3 }
  );
})();
"""


def render_index_html(data: IndexData, output_path: str | Path) -> Path:
    """Render a cross-run master report (`data`) to a standalone HTML file."""
    output_path = Path(output_path)

    payload = _json_safe(
        {
            "runs": [
                {
                    "name": r.run_name,
                    "label": r.display_date or r.run_name,
                    "training_curve": r.training_curve,
                    "rmse": r.final_metrics.get("rmse"),
                    "mae": r.final_metrics.get("mae"),
                    "skill": r.final_metrics.get("skill"),
                    "error": r.error,
                }
                for r in data.runs
            ]
        }
    )

    html = _INDEX_PAGE_TEMPLATE
    html = html.replace("%%TITLE%%", escape("LOAF run summary"))
    html = html.replace("%%RUN_COUNT%%", str(len(data.runs)))
    html = html.replace("%%RUNS_DIR%%", escape(data.runs_dir))
    html = html.replace("%%GENERATED_AT%%", escape(data.generated_at))
    html = html.replace("%%ERROR_BANNER%%", _index_errors_banner(data.runs))
    html = html.replace("%%STAT_TILES%%", _index_stat_tiles(data.runs))
    html = html.replace("%%RUN_TABLE%%", _index_table(data.runs, output_path))
    index_json = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    html = html.replace("%%INDEX_JSON%%", index_json)
    html = html.replace("%%INDEX_CHART_JS%%", _INDEX_CHART_JS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path
