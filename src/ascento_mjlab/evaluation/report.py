"""Aggregation and a dependency-free HTML evaluation report."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any

from .schema import EpisodeResult
from .statistics import summarize_binary, summarize_numeric


BINARY_METRICS = {"recovered", "recovery_success"}


def summarize_results(
  results: list[EpisodeResult],
  *,
  bootstrap_seed: int = 0,
) -> dict[str, dict[str, dict[str, float]]]:
  by_family: dict[str, list[EpisodeResult]] = defaultdict(list)
  for result in results:
    by_family[result.family].append(result)

  output: dict[str, dict[str, dict[str, float]]] = {}
  for family, episodes in by_family.items():
    metrics: dict[str, dict[str, float]] = {}
    metrics["success"] = summarize_binary([episode.success for episode in episodes])
    names = sorted({name for episode in episodes for name in episode.metrics})
    for metric_index, name in enumerate(names):
      values = [episode.metrics[name] for episode in episodes if name in episode.metrics]
      if name in BINARY_METRICS:
        metrics[name] = summarize_binary([bool(value) for value in values])
      else:
        metrics[name] = summarize_numeric(
          values, seed=bootstrap_seed + metric_index
        )
    termination_counts = Counter(episode.termination_reason for episode in episodes)
    metrics["termination_count"] = {
      key: float(value) for key, value in sorted(termination_counts.items())
    }
    output[family] = metrics
  return output


def select_worst_scenarios(results: list[EpisodeResult], limit: int = 5) -> dict[str, list[str]]:
  failures = [result.scenario_id for result in results if not result.success][:limit]
  selectors = {
    "max_tilt": True,
    "net_displacement": True,
    "effort_rms": True,
    "recovery_time_s": True,
  }
  output = {"failures": failures}
  for metric, descending in selectors.items():
    eligible = [
      result
      for result in results
      if metric in result.metrics and result.metrics[metric] == result.metrics[metric]
    ]
    eligible.sort(key=lambda item: item.metrics[metric], reverse=descending)
    output[metric] = [result.scenario_id for result in eligible[:limit]]
  return output


def render_html(
  path: Path,
  *,
  manifest: dict[str, Any],
  summary: dict[str, Any],
  gate_payload: dict[str, Any],
  worst: dict[str, list[str]],
) -> None:
  status = escape(str(gate_payload.get("status", "UNKNOWN")))
  rows: list[str] = []
  for gate in gate_payload.get("gates", []):
    observed = gate.get("observed")
    observed_text = "missing" if observed is None else f"{observed:.6g}"
    rows.append(
      "<tr>"
      f"<td>{escape(gate['gate_id'])}</td>"
      f"<td>{escape(gate['family'])}</td>"
      f"<td>{escape(gate['metric'])}.{escape(gate['statistic'])}</td>"
      f"<td>{escape(observed_text)}</td>"
      f"<td>{escape(gate['op'])} {gate['threshold']:.6g}</td>"
      f"<td>{'PASS' if gate['passed'] else 'FAIL'}</td>"
      "</tr>"
    )
  worst_html = "".join(
    f"<h3>{escape(name)}</h3><ul>"
    + "".join(f"<li><code>{escape(item)}</code></li>" for item in values)
    + "</ul>"
    for name, values in worst.items()
  )
  payload = escape(json.dumps(summary, indent=2, sort_keys=True))
  html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Ascento evaluation {status}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 1200px; }}
code, pre {{ font-family: ui-monospace, monospace; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: .4rem; text-align: left; }}
.status {{ font-size: 2rem; font-weight: 700; }}
pre {{ overflow: auto; background: #f4f4f4; padding: 1rem; }}
</style>
</head>
<body>
<h1>Ascento quantitative evaluation</h1>
<div class="status">{status}</div>
<p><strong>Suite:</strong> {escape(str(manifest.get("suite_id")))}</p>
<p><strong>Checkpoint:</strong> <code>{escape(str(manifest.get("checkpoint")))}</code></p>
<h2>Gates</h2>
<table><thead><tr><th>Gate</th><th>Family</th><th>Metric</th><th>Observed</th><th>Required</th><th>Result</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Worst scenarios</h2>
{worst_html}
<h2>Full summary</h2>
<pre>{payload}</pre>
</body></html>
"""
  path.write_text(html, encoding="utf-8")
