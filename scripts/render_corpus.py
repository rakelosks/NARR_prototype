#!/usr/bin/env python3
"""
Build a self-contained HTML viewer for narrative corpus JSON files.

Reads every *.json under corpus/, groups runs by query, and writes
corpus/corpus_viewer.html with inline CSS, Vega / Vega-Lite / Vega-Embed
from jsDelivr, and collapsible evidence sections.

Field alignment with backend models:
- GeneratedNarrative / NarrativePackage: headline, lede, story_blocks,
  data_note, followup_question (API uses followup_question; some exports
  may use follow_up_question).
- StoryBlock: type (narrative | timeline | callout), heading, body,
  viz_index, milestones, highlight_value, highlight_label.
- EvidenceBundle (when embedded as response["evidence_bundle"]): metrics,
  matched_columns, narrative_context.key_findings, etc. Current /ask
  responses often omit the full bundle; dataset summary still exposes
  matched_columns and shape fields for cross-reference.
"""

from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"
OUTPUT_HTML = CORPUS_DIR / "corpus_viewer.html"


def run_sort_key(run_id: str) -> tuple[int, str]:
    """Sort run_01a, run_01b, run_10b, … consistently."""
    m = re.match(r"^run_(\d+)([a-z]?)$", (run_id or "").strip(), re.I)
    if m:
        return (int(m.group(1)), m.group(2).lower())
    return (10**9, run_id or "")


def slug_id(run_id: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", run_id or "run").strip("-").lower()
    return s or "run"


def load_records(corpus_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not corpus_dir.is_dir():
        return records
    for path in sorted(corpus_dir.glob("*.json")):
        if path.name == OUTPUT_HTML.name:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            data["_source_file"] = path.name
            records.append(data)
    records.sort(key=lambda r: run_sort_key(str(r.get("run_id", ""))))
    return records


def group_by_query(records: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    query_order: list[str] = []
    for rec in records:
        q = str(rec.get("query") or "(no query)").strip() or "(no query)"
        if q not in buckets:
            query_order.append(q)
        buckets[q].append(rec)
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    for q in query_order:
        runs = buckets[q]
        runs.sort(key=lambda r: run_sort_key(str(r.get("run_id", ""))))
        groups.append((q, runs))
    groups.sort(
        key=lambda item: (
            min(run_sort_key(str(r.get("run_id", ""))) for r in item[1]),
            item[0],
        )
    )
    return groups


def esc(s: Any) -> str:
    if s is None:
        return ""
    return html.escape(str(s), quote=True)


def json_for_script(obj: Any) -> str:
    """Serialize for embedding inside <script>; avoid closing the tag."""
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return raw.replace("</", "<\\/")


def follow_up_text(resp: dict[str, Any]) -> str:
    for key in ("followup_question", "follow_up_question"):
        v = resp.get(key)
        if v is not None and str(v).strip():
            return str(v)
    return ""


def render_metrics_block(metrics: Any) -> str:
    if not metrics:
        return '<p class="muted">No <code>metrics</code> object in this saved response. '
    if not isinstance(metrics, dict):
        return f"<pre>{esc(json.dumps(metrics, indent=2, ensure_ascii=False))}</pre>"
    lines = ["<dl class=\"metrics-dl\">"]
    for k, v in sorted(metrics.items(), key=lambda x: str(x[0])):
        lines.append(f"<dt>{esc(k)}</dt><dd>")
        if isinstance(v, (dict, list)):
            lines.append(f"<pre>{esc(json.dumps(v, indent=2, ensure_ascii=False))}</pre>")
        else:
            lines.append(f"<span>{esc(v)}</span>")
        lines.append("</dd>")
    lines.append("</dl>")
    return "".join(lines)


def render_matched_columns_table(mc: Any) -> str:
    if not mc or not isinstance(mc, dict):
        return '<p class="muted">No column mapping.</p>'
    rows = "".join(
        f"<tr><td>{esc(role)}</td><td><code>{esc(col)}</code></td></tr>"
        for role, col in sorted(mc.items(), key=lambda x: str(x[0]))
    )
    return f"<table class=\"kv\"><thead><tr><th>Role</th><th>Column</th></tr></thead><tbody>{rows}</tbody></table>"


def render_narrative_context(nc: Any) -> str:
    if not nc or not isinstance(nc, dict):
        return ""
    parts: list[str] = []
    if nc.get("template_name"):
        parts.append(f"<p><strong>Template name:</strong> {esc(nc['template_name'])}</p>")
    if nc.get("focus"):
        parts.append(f"<p><strong>Focus:</strong> {esc(nc['focus'])}</p>")
    kf = nc.get("key_findings") or []
    if isinstance(kf, list) and kf:
        items = "".join(f"<li>{esc(x)}</li>" for x in kf)
        parts.append(f"<p><strong>Key findings</strong></p><ul>{items}</ul>")
    sq = nc.get("suggested_questions") or []
    if isinstance(sq, list) and sq:
        items = "".join(f"<li>{esc(x)}</li>" for x in sq)
        parts.append(f"<p><strong>Suggested questions</strong></p><ul>{items}</ul>")
    return "".join(parts) if parts else '<p class="muted">Empty narrative_context.</p>'


def render_story_block(block: dict[str, Any], idx: int) -> str:
    btype = (block.get("type") or "narrative").lower()
    inner: list[str] = [f'<div class="story-block story-block--{esc(btype)}" data-block-index="{idx}">']

    if btype == "callout":
        inner.append('<div class="callout">')
        hv, hl = block.get("highlight_value"), block.get("highlight_label")
        if hv or hl:
            inner.append('<div class="callout-highlight">')
            if hv:
                inner.append(f'<span class="callout-value">{esc(hv)}</span>')
            if hl:
                inner.append(f'<span class="callout-label">{esc(hl)}</span>')
            inner.append("</div>")
        body = block.get("body") or ""
        if body:
            inner.append(f'<p class="callout-body">{esc(body)}</p>')
        inner.append("</div>")

    elif btype == "timeline":
        if block.get("heading"):
            inner.append(f"<h4>{esc(block['heading'])}</h4>")
        milestones = block.get("milestones") or []
        inner.append('<ul class="timeline">')
        if isinstance(milestones, list):
            for m in milestones:
                if not isinstance(m, dict):
                    continue
                label = m.get("label") or ""
                desc = m.get("description") or ""
                inner.append("<li>")
                if label:
                    inner.append(f"<strong>{esc(label)}</strong>")
                if desc:
                    inner.append(f"<p>{esc(desc)}</p>")
                inner.append("</li>")
        inner.append("</ul>")

    else:
        if block.get("heading"):
            inner.append(f"<h4>{esc(block['heading'])}</h4>")
        body = block.get("body") or ""
        if body:
            inner.append(f"<p>{esc(body)}</p>")
        vi = block.get("viz_index")
        if vi is not None:
            inner.append(f'<p class="viz-ref muted">Refers to chart index <code>{esc(vi)}</code></p>')

    inner.append("</div>")
    return "".join(inner)


def render_source_links(resp: dict[str, Any], dataset: dict[str, Any], bundle: Any) -> str:
    lines: list[str] = []
    src = (resp.get("dataset_source") or dataset.get("source") or "").strip()
    if src:
        lines.append(
            f'<p><strong>Data URL:</strong> <a href="{esc(src)}" target="_blank" rel="noopener noreferrer">'
            f"{esc(src)}</a></p>"
        )
    if bundle and isinstance(bundle, dict):
        bsrc = (bundle.get("source") or "").strip()
        if bsrc and bsrc != src:
            lines.append(
                f'<p><strong>Bundle source:</strong> <a href="{esc(bsrc)}" target="_blank" rel="noopener noreferrer">'
                f"{esc(bsrc)}</a></p>"
            )
    did = dataset.get("dataset_id") or (bundle.get("dataset_id") if isinstance(bundle, dict) else None)
    if did:
        lines.append(f'<p><strong>dataset_id:</strong> <code>{esc(did)}</code></p>')
    return "".join(lines)


def render_evidence_section(resp: dict[str, Any]) -> str:
    bundle = resp.get("evidence_bundle")
    dataset = resp.get("dataset") or {}

    parts: list[str] = [
        "<details class=\"evidence-details\">",
        "<summary>Evidence bundle — metrics, mappings &amp; key findings</summary>",
        '<div class="evidence-body">',
        render_source_links(resp, dataset, bundle),
    ]

    if bundle and isinstance(bundle, dict):
        parts.append("<h4>Full evidence bundle (embedded)</h4>")
        mc = bundle.get("matched_columns")
        parts.append("<h5>matched_columns</h5>")
        parts.append(render_matched_columns_table(mc))
        parts.append("<h5>metrics</h5>")
        parts.append(render_metrics_block(bundle.get("metrics")))
        nc = bundle.get("narrative_context")
        if nc:
            parts.append("<h5>narrative_context</h5>")
            parts.append(render_narrative_context(nc))
        rc = bundle.get("row_count")
        cc = bundle.get("column_count")
        if rc is not None or cc is not None:
            parts.append(
                f"<p class=\"muted\">Bundle shape: {esc(rc)} rows × {esc(cc)} columns</p>"
            )
        tmpl = bundle.get("template_type")
        if tmpl:
            parts.append(f"<p><strong>Template:</strong> {esc(tmpl)}</p>")
    else:
        parts.append(
            "<p class=\"muted\">This export has no top-level <code>evidence_bundle</code> "
            "(the current <code>/narratives/ask</code> payload is a narrative package). "
            "Showing <code>dataset</code> summary fields that overlap with the bundle.</p>"
        )
        parts.append("<h4>Dataset summary (matched columns &amp; shape)</h4>")
        parts.append("<h5>matched_columns</h5>")
        parts.append(render_matched_columns_table(dataset.get("matched_columns")))
        parts.append(
            "<p><strong>Rows × columns:</strong> "
            f"{esc(dataset.get('row_count'))} × {esc(dataset.get('column_count'))}</p>"
        )
        if dataset.get("template_type"):
            parts.append(f"<p><strong>Template:</strong> {esc(dataset['template_type'])}</p>")
        ct = dataset.get("column_types")
        if ct and isinstance(ct, dict):
            parts.append(
                "<h5>column_types</h5>"
                f"<pre>{esc(json.dumps(ct, indent=2, ensure_ascii=False))}</pre>"
            )

    parts.append("</div></details>")
    return "".join(parts)


def render_run_card(rec: dict[str, Any], embed_specs: list[dict[str, Any]]) -> str:
    run_id = str(rec.get("run_id") or "unknown")
    rid = slug_id(run_id)
    query = str(rec.get("query") or "")
    src = str(rec.get("_source_file") or "")
    status = rec.get("status_code")
    elapsed = rec.get("response_time_seconds")

    header_bits = [
        f'<header class="card-head">',
        f'<h3 id="run-{esc(rid)}">{esc(run_id)}</h3>',
        f'<p class="query-line">{esc(query)}</p>',
    ]
    if src:
        header_bits.append(f'<p class="meta muted">Source: <code>{esc(src)}</code></p>')
    if status is not None or elapsed is not None:
        header_bits.append(
            f'<p class="meta muted">HTTP {esc(status)} · {esc(elapsed)}s</p>'
        )
    header_bits.append("</header>")

    err = rec.get("error")
    resp = rec.get("response")
    if err or not isinstance(resp, dict):
        body = f'<div class="error-box"><pre>{esc(err or "No response body")}</pre></div>'
        return f'<article class="card" data-run-id="{esc(rid)}">{"".join(header_bits)}{body}</article>'

    blocks_html = []
    for i, block in enumerate(resp.get("story_blocks") or []):
        if isinstance(block, dict):
            blocks_html.append(render_story_block(block, i))

    viz_html: list[str] = []
    visualizations = resp.get("visualizations") or []
    if isinstance(visualizations, list):
        for vi, viz in enumerate(visualizations):
            if not isinstance(viz, dict):
                continue
            spec = viz.get("vega_lite_spec")
            chart_type = viz.get("chart_type") or "chart"
            title = viz.get("title") or f"Visualization {vi}"
            primary = viz.get("is_primary")
            badge = " (primary)" if primary else ""
            div_id = f"viz-{rid}-{vi}"
            if spec is None:
                viz_html.append(
                    f'<figure class="viz-wrap"><figcaption>{esc(title)} — no spec</figcaption></figure>'
                )
                continue
            embed_specs.append({"id": div_id, "spec": spec})
            viz_html.append(
                f'<figure class="viz-wrap">'
                f'<figcaption>{esc(title)} <span class="muted">· {esc(chart_type)}{badge}</span></figcaption>'
                f'<div id="{esc(div_id)}" class="vega-embed-host"></div>'
                f"</figure>"
            )

    fu = follow_up_text(resp)

    inner = [
        "".join(header_bits),
        '<div class="card-body">',
        f'<h2 class="headline">{esc(resp.get("headline") or "")}</h2>',
        f'<p class="lede">{esc(resp.get("lede") or "")}</p>',
        '<section class="story"><h3 class="sr-only">Story</h3>',
        "".join(blocks_html),
        "</section>",
        '<section class="footnotes">',
        f'<h3>Data note</h3><p>{esc(resp.get("data_note") or "")}</p>',
        f'<h3>Follow-up question</h3><p>{esc(fu)}</p>',
        "</section>",
        '<section class="charts"><h3>Charts</h3>',
        "".join(viz_html) if viz_html else "<p class=\"muted\">No visualizations.</p>",
        "</section>",
        render_evidence_section(resp),
        "</div>",
    ]
    return f'<article class="card" data-run-id="{esc(rid)}">{"".join(inner)}</article>'


def build_html(groups: list[tuple[str, list[dict[str, Any]]]], embed_specs: list[dict[str, Any]]) -> str:
    toc_parts: list[str] = ['<nav id="toc"><h2>Table of contents</h2><ul class="toc-queries">']
    for query, runs in groups:
        toc_parts.append(f'<li class="toc-query"><span class="toc-query-text">{esc(query)}</span><ul>')
        for rec in runs:
            run_id = str(rec.get("run_id") or "run")
            rid = slug_id(run_id)
            toc_parts.append(
                f'<li><a href="#run-{esc(rid)}">{esc(run_id)}</a></li>'
            )
        toc_parts.append("</ul></li>")
    toc_parts.append("</ul></nav>")

    main_parts: list[str] = ['<main id="main">']
    for query, runs in groups:
        main_parts.append('<section class="query-group">')
        main_parts.append(f'<h2 class="query-heading">{esc(query)}</h2>')
        for rec in runs:
            main_parts.append(render_run_card(rec, embed_specs))
        main_parts.append("</section>")
    main_parts.append("</main>")

    embed_json = json_for_script(embed_specs)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NARR corpus viewer</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --card: #fff;
      --text: #1a1d24;
      --muted: #5c6570;
      --border: #dde2e8;
      --accent: #2563eb;
      --callout-bg: #eff6ff;
      --callout-border: #3b82f6;
      --timeline: #94a3b8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
    }}
    .wrap {{ max-width: 920px; margin: 0 auto; padding: 1.5rem 1.25rem 4rem; }}
    h1 {{ font-size: 1.75rem; margin: 0 0 1rem; }}
    #toc {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem 1.25rem;
      margin-bottom: 2rem;
    }}
    #toc h2 {{ margin-top: 0; font-size: 1.1rem; }}
    .toc-queries {{ list-style: none; padding-left: 0; margin: 0; }}
    .toc-query {{ margin-bottom: 0.75rem; }}
    .toc-query-text {{ font-weight: 600; display: block; margin-bottom: 0.25rem; }}
    .toc-query ul {{ margin: 0; padding-left: 1.25rem; }}
    .query-group {{ margin-bottom: 3rem; }}
    .query-heading {{
      font-size: 1.2rem;
      border-bottom: 2px solid var(--border);
      padding-bottom: 0.35rem;
      margin-bottom: 1.25rem;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      margin-bottom: 1.5rem;
      overflow: hidden;
      scroll-margin-top: 1rem;
    }}
    .card-head {{
      padding: 1rem 1.25rem;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, #fafbfc 0%, var(--card) 100%);
    }}
    .card-head h3 {{ margin: 0; font-size: 1.05rem; color: var(--accent); }}
    .query-line {{ margin: 0.35rem 0 0; font-size: 0.95rem; }}
    .meta {{ margin: 0.25rem 0 0; font-size: 0.8rem; }}
    .card-body {{ padding: 1.25rem; }}
    .headline {{ font-size: 1.45rem; margin: 0 0 0.5rem; line-height: 1.25; }}
    .lede {{ font-size: 1.05rem; color: var(--muted); margin: 0 0 1.25rem; }}
    .story .story-block {{ margin-bottom: 1.15rem; }}
    .story-block--callout .callout {{
      border-left: 4px solid var(--callout-border);
      background: var(--callout-bg);
      padding: 1rem 1.1rem;
      border-radius: 0 8px 8px 0;
    }}
    .callout-highlight {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; align-items: baseline; margin-bottom: 0.5rem; }}
    .callout-value {{ font-size: 1.75rem; font-weight: 700; color: #1d4ed8; }}
    .callout-label {{ font-size: 0.95rem; color: var(--muted); }}
    .callout-body {{ margin: 0; }}
    .story-block--narrative h4 {{ margin: 0 0 0.35rem; font-size: 1.05rem; }}
    .story-block--timeline ul.timeline {{
      list-style: none;
      padding-left: 0;
      margin: 0;
      border-left: 2px solid var(--timeline);
      padding-left: 1rem;
    }}
    .story-block--timeline li {{ margin-bottom: 0.75rem; position: relative; }}
    .story-block--timeline li::before {{
      content: "";
      position: absolute;
      left: -1.2rem;
      top: 0.35rem;
      width: 8px;
      height: 8px;
      background: var(--accent);
      border-radius: 50%;
    }}
    .footnotes h3 {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); margin: 1.25rem 0 0.35rem; }}
    .footnotes p {{ margin: 0 0 0.5rem; }}
    .charts h3 {{ margin-top: 1.5rem; }}
    .viz-wrap {{ margin: 1.25rem 0; }}
    .viz-wrap figcaption {{ font-size: 0.9rem; margin-bottom: 0.5rem; }}
    .vega-embed-host {{ width: 100%; min-height: 120px; }}
    .muted {{ color: var(--muted); font-size: 0.92rem; }}
    .error-box {{
      margin: 1rem;
      padding: 1rem;
      background: #fef2f2;
      border: 1px solid #fecaca;
      border-radius: 8px;
    }}
    .evidence-details {{
      margin-top: 1.5rem;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.5rem 1rem;
      background: #fafbfc;
    }}
    .evidence-details summary {{
      cursor: pointer;
      font-weight: 600;
      padding: 0.35rem 0;
    }}
    .evidence-body {{ padding: 0.5rem 0 1rem; }}
    .evidence-body h4, .evidence-body h5 {{ margin: 0.75rem 0 0.35rem; font-size: 0.95rem; }}
    .kv {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    .kv th, .kv td {{ border: 1px solid var(--border); padding: 0.35rem 0.5rem; text-align: left; }}
    .kv thead {{ background: #eef1f5; }}
    .metrics-dl dt {{ font-weight: 600; margin-top: 0.5rem; }}
    .metrics-dl dd {{ margin: 0.15rem 0 0 1rem; }}
    .metrics-dl pre {{ margin: 0; font-size: 0.78rem; overflow: auto; max-height: 320px; }}
    pre {{ background: #0f172a0a; padding: 0.6rem 0.75rem; border-radius: 6px; overflow: auto; font-size: 0.82rem; }}
    .sr-only {{
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0,0,0,0);
      border: 0;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>NARR corpus viewer</h1>
    {"".join(toc_parts)}
    {"".join(main_parts)}
  </div>
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <script>
window.__CORPUS_EMBED_SPECS__ = {embed_json};
document.addEventListener("DOMContentLoaded", function () {{
  var list = window.__CORPUS_EMBED_SPECS__ || [];
  var opts = {{ actions: true, renderer: "svg" }};
  list.forEach(function (entry) {{
    var el = document.getElementById(entry.id);
    if (!el || !entry.spec) return;
    vegaEmbed(el, entry.spec, opts).catch(function (e) {{
      el.innerHTML = "<pre class=\\"muted\\">Chart error: " + String(e) + "</pre>";
    }});
  }});
}});
  </script>
</body>
</html>
"""


def main() -> None:
    records = load_records(CORPUS_DIR)
    groups = group_by_query(records)
    embed_specs: list[dict[str, Any]] = []
    html_doc = build_html(groups, embed_specs)
    OUTPUT_HTML.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML} ({len(records)} files, {len(embed_specs)} Vega-Lite embeds).")


if __name__ == "__main__":
    main()
