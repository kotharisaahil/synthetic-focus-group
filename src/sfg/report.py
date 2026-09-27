"""Write a run to disk: transcript, findings, structured data, a call log, and an HTML report."""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from .config import CategoricalDim, LognormalDim, NormalDim, ScaleDim, Study, UniformDim
from .llm import CallRecord
from .metrics import ItemStats, Metrics
from .sampling import format_attr, format_range, format_value
from .session import SessionResult

MOCK_NOTICE = (
    "Mock run: every response below came from the offline test provider, not a language model. "
    "It demonstrates the pipeline only."
)


def spec_text(dim) -> str:
    """Plain-language description of how a dimension is distributed."""
    if isinstance(dim, UniformDim):
        return f"evenly spread, {format_range(dim, dim.min, dim.max)}"
    if isinstance(dim, NormalDim):
        return f"centered on {format_value(dim, dim.mean)} (SD {dim.sd:g})"
    if isinstance(dim, LognormalDim):
        return f"typical {format_value(dim, dim.median)}, top 10% above {format_value(dim, dim.p90)}"
    if isinstance(dim, CategoricalDim):
        return ", ".join(f"{k} {v:g}%" for k, v in dim.options.items())
    if isinstance(dim, ScaleDim):
        return f"{dim.low} to {dim.high} scale, centered on {dim.mean:g} (SD {dim.sd:g}); {dim.low} = {dim.low_label}, {dim.high} = {dim.high_label}"
    return ""


# ---------------------------------------------------------------------------
# Markdown outputs
# ---------------------------------------------------------------------------


def transcript_md(result: SessionResult) -> str:
    study = result.study
    out = [f"# Transcript: {study.title}", ""]
    if result.provider == "mock":
        out += [f"> {MOCK_NOTICE}", ""]
    for g in sorted({p.group for p in result.personas}):
        out.append(f"## Group {g}")
        out.append("")
        for p in (p for p in result.personas if p.group == g):
            out.append(f"- **{p.name}** ({p.summary(study)})")
        out.append("")
        current = None
        for t in (t for t in result.turns if t.group == g):
            if t.topic != current:
                current = t.topic
                out += ["", f"### {study.guide[t.topic].title}", ""]
            who = f"**{t.speaker}**" if t.kind == "participant" else "*Moderator*"
            out.append(f"{who}: {t.text}  ")
            if t.note:
                out.append(f"  _(guardrail: {t.note})_  ")
        out.append("")
    return "\n".join(out)


def report_md(result: SessionResult, metrics: Metrics) -> str:
    a = result.analysis
    study = result.study
    out = [f"# {study.title}", "", f"**Objective:** {study.objective}", ""]
    if result.provider == "mock":
        out += [f"> {MOCK_NOTICE}", ""]
    out += [f"## {a.headline}", "", a.summary, "", "## Themes", ""]
    for th in a.themes:
        out += [f"### {th.title} ({th.prevalence})", "", th.description, ""]
        for q in th.quotes:
            out.append(f'> "{q.quote}" (**{q.speaker}**)')
            out.append("")
    if a.disagreements:
        out += ["## Where participants split", ""] + [f"- {d}" for d in a.disagreements] + [""]
    if a.open_questions:
        out += ["## Open questions", ""] + [f"- {q}" for q in a.open_questions] + [""]
    out += ["## Private ratings", "", "| Item | Before: mean (SD) | After: mean (SD) | Spread shrank by |", "|---|---|---|---|"]
    for it in metrics.items:
        out.append(
            f"| {it.label} | {it.pre['mean']} ({it.pre['sd']}) | {it.post['mean']} ({it.post['sd']}) | {it.conformity['convergence']:.0%} |"
        )
    out += ["", "## Reliability checks", ""]
    out += [f"- **{f.level.upper()}** [{f.check}] {f.message}" for f in metrics.flags] or ["- No issues flagged."]
    out += ["", "## Method", "", f"- Groups: {study.groups} × {study.group_size} participants, seed {study.seed}",
            f"- Provider: {result.provider}; models: {', '.join(f'{k}={v}' for k, v in result.models.items())}", ""]
    out += ["| Attribute | Distribution | Fixed attitude |", "|---|---|---|"]
    for d in study.population:
        out.append(f"| {d.display} | {spec_text(d)} | {'yes' if d.anchor else ''} |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

CSS = """
:root{--bg:#f6f5f2;--card:#fff;--ink:#1c1917;--muted:#6b6660;--line:#e7e3dc;--accent:#0f766e;--accent-soft:#e6f2f0;
--warn:#b45309;--warn-soft:#fdf3e7;--ok:#15803d;--ok-soft:#eaf6ee;--pre:#94a3b8;--post:#0f766e;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:40px 24px 80px}
header .eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);font-weight:700}
h1{font-size:30px;margin:6px 0 8px;line-height:1.2}h2{font-size:20px;margin:40px 0 14px}h3{font-size:16px;margin:0 0 6px}
.meta{color:var(--muted);font-size:13px}.objective{margin:10px 0 0;max-width:760px}
.mock{background:var(--warn-soft);border:1px solid #f1d3ae;color:var(--warn);padding:12px 16px;border-radius:10px;margin-top:18px;font-weight:600;font-size:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px 24px}
.headline{font-size:21px;font-weight:650;line-height:1.35;margin:0 0 10px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:18px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}.tile .v{font-size:22px;font-weight:700;margin-top:2px}
.tile .s{font-size:12px;color:var(--muted)}.tile.warn .v{color:var(--warn)}.tile.ok .v{color:var(--ok)}
.themes{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.pill{display:inline-block;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;padding:2px 8px;border-radius:99px;background:var(--accent-soft);color:var(--accent);margin-left:6px;vertical-align:middle}
blockquote{margin:10px 0 0;padding:8px 12px;border-left:3px solid var(--accent);background:#fafaf8;border-radius:0 8px 8px 0;font-size:14px}
blockquote cite{display:block;font-style:normal;color:var(--muted);font-size:12px;margin-top:2px}
ul.clean{margin:6px 0 0;padding-left:18px}
.item{margin-bottom:14px}.item .lbl{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--accent);font-weight:700}.item .q{font-weight:650}
.dist{display:grid;grid-template-columns:28px 1fr 26px 1fr 26px;gap:4px 10px;align-items:center;margin:12px 0 8px;font-size:12px}.dist .n{color:#44403c;font-weight:600}
.dist .h{font-weight:700;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.bar{height:14px;background:#f0eee9;border-radius:4px;position:relative;overflow:hidden}.bar i{position:absolute;left:0;top:0;bottom:0;border-radius:4px}
.bar.pre i{background:var(--pre)}.bar.post i{background:var(--post)}
.stats{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:13px;color:var(--muted)}.stats strong{color:var(--ink)}
.chip{display:inline-block;font-size:12px;padding:2px 8px;border-radius:6px;margin:4px 6px 0 0}.chip.ok{background:var(--ok-soft);color:var(--ok)}.chip.warn{background:var(--warn-soft);color:var(--warn)}
.flags li{margin:6px 0}.flags .lvl{font-size:11px;font-weight:800;padding:1px 6px;border-radius:4px;margin-right:6px}
.flags .warn{background:var(--warn-soft);color:var(--warn)}.flags .info{background:#eef2f7;color:#475569}
.people{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.person .id{font-size:12px;color:var(--muted)}.person ul{margin:8px 0;padding-left:16px;font-size:12.5px;color:#44403c}.person p{font-size:13px;margin:6px 0 0;color:#44403c}
details{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 18px;margin-bottom:10px}summary{cursor:pointer;font-weight:650}
.topic{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--accent);font-weight:700;margin:18px 0 6px}
.turn{margin:6px 0;font-size:14px}.turn .who{font-weight:650}.turn.mod{color:var(--muted);font-style:italic}.turn .note{display:block;font-size:11.5px;color:var(--warn);font-style:normal}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600}
footer{margin-top:48px;color:var(--muted);font-size:12px}
"""


def _dist_html(it: ItemStats) -> str:
    rows = ['<div class="h"></div><div class="h">Before discussion</div><div></div><div class="h">After discussion</div><div></div>']
    peak = max([*it.pre["counts"].values(), *it.post["counts"].values(), 1])
    for v in range(it.high, it.low - 1, -1):
        a, b = it.pre["counts"].get(v, 0), it.post["counts"].get(v, 0)
        rows.append(
            f'<div>{v}</div>'
            f'<div class="bar pre"><i style="width:{100 * a / peak:.0f}%"></i></div><div class="n">{a or ""}</div>'
            f'<div class="bar post"><i style="width:{100 * b / peak:.0f}%"></i></div><div class="n">{b or ""}</div>'
        )
    return '<div class="dist">' + "".join(rows) + "</div>"


def _tile(k: str, v: str, s: str, cls: str = "") -> str:
    return f'<div class="tile {cls}"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div><div class="s">{escape(s)}</div></div>'


def report_html(result: SessionResult, metrics: Metrics) -> str:
    study, a = result.study, result.analysis
    e = escape
    n = len(result.personas)
    spread_warns = sum(1 for f in metrics.flags if f.check in ("spread", "benchmark") and f.level == "warn")
    fid = [f for it in metrics.items for f in it.anchor_fidelity]
    fid_ok = sum(1 for f in fid if f["ok"])
    max_conv = max((it.conformity["convergence"] for it in metrics.items), default=0.0)

    tiles = "".join([
        _tile("Participants", str(n), f"{study.groups} {'group' if study.groups == 1 else 'groups'} of {study.group_size}"),
        _tile("Response spread", "OK" if not spread_warns else f"{spread_warns} flag" + ("" if spread_warns == 1 else "s"), "mean regression check", "ok" if not spread_warns else "warn"),
        _tile("Persona fidelity", f"{fid_ok}/{len(fid)}" if fid else "n/a", "attribute hypotheses that held", "ok" if fid and fid_ok == len(fid) else ("warn" if fid else "")),
        _tile("Group pull", f"{max_conv:.0%}", "largest drop in spread after discussion", "warn" if max_conv > 0.4 else "ok"),
        _tile("Quotes removed", str(metrics.quotes_removed), "failed verbatim check", "warn" if metrics.quotes_removed else "ok"),
    ])

    themes = []
    for th in a.themes:
        quotes = "".join(f"<blockquote>“{e(q.quote)}”<cite>{e(q.speaker)}</cite></blockquote>" for q in th.quotes)
        themes.append(f'<div class="card"><h3>{e(th.title)}<span class="pill">{e(th.prevalence)}</span></h3><div>{e(th.description)}</div>{quotes}</div>')

    split = ""
    if a.disagreements or a.open_questions:
        split = '<div class="themes" style="margin-top:14px">'
        if a.disagreements:
            split += '<div class="card"><h3>Where participants split</h3><ul class="clean">' + "".join(f"<li>{e(d)}</li>" for d in a.disagreements) + "</ul></div>"
        if a.open_questions:
            split += '<div class="card"><h3>Open questions</h3><ul class="clean">' + "".join(f"<li>{e(q)}</li>" for q in a.open_questions) + "</ul></div>"
        split += "</div>"

    items_html = []
    for it in metrics.items:
        chips = ""
        for f in it.anchor_fidelity:
            cls = "ok" if f["ok"] else "warn"
            rho = "n/a" if f["rho"] is None else f"{f['rho']:+.2f}"
            chips += f'<span class="chip {cls}">{e(f["label"])}: expected {e(f["direction"])} link, rho {rho}</span>'
        bench = ""
        if it.benchmark:
            b = it.benchmark
            bench = f'<div class="stats" style="margin-top:6px"><span>Human benchmark SD <strong>{b["human_sd"]}</strong></span><span>Variance ratio <strong>{b["variance_ratio"]}</strong></span><span>Distance (TVD) <strong>{b["tvd"]}</strong></span></div>'
        c = it.conformity
        items_html.append(
            f'<div class="card item"><div class="lbl">{e(it.label)}</div><div class="q">{e(it.question)}</div>'
            f'<div class="meta">{it.low} to {it.high} scale</div>{_dist_html(it)}'
            f'<div class="stats"><span>Before: mean <strong>{it.pre["mean"]}</strong>, SD <strong>{it.pre["sd"]}</strong></span>'
            f'<span>After: mean <strong>{it.post["mean"]}</strong>, SD <strong>{it.post["sd"]}</strong></span>'
            f'<span>Changed their answer <strong>{c["changed_share"]:.0%}</strong></span>'
            f'<span>Spread change <strong>{-c["convergence"]:+.0%}</strong></span></div>'
            f"{bench}<div>{chips}</div></div>"
        )

    flags = "".join(f'<li><span class="lvl {f.level}">{f.level.upper()}</span>{e(f.message)}</li>' for f in metrics.flags) or "<li>No issues flagged.</li>"

    people = []
    for p in result.personas:
        attrs = "".join(f"<li>{e(d.display)}: {e(format_attr(d, p.attributes[d.name]))}</li>" for d in study.population)
        people.append(f'<div class="card person"><h3>{e(p.name)}</h3><div class="id">{e(p.id)}</div><ul>{attrs}</ul><p>{e(p.backstory)}</p></div>')

    transcripts = []
    for g in sorted({p.group for p in result.personas}):
        body, current = [], None
        for t in (t for t in result.turns if t.group == g):
            if t.topic != current:
                current = t.topic
                body.append(f'<div class="topic">{e(study.guide[t.topic].title)}</div>')
            note = f'<span class="note">Guardrail: {e(t.note)}</span>' if t.note else ""
            if t.kind == "moderator":
                body.append(f'<div class="turn mod"><span class="who">Moderator:</span> {e(t.text)}{note}</div>')
            else:
                body.append(f'<div class="turn"><span class="who">{e(t.speaker)}:</span> {e(t.text)}</div>')
        transcripts.append(f"<details><summary>Group {g} transcript ({sum(1 for t in result.turns if t.group == g)} turns)</summary>{''.join(body)}</details>")

    spec_rows = "".join(
        f"<tr><td>{e(d.display)}</td><td>{e(spec_text(d))}</td><td>{'yes' if d.anchor else ''}</td></tr>" for d in study.population
    )
    usage_rows = "".join(
        f"<tr><td>{e(model)}</td><td>{u['calls']}</td><td>{u['input']:,}</td><td>{u['output']:,}</td></tr>"
        for model, u in result.usage.items()
    )
    usage_block = (
        f'<h3 style="margin-top:18px">Model usage</h3><table><tr><th>Model</th><th>Calls</th><th>Input tokens</th><th>Output tokens</th></tr>{usage_rows}</table>'
        if result.usage and result.provider != "mock"
        else ""
    )
    guide_rows = "".join(f"<tr><td>{i + 1}. {e(t.title)}</td><td>{e(t.question)}</td></tr>" for i, t in enumerate(study.guide))
    models = ", ".join(f"{k}: {v}" for k, v in result.models.items())
    mock = f'<div class="mock">{e(MOCK_NOTICE)}</div>' if result.provider == "mock" else ""
    now = datetime.now().strftime("%B %d, %Y %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(study.title)} · Synthetic focus group report</title><style>{CSS}</style></head>
<body><div class="wrap">
<header><div class="eyebrow">Synthetic focus group report</div><h1>{e(study.title)}</h1>
<div class="meta">{now} · {e(result.provider)} ({e(models)}) · seed {study.seed}</div>
<p class="objective"><strong>Objective:</strong> {e(study.objective)}</p>{mock}</header>

<h2>At a glance</h2>
<div class="card"><p class="headline">{e(a.headline)}</p><div>{e(a.summary)}</div></div>
<div class="tiles">{tiles}</div>

<h2>Themes</h2><div class="themes">{''.join(themes)}</div>{split}

<h2>Private ratings, before and after discussion</h2>
<p class="meta">Every participant rated privately before hearing anyone else, then again at the end. The gap shows how much the conversation moved people.</p>
{''.join(items_html)}

<h2>Reliability checks</h2><div class="card"><ul class="flags clean" style="list-style:none;padding:0">{flags}</ul></div>

<h2>Participants</h2><div class="people">{''.join(people)}</div>

<h2>Transcripts</h2>{''.join(transcripts)}

<h2>Method</h2>
<div class="card"><h3>What participants reacted to</h3><p>{e(study.stimulus)}</p>
<h3 style="margin-top:18px">Population</h3><table><tr><th>Attribute</th><th>Distribution</th><th>Fixed attitude</th></tr>{spec_rows}</table>
<h3 style="margin-top:18px">Session protocol</h3><p>Participants read the concept before the session and each rated it privately. The moderator then ran the discussion guide below. At the end, each participant saw the whole discussion and their own first answer, and rated again in private.</p>
<h3 style="margin-top:18px">Discussion guide</h3><table><tr><th>Topic</th><th>Guide question</th></tr>{guide_rows}</table>{usage_block}</div>

<footer>Generated by synthetic-focus-group. Synthetic participants are useful for sharpening hypotheses and piloting discussion guides; they are not a substitute for talking to real people.</footer>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def run_data(result: SessionResult, metrics: Metrics) -> dict[str, Any]:
    return {
        "study": result.study.model_dump(),
        "provider": result.provider,
        "models": result.models,
        "personas": [p.to_dict() for p in result.personas],
        "ratings": [r.to_dict() for r in result.ratings],
        "turns": [t.to_dict() for t in result.turns],
        "analysis": result.analysis.model_dump(),
        "metrics": metrics.to_dict(),
        "guardrail_events": result.guardrail_events,
        "usage": result.usage,
    }


def write_run(result: SessionResult, metrics: Metrics, calls: list[CallRecord], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report_html": out_dir / "report.html",
        "report_md": out_dir / "report.md",
        "transcript": out_dir / "transcript.md",
        "data": out_dir / "data.json",
        "calls": out_dir / "calls.jsonl",
    }
    paths["report_html"].write_text(report_html(result, metrics), encoding="utf-8")
    paths["report_md"].write_text(report_md(result, metrics), encoding="utf-8")
    paths["transcript"].write_text(transcript_md(result), encoding="utf-8")
    paths["data"].write_text(json.dumps(run_data(result, metrics), indent=2, default=str), encoding="utf-8")
    with paths["calls"].open("w", encoding="utf-8") as f:
        for c in calls:
            f.write(json.dumps(c.to_dict(), default=str) + "\n")
    return paths
