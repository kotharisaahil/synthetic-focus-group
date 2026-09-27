"""Command line interface: validate a study, preview its sample, and run it."""

from __future__ import annotations

import json
import os
import random
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import CategoricalDim, Study, load_study
from .llm import LLM, LLMError, make_provider
from .metrics import compute_metrics
from .report import spec_text, write_run
from .sampling import format_attr, format_value, sample_population
from .session import run_study

app = typer.Typer(add_completion=False, help="Run a synthetic focus group from a study file.")
console = Console()


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]


def _find_dotenv(study_file: Path) -> Optional[Path]:
    """Look for .env in the current directory, then in the study file's folder and its parents."""
    candidates = [Path.cwd() / ".env"]
    folder = study_file.resolve().parent
    candidates += [d / ".env" for d in (folder, *folder.parents)]
    return next((c for c in candidates if c.is_file()), None)


def _load_dotenv(path: Optional[Path]) -> None:
    """Minimal .env loader so the tool has no extra dependency for it."""
    if path is None:
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _load(path: Path) -> Study:
    try:
        return load_study(path)
    except ValidationError as e:
        console.print(f"[red]Study file has problems:[/red] {path}")
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            console.print(f"  • {loc}: {err['msg']}")
        raise typer.Exit(1)
    except FileNotFoundError:
        console.print(f"[red]No such file:[/red] {path}")
        raise typer.Exit(1)


@app.command()
def validate(study_file: Path = typer.Argument(..., help="Path to a study YAML")) -> None:
    """Check a study file and summarize what it will do."""
    s = _load(study_file)
    console.print(f"[green]Valid.[/green] [bold]{s.title}[/bold]")
    groups = f"{s.groups} group" + ("" if s.groups == 1 else "s")
    console.print(f"  {groups} × {s.group_size} participants, {len(s.guide)} topic{'' if len(s.guide) == 1 else 's'}, {len(s.ratings)} private ratings")
    hyps = sum(len(r.expect) for r in s.ratings)
    console.print(f"  {len(s.population)} population attributes ({sum(d.anchor for d in s.population)} fixed attitudes), {hyps} fidelity hypotheses")
    if s.benchmark:
        console.print(f"  Benchmark: {s.benchmark}")


@app.command()
def personas(
    study_file: Path = typer.Argument(..., help="Path to a study YAML"),
    n: Optional[int] = typer.Option(None, help="How many to draw (default: groups × group size)"),
) -> None:
    """Preview the sampled population. No model calls, no cost."""
    s = _load(study_file)
    count = n or s.groups * s.group_size
    draws = sample_population(s, count, random.Random(s.seed))

    table = Table(title=f"{count} sampled participants (seed {s.seed})")
    table.add_column("#", justify="right")
    for d in s.population:
        table.add_column(d.display)
    for i, attrs in enumerate(draws, 1):
        table.add_row(str(i), *[format_attr(d, attrs[d.name]) for d in s.population])
    console.print(table)

    summary = Table(title="Sample vs. spec")
    summary.add_column("Attribute")
    summary.add_column("Spec")
    summary.add_column("This sample")
    for d in s.population:
        vals = [a[d.name] for a in draws]
        if isinstance(d, CategoricalDim):
            got = ", ".join(f"{k} {vals.count(k) / len(vals):.0%}" for k in d.options)
        else:
            got = f"median {format_value(d, statistics.median(vals))}, range {format_value(d, min(vals))} to {format_value(d, max(vals))}"
        summary.add_row(d.display, spec_text(d), got)
    console.print(summary)


@app.command()
def run(
    study_file: Path = typer.Argument(..., help="Path to a study YAML"),
    provider: Optional[str] = typer.Option(None, help="anthropic, openai, or mock (overrides the study file)"),
    mock: bool = typer.Option(False, "--mock", help="Shortcut for --provider mock (offline, no API key)"),
    out: Path = typer.Option(Path("runs"), help="Where to write run folders"),
    groups: Optional[int] = typer.Option(None, help="Override the number of groups"),
    size: Optional[int] = typer.Option(None, help="Override participants per group"),
    seed: Optional[int] = typer.Option(None, help="Override the sampling seed"),
    quick: bool = typer.Option(
        False, "--quick", help="Cheap preflight: 1 group of 3, first topic only. Exercises every agent for a few cents."
    ),
) -> None:
    """Run the focus group and write a report."""
    _load_dotenv(_find_dotenv(study_file))
    s = _load(study_file)
    updates = {k: v for k, v in {"groups": groups, "group_size": size, "seed": seed}.items() if v is not None}
    if quick:
        first = s.guide[0].model_copy(update={"max_turns": min(s.guide[0].max_turns, 4)})
        updates.update({"groups": 1, "group_size": 3, "guide": [first.model_dump()]})
    if updates:
        s = Study.model_validate({**s.model_dump(), **updates})
    prov = "mock" if mock else (provider or s.models.provider)

    try:
        provider_obj = make_provider(prov)
    except LLMError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # The run folder exists from the start and its call log fills in live, so a run in
    # progress can be watched and a failed run always leaves evidence behind.
    run_dir = out / f"{_slug(s.title)}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(provider_obj, s.models.resolved(prov), log_path=run_dir / "calls.jsonl")

    total = s.groups * s.group_size
    console.print(f"[bold]{s.title}[/bold] · {prov} · {s.groups} × {s.group_size} participants · {len(s.guide)} topic{'' if len(s.guide) == 1 else 's'}")
    console.print(f"Writing to {run_dir}")
    try:
        with console.status("Starting...") as status:
            result = run_study(s, llm, progress=lambda msg: status.update(msg))
    except LLMError as e:
        console.print(f"[red]Run stopped:[/red] {e}")
        if llm.calls:
            failed = run_dir.with_name(run_dir.name + "-failed")
            run_dir.rename(failed)
            console.print(f"The {len(llm.calls)} calls made before the failure are in {failed / 'calls.jsonl'}")
        else:
            run_dir.rmdir()
        raise typer.Exit(1)

    metrics = compute_metrics(result)
    paths = write_run(result, metrics, llm.calls, run_dir)

    console.print(f"[green]Done.[/green] {total} participants, {len(result.turns)} turns, {len(llm.calls)} model calls")
    if prov != "mock":
        for model, u in result.usage.items():
            console.print(f"  {model}: {u['calls']} calls, {u['input']:,} input / {u['output']:,} output tokens")
    warns = [f for f in metrics.flags if f.level == "warn"]
    if warns:
        console.print(f"[yellow]{len(warns)} reliability warning{'' if len(warns) == 1 else 's'}:[/yellow]")
        for f in warns:
            console.print(f"  • {f.message}")
    console.print(f"Report: [link=file://{paths['report_html'].resolve()}]{paths['report_html']}[/link]")


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
