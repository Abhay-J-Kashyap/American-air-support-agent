"""Command line entry points."""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from aa_agent.config import Config, load_config
from aa_agent.ingest import run_ingest
from aa_agent.llm.client import LLMClient

app = typer.Typer(add_completion=False, help="AmericanAir support agent pipeline")
console = Console()

load_dotenv()

RATE_HEADERS = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
)


def _error_detail(r: httpx.Response, limit: int = 160) -> str:
    """Pull the actual message + error type out of the response body.

    A blind `r.text[:60]` slice cuts most OpenAI-compatible error bodies off
    right before the `"type"` field, which is usually the one piece of text
    that tells you WHY (rate_limit_exceeded vs invalid_api_key vs
    model_not_found are very different fixes) rather than just THAT
    something failed.
    """
    try:
        body = r.json()
    except ValueError:
        return f"{r.status_code} {r.text[:limit]}"
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        msg, typ = err.get("message") or str(err), err.get("type")
    elif isinstance(body, dict):
        msg, typ = body.get("message") or str(body), body.get("type")
    else:
        msg, typ = str(body), None
    text = f"{r.status_code} {msg}" + (f" [{typ}]" if typ else "")
    return text[:limit]


def _cfg(path: Path | None) -> Config:
    return load_config(path)


@app.command("check-providers")
def check_providers(
    config: Path = typer.Option(None, "--config", "-c"),
    role: str = typer.Option(None, "--role", help="Check one role only"),
) -> None:
    """Ping every configured model and print the limits the API actually reports.

    Ninety seconds that saves an overnight run. Published free-tier limits go
    stale fast and differ per account -- the response headers are the truth.
    Run this before every long sweep.
    """
    cfg = _cfg(config)
    roles = [role] if role else list(cfg.roles)

    table = Table(show_header=True, header_style="bold")
    for col in ("role", "provider/model", "status", "ms", "reported limits"):
        table.add_column(col)

    failures = 0
    with httpx.Client(timeout=30.0) as http:
        for rname in roles:
            rcfg = cfg.role(rname)
            for i, ref in enumerate(rcfg.chain):
                provider = cfg.provider(ref.provider)
                label = f"{rname}{'' if i == 0 else ' (fb)'}"
                if not provider.enabled:
                    table.add_row(label, str(ref), "[dim]disabled[/dim]", "-", "-")
                    continue
                if provider.api_key is None:
                    table.add_row(label, str(ref), "[red]no key[/red]", "-", provider.api_key_env)
                    if i == 0:
                        failures += 1
                    continue

                started = time.monotonic()
                try:
                    r = http.post(
                        provider.base_url.rstrip("/") + "/chat/completions",
                        headers={"Authorization": f"Bearer {provider.api_key}"},
                        json={
                            "model": ref.model,
                            "messages": [{"role": "user", "content": "ping"}],
                            "max_tokens": 5,
                            "temperature": 0,
                        },
                    )
                except httpx.HTTPError as exc:
                    table.add_row(label, str(ref), "[red]error[/red]", "-", str(exc)[:60])
                    if i == 0:
                        failures += 1
                    continue

                ms = int((time.monotonic() - started) * 1000)
                limits = (
                    " ".join(
                        f"{h.rsplit('-', 1)[-1][:3]}={r.headers[h]}"
                        for h in RATE_HEADERS
                        if h in r.headers
                    )
                    or "[dim]none reported[/dim]"
                )

                if r.status_code == 200:
                    table.add_row(label, str(ref), "[green]ok[/green]", str(ms), limits)
                else:
                    table.add_row(label, str(ref), "[red]fail[/red]", str(ms), _error_detail(r))
                    if i == 0:
                        failures += 1

    console.print(table)
    if failures:
        console.print(f"[red]{failures} primary model(s) unreachable.[/red]")
        raise typer.Exit(code=1)
    console.print("[green]All primary models reachable.[/green]")


@app.command("ingest")
def ingest(
    config: Path = typer.Option(None, "--config", "-c"),
    keep_raw: bool = typer.Option(
        False, "--keep-raw", help="Don't delete the raw CSV after filtering."
    ),
) -> None:
    """Filter the raw multi-brand CSV down to config.project.brand's slice.

    Two passes over the raw file (see aa_agent.ingest module docstring for
    why it can't be one), then thread reconstruction, multi-part reply
    rejoining, PII scrubbing, and the time-based train/eval split -- all
    written to one parquet file.
    """
    cfg = _cfg(config)
    if not cfg.data.raw_csv.exists():
        console.print(f"[red]{cfg.data.raw_csv} not found.[/red]")
        console.print(
            "Download twcs.csv from "
            "kaggle.com/datasets/thoughtvector/customer-support-on-twitter "
            f"and place it at {cfg.data.raw_csv}."
        )
        raise typer.Exit(code=1)

    with console.status(f"Filtering to {cfg.project.brand}..."):
        stats = run_ingest(
            raw_csv=cfg.data.raw_csv,
            out_parquet=cfg.data.brand_parquet,
            brand_author_id=cfg.project.brand,
            split_quantile=cfg.data.split_quantile,
        )

    table = Table("metric", "value")
    table.add_row("raw rows scanned", f"{stats.raw_rows:,}")
    table.add_row("brand-connected ids", f"{stats.brand_connected_ids:,}")
    table.add_row("rows kept", f"{stats.kept_rows:,}")
    table.add_row("duplicate ids dropped", f"{stats.duplicate_ids_dropped:,}")
    table.add_row("undated rows dropped", f"{stats.undated_rows_dropped:,}")
    table.add_row("multi-part replies merged", str(stats.multipart_merges))
    table.add_row("training window rows", f"{stats.training_rows:,}")
    table.add_row("eval window rows", f"{stats.eval_rows:,}")
    console.print(table)
    console.print(f"[green]wrote {cfg.data.brand_parquet}[/green]")

    if keep_raw:
        console.print(f"[dim]--keep-raw set, leaving {cfg.data.raw_csv} in place.[/dim]")
    else:
        cfg.data.raw_csv.unlink()
        console.print(f"[dim]deleted {cfg.data.raw_csv} (pass --keep-raw to retain it).[/dim]")


@app.command("cache-stats")
def cache_stats(config: Path = typer.Option(None, "--config", "-c")) -> None:
    """Show what is in the response cache, by model."""
    cfg = _cfg(config)
    with LLMClient(cfg, offline=True) as client:
        stats = client.cache.stats()
        if not stats:
            console.print("[yellow]Cache is empty.[/yellow]")
            return
        table = Table("provider/model", "responses")
        for model, n in sorted(stats.items(), key=lambda kv: -kv[1]):
            table.add_row(model, str(n))
        table.add_row("[bold]total[/bold]", f"[bold]{len(client.cache)}[/bold]")
        console.print(table)


@app.command("config-show")
def config_show(config: Path = typer.Option(None, "--config", "-c")) -> None:
    """Print the resolved configuration."""
    cfg = _cfg(config)
    table = Table("role", "primary", "fallbacks", "temp", "max_tokens")
    for name, rcfg in cfg.roles.items():
        table.add_row(
            name,
            str(rcfg.primary),
            ", ".join(str(f) for f in rcfg.fallbacks) or "-",
            str(rcfg.temperature),
            str(rcfg.max_tokens),
        )
    console.print(table)


if __name__ == "__main__":
    app()
