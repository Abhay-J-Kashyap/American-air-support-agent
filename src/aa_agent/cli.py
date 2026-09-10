"""Command line entry points."""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pandas as pd
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


@app.command("discover-intents")
def discover_intents(
    config: Path = typer.Option(None, "--config", "-c"),
    sample: int = typer.Option(
        15000, "--sample", help="Max first-turn messages to cluster (0 = all)."
    ),
    k_min: int = typer.Option(8, "--k-min"),
    k_max: int = typer.Option(14, "--k-max"),
    k: int = typer.Option(None, "--k", help="Force a specific k, overriding the silhouette pick."),
    parallel: int = typer.Option(
        0,
        "--parallel",
        help="Embedding workers: 0=all cores, -1=single-core, N=that many.",
    ),
    out_dir: Path = typer.Option(Path("artifacts/intents"), "--out"),
) -> None:
    """Cluster first-turn customer messages and propose an intent taxonomy.

    Writes taxonomy.json plus cluster_audit.json -- the audit trail is a
    deliverable, not a debug artifact: it evidences that the taxonomy was
    derived from the data rather than asserted.
    """
    from aa_agent.embed import FastEmbedEmbedder, embed_cached, l2_normalize  # noqa: PLC0415
    from aa_agent.intents import (  # noqa: PLC0415
        WEAK_STRUCTURE_THRESHOLD,
        extract_first_inbound,
        fit_clusters,
        label_clusters,
        pick_best_k,
        representatives,
        silhouette_is_monotonic_decreasing,
        structure_is_weak,
        sweep_k,
        write_taxonomy,
    )

    cfg = _cfg(config)
    if not cfg.data.brand_parquet.exists():
        console.print(f"[red]{cfg.data.brand_parquet} not found. Run `make data` first.[/red]")
        raise typer.Exit(code=1)

    df = pd.read_parquet(cfg.data.brand_parquet)
    first = extract_first_inbound(df, training_only=True)
    console.print(f"first-turn messages in training window: [bold]{len(first):,}[/bold]")

    if sample and len(first) > sample:
        first = first.sample(n=sample, random_state=cfg.project.seed).reset_index(drop=True)
        console.print(f"sampled down to [bold]{len(first):,}[/bold] for clustering")

    texts = first["text_scrubbed"].astype(str).tolist()

    # Model load and encoding are reported separately: previously both sat
    # under one static "Embedding..." spinner, so a slow first-run model
    # download was indistinguishable from a hung encode.
    parallel_arg = None if parallel < 0 else parallel
    with console.status(f"Loading {cfg.embedding.model}..."):
        embedder = FastEmbedEmbedder(cfg.embedding.model, cfg.embedding.dim, parallel_arg)
    console.print(f"model ready, encoding [bold]{len(texts):,}[/bold] messages")

    started = time.monotonic()
    X = l2_normalize(
        embed_cached(texts, embedder, Path("artifacts/embeddings"), cfg.embedding.dtype)
    )
    elapsed = time.monotonic() - started
    rate = len(texts) / elapsed if elapsed > 0 else 0
    console.print(f"embedded in [bold]{elapsed:.1f}s[/bold] ({rate:,.0f}/s)")

    with console.status(f"Sweeping k={k_min}..{k_max}..."):
        sweep = sweep_k(X, list(range(k_min, k_max + 1)), seed=cfg.project.seed)

    sweep_table = Table("k", "silhouette", "inertia")
    for r in sweep:
        sweep_table.add_row(str(r.k), f"{r.silhouette:.4f}", f"{r.inertia:,.0f}")
    console.print(sweep_table)

    best_k = pick_best_k(sweep)
    max_sil = max(r.silhouette for r in sweep)

    if structure_is_weak(sweep):
        console.print(
            f"[yellow]Best silhouette is {max_sil:.3f} (<{WEAK_STRUCTURE_THRESHOLD}): the data "
            "shows no substantial cluster structure. Short-text embeddings often form a "
            "continuum rather than discrete groups, so k-means boundaries here are imposed, "
            "not discovered. Treat k as a human choice about taxonomy granularity, and say "
            "so in the report -- do not present it as a discovered optimum.[/yellow]"
        )
    if silhouette_is_monotonic_decreasing(sweep):
        console.print(
            "[yellow]Silhouette falls monotonically with k, so argmax necessarily returns the "
            "smallest k swept. The 'chosen' k is an artifact of --k-min, not a finding. "
            "Use --k to set it deliberately.[/yellow]"
        )

    if k is not None:
        best_k = k
        console.print(f"using [bold]k={best_k}[/bold] (set explicitly via --k)")
    else:
        console.print(f"chosen k = [bold]{best_k}[/bold] (highest silhouette = {max_sil:.3f})")

    labels, centroids = fit_clusters(X, best_k, seed=cfg.project.seed)
    sizes = {int(c): int((labels == c).sum()) for c in set(labels.tolist())}
    reps = representatives(X, labels, centroids, texts, n_per_cluster=12)

    with console.status(f"Labelling {best_k} clusters..."), LLMClient(cfg) as client:
        clusters = label_clusters(client, reps, sizes)

    result = Table("id", "label", "size", "description")
    for c in clusters:
        pct = 100 * c.size / len(texts)
        result.add_row(str(c.cluster_id), c.label, f"{c.size:,} ({pct:.1f}%)", c.description)
    console.print(result)

    # Surface the two things that most often make a taxonomy wrong, rather
    # than leaving them for the reader to spot in the table.
    from collections import Counter  # noqa: PLC0415

    dupes = [lbl for lbl, n in Counter(c.label for c in clusters).items() if n > 1]
    if dupes:
        console.print(
            f"[yellow]Duplicate labels: {', '.join(dupes)}. Either merge these clusters "
            "or split them further -- decide by reading cluster_audit.json.[/yellow]"
        )
    failed = [c.cluster_id for c in clusters if c.label.startswith("unlabelled_cluster_")]
    if failed:
        console.print(
            f"[red]{len(failed)} cluster(s) failed to parse: {failed}. "
            "Raw responses are in cluster_audit.json under 'raw_response'.[/red]"
        )
    if best_k in (k_min, k_max):
        console.print(
            f"[yellow]Chosen k={best_k} is at the edge of the swept range "
            f"({k_min}-{k_max}); the true optimum may lie outside it. "
            "Re-run with a wider range to check.[/yellow]"
        )

    write_taxonomy(clusters, sweep, out_dir)
    console.print(f"[green]wrote {out_dir}/taxonomy.json and cluster_audit.json[/green]")
    console.print(
        f"[yellow]Next: this is a PROPOSAL only. Curate it by hand into "
        f"{cfg.taxonomy_path}, which is the source of truth downstream and is never "
        "overwritten by this command. Merge near-duplicate intents, rename anything "
        "vague, set risk_tier per intent, and record changes in DECISIONS.md.[/yellow]"
    )


@app.command("taxonomy-show")
def taxonomy_show(config: Path = typer.Option(None, "--config", "-c")) -> None:
    """Validate and display the curated taxonomy.

    This is the file downstream stages read -- NOT the proposal that
    `discover-intents` writes. Run it after hand-editing to confirm the file
    still parses and the risk tiers are what you intend.
    """
    from aa_agent.taxonomy import IntentSource, load_taxonomy  # noqa: PLC0415

    cfg = _cfg(config)
    try:
        tax = load_taxonomy(cfg.taxonomy_path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print(f"[red]Taxonomy is invalid: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    table = Table("label", "tier", "source", "~share", "description")
    for i in tax.intents:
        tier_colour = {
            "always_escalate": "[red]always_escalate[/red]",
            "review": "[yellow]review[/yellow]",
            "auto_ok": "[green]auto_ok[/green]",
        }[i.risk_tier.value]
        share = f"{i.approx_share:.1%}" if i.approx_share is not None else "-"
        table.add_row(i.label, tier_colour, i.source.value, share, i.description)
    console.print(table)

    cluster_total = sum(
        i.approx_share for i in tax.intents if i.source is IntentSource.CLUSTER and i.approx_share
    )
    console.print(
        f"{len(tax.intents)} intents | cluster shares sum to {cluster_total:.1%} "
        "(these partition the sample; keyword shares are overlapping subsets on a "
        "different denominator, so the full column does NOT sum to 100%)"
    )
    console.print(f"always_escalate: [red]{', '.join(tax.always_escalate_labels())}[/red]")


@app.command("label")
def label(
    config: Path = typer.Option(None, "--config", "-c"),
    sample_path: Path = typer.Option(Path("artifacts/golden/sample.jsonl"), "--sample-file"),
    labels_path: Path = typer.Option(Path("artifacts/golden/labels.jsonl"), "--labels-file"),
    resample: bool = typer.Option(False, "--resample", help="Redraw the sample from scratch."),
) -> None:
    """Hand-label the golden set. Resumable -- stop and restart any time.

    Label BEFORE the agent exists (DECISIONS.md #1). Labelling afterwards
    lets your judgements drift toward whatever the system happens to output,
    which quietly turns the evaluation into a measure of self-agreement.
    """
    from aa_agent.golden import (  # noqa: PLC0415
        append_label,
        build_golden_sample,
        compile_golden_set,
        label_progress,
        load_examples,
        load_labels,
        save_examples,
    )
    from aa_agent.taxonomy import load_taxonomy  # noqa: PLC0415

    cfg = _cfg(config)
    tax = load_taxonomy(cfg.taxonomy_path)

    if resample or not sample_path.exists():
        if not cfg.data.brand_parquet.exists():
            console.print(f"[red]{cfg.data.brand_parquet} not found. Run `make data`.[/red]")
            raise typer.Exit(code=1)
        df = pd.read_parquet(cfg.data.brand_parquet)
        try:
            examples, stats = build_golden_sample(
                df,
                brand=cfg.project.brand,
                n_stratum_a=cfg.golden.stratum_a,
                n_stratum_b=cfg.golden.stratum_b,
                seed=cfg.project.seed,
            )
        except ValueError as exc:
            console.print(f"[red]Cannot draw the sample: {exc}[/red]")
            console.print(
                "Lower golden.stratum_a / golden.stratum_b in config.yaml, or widen the "
                "eval window by lowering data.split_quantile."
            )
            raise typer.Exit(code=1) from exc
        save_examples(examples, sample_path)
        console.print(
            f"drew [bold]{stats.total}[/bold] examples "
            f"(A={stats.stratum_a} uniform, B={stats.stratum_b} stratified) -> {sample_path}"
        )
        breakdown = Table("stratum B category", "n")
        for reason, n in sorted(stats.by_reason.items(), key=lambda kv: -kv[1]):
            if reason != "uniform_random":
                breakdown.add_row(reason, str(n))
        console.print(breakdown)
    else:
        examples = load_examples(sample_path)
        console.print(f"loaded [bold]{len(examples)}[/bold] sampled examples from {sample_path}")

    labels = load_labels(labels_path)
    done, total = label_progress(examples, labels)
    console.print(f"progress: [bold]{done}/{total}[/bold] labelled\n")

    intents = tax.labels
    todo = [e for e in examples if e.tweet_id not in labels or not labels[e.tweet_id].is_labelled]
    if not todo:
        console.print("[green]All examples labelled.[/green]")
        _write_golden_csv(compile_golden_set(examples, labels), labels_path)
        return

    console.print("[dim]Enter the number for each field. 's' skips, 'q' saves and quits.[/dim]\n")

    for n, ex in enumerate(todo, start=1):
        console.print(
            f"[bold cyan]── {done + n}/{total} ── stratum {ex.stratum} "
            f"({ex.sample_reason}) ──[/bold cyan]"
        )
        console.print(f"[white]{ex.text}[/white]\n")

        intent = _choose("INTENT", intents)
        if intent is None:
            console.print("[yellow]Saved. Re-run to continue.[/yellow]")
            break
        if intent == "__skip__":
            continue

        tier = tax.risk_tier(intent).value
        if tier == "always_escalate":
            console.print(f"[red]{intent} is always_escalate -- action forced.[/red]")
            action: str | None = "escalate"
        else:
            action = _choose("ACTION", ["auto", "escalate"])
            if action is None:
                console.print("[yellow]Saved. Re-run to continue.[/yellow]")
                break
            if action == "__skip__":
                continue

        reason = ""
        if action == "escalate":
            picked = _choose(
                "REASON",
                [
                    "high_risk_intent",
                    "needs_account_access",
                    "policy_or_compensation",
                    "angry_or_repeat_contact",
                    "ambiguous_request",
                    "other",
                ],
            )
            if picked is None:
                break
            reason = "" if picked == "__skip__" else picked

        ref_good: bool | None = None
        if ex.reference_reply:
            console.print(
                f"\n[dim]brand's actual reply:[/dim] [italic]{ex.reference_reply}[/italic]"
            )
            picked = _choose("WAS THAT REPLY ANY GOOD?", ["yes", "no"])
            if picked is None:
                break
            if picked != "__skip__":
                ref_good = picked == "yes"

        ex.label_intent = intent
        ex.label_action = action
        ex.label_reason = reason
        ex.reference_is_good = ref_good
        append_label(ex, labels_path)
        console.print("[green]saved[/green]\n")

    labels = load_labels(labels_path)
    done, total = label_progress(examples, labels)
    console.print(f"\nprogress: [bold]{done}/{total}[/bold]")
    _write_golden_csv(compile_golden_set(examples, labels), labels_path)


def _choose(prompt: str, options: list[str]) -> str | None:
    """Numbered prompt. Returns the choice, '__skip__', or None to quit."""
    console.print(f"[bold]{prompt}[/bold]")
    for i, opt in enumerate(options, start=1):
        console.print(f"  [cyan]{i}[/cyan] {opt}")
    while True:
        raw = typer.prompt("  >", default="", show_default=False).strip().lower()
        if raw == "q":
            return None
        if raw == "s":
            return "__skip__"
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        console.print("[red]  enter a number, 's' to skip, or 'q' to quit[/red]")


def _write_golden_csv(df: pd.DataFrame, labels_path: Path) -> None:
    out = labels_path.with_suffix(".csv")
    df.to_csv(out, index=False)
    console.print(f"[dim]wrote {out}[/dim]")


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
