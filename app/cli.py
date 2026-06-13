from __future__ import annotations

from pathlib import Path

import typer

from app.config import Settings
from app.errors import SefmError
from app.logging_config import setup_logging

app = typer.Typer(
    name="sefm",
    no_args_is_help=True,
    help="Offline RAG assistant for ICS/SCADA technical documentation.",
)


@app.callback()
def _init(ctx: typer.Context) -> None:
    s = Settings()
    setup_logging(s.log_level, s.resolved_log_file)


@app.command()
def info() -> None:
    """Show the current configuration and component status."""
    from app.hardware import detect_hardware, plan_acceleration

    settings = Settings()
    typer.echo("=== sefm configuration ===")
    typer.echo(f"  data_dir:        {settings.data_dir}")
    typer.echo(f"  models_dir:      {settings.models_dir}")
    typer.echo(f"  embedding_model: {settings.embedding_model}")
    typer.echo(f"  llm_model_path:  {settings.llm_model_path or '(unset)'}")
    typer.echo(f"  chroma_dir:      {settings.chroma_dir}")
    typer.echo(f"  bm25_dir:        {settings.bm25_dir}")

    profile = detect_hardware() if settings.hw_detect else None
    if profile is not None:
        plan = plan_acceleration(profile, settings)
        typer.echo(f"  {plan.summary()}")
        typer.echo("  (run `sefm hardware` for the full report)")


@app.command()
def hardware() -> None:
    """Detect CPU/GPU/NPU and show the chosen acceleration plan."""
    from app.hardware import (
        detect_hardware,
        format_report,
        plan_acceleration,
        recommended_install_commands,
    )

    settings = Settings()
    typer.echo("Probing hardware ...")
    profile = detect_hardware()
    plan = plan_acceleration(profile, settings)
    typer.echo("=== hardware & acceleration ===")
    typer.echo(format_report(profile, plan))

    cmds = recommended_install_commands(profile, settings.acceleration, settings.embedding_device)
    if cmds:
        typer.echo("\nTo enable the above, run from S:\\SEFM:")
        for c in cmds:
            typer.echo(f"  {c}")


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, help="PDF file or folder of PDFs"),  # noqa: B008
) -> None:
    """Ingest a PDF or folder of PDFs into the local index."""
    from app.factory import build_app_service

    try:
        svc = build_app_service(with_llm=False)
        typer.echo(f"Ingesting {path} ...")
        n = svc.ingest_path(path)
    except SefmError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    typer.secho(f"  indexed {n} chunks", fg=typer.colors.GREEN)
    for name, reason in getattr(svc.ingestion, "skipped", []):
        typer.secho(f"  skipped {name}: {reason}", fg=typer.colors.YELLOW)


@app.command()
def search(
    question: str = typer.Argument(..., help="Natural-language query"),
    top_n: int = typer.Option(5, "--top", "-n", help="Number of results to show"),
) -> None:
    """Hybrid search over the indexed documents (no LLM required)."""
    from app.factory import build_app_service

    try:
        svc = build_app_service(with_llm=False)
        hits = svc.search(question, top_n=top_n)
    except SefmError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    if not hits:
        typer.secho("No results. Did you ingest anything?", fg=typer.colors.YELLOW)
        return
    for i, hit in enumerate(hits, 1):
        c = hit.chunk
        snippet = _snippet(c.text, 220)
        typer.echo(f"\n[{i}] {c.source}  p.{c.page}  ({c.kind})  score={hit.score:.3f}")
        typer.echo(f"    {snippet}")


@app.command()
def gui() -> None:
    """Launch the desktop UI."""
    from app.ui.run import main as gui_main

    raise typer.Exit(code=gui_main())


@app.command()
def ask(question: str = typer.Argument(..., help="Natural-language question")) -> None:
    """Generate a cited answer (requires SEFM_LLM_MODEL_PATH)."""
    from app.factory import build_app_service

    svc = build_app_service(with_llm=True)
    if svc.llm is None:
        typer.secho(
            "No LLM configured. Set SEFM_LLM_MODEL_PATH to a GGUF file. "
            "Use `search` for retrieval-only.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=2)
    try:
        result = svc.ask(question)
    except SefmError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    typer.echo(result.answer)


def _snippet(text: str, max_len: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= max_len else flat[: max_len - 1] + "..."


if __name__ == "__main__":
    app()
