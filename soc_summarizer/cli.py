"""Command-line interface for the SOC offense summarizer.

Examples
--------
    # Analyse the bundled sample offense
    soc-summarize --offense examples/offense_sample.json

    # Read an offense from stdin and emit machine-readable JSON
    cat offense.json | soc-summarize --format json

    # Inspect retrieval only (no LLM call) — useful for debugging the RAG step
    soc-summarize --offense examples/offense_sample.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .models import AnalysisResult, RetrievedChunk


def _read_offense(path: str | None) -> dict:
    if path in (None, "-"):
        data = sys.stdin.read()
    else:
        data = Path(path).read_text(encoding="utf-8")
    return json.loads(data)


def _build_settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    overrides: dict = {}
    if args.kb_dir:
        overrides["knowledge_base_dir"] = Path(args.kb_dir)
    if args.model:
        overrides["llm_model"] = args.model
    if args.embedding_model:
        overrides["embedding_model"] = args.embedding_model
    if args.host:
        overrides["ollama_host"] = args.host
    if args.top_k is not None:
        overrides["top_k"] = args.top_k
    return settings if not overrides else Settings(**{**settings.__dict__, **overrides})


# -- rendering ----------------------------------------------------------------

_CONF_ICON = {"High": "●●●", "Medium": "●●○", "Low": "●○○"}


def _render_sources(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        section = f" / {c.heading}" if c.heading else ""
        lines.append(f"  [{i}] {c.source}{section}  (relevance {c.score:.2f})")
    return "\n".join(lines) if lines else "  (none)"


def _render_text(result: AnalysisResult) -> str:
    a = result.analysis
    verdict = "TRUE POSITIVE (incident)" if a.classification == "TP" else "FALSE POSITIVE (benign)"
    conf = f"{a.confidence} ({a.confidence_pct}%) {_CONF_ICON.get(a.confidence, '')}"
    out = [
        "=" * 70,
        " SOC OFFENSE — AI TRIAGE SUMMARY",
        "=" * 70,
        "",
        "SUMMARY",
        f"  {a.summary}",
        "",
        f"CLASSIFICATION : {verdict}",
        f"CONFIDENCE     : {conf}",
        "",
        "RATIONALE",
        f"  {a.classification_rationale}",
        "",
        "RECOMMENDED ACTION",
        f"  {a.recommended_action}",
    ]
    if a.key_indicators:
        out += ["", "KEY INDICATORS"]
        out += [f"  - {ind}" for ind in a.key_indicators]
    if a.referenced_sources:
        out += ["", "REFERENCED PLAYBOOKS / SOURCES"]
        out += [f"  - {src}" for src in a.referenced_sources]
    out += [
        "",
        "-" * 70,
        f" Retrieved evidence (query: {result.retrieval_query[:90]}...)",
        _render_sources(result.retrieved_chunks),
        "",
        f" models: llm={result.llm_model}  embeddings={result.embedding_model}",
        "=" * 70,
    ]
    return "\n".join(out)


def _render_dry_run(query: str, chunks: list[RetrievedChunk]) -> str:
    out = [
        "=" * 70,
        " DRY RUN — retrieval only, no LLM call",
        "=" * 70,
        "",
        "RETRIEVAL QUERY",
        f"  {query}",
        "",
        f"TOP {len(chunks)} KNOWLEDGE-BASE PASSAGES",
    ]
    for i, c in enumerate(chunks, start=1):
        section = f" / {c.heading}" if c.heading else ""
        out += [
            "",
            f"  [{i}] {c.source}{section}  (relevance {c.score:.2f})",
            "      " + c.text.replace("\n", "\n      "),
        ]
    out += ["", "=" * 70]
    return "\n".join(out)


# -- main ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soc-summarize",
        description="Generate a structured AI triage summary for a SIEM offense (RAG over a local knowledge base, served by Ollama).",
    )
    parser.add_argument(
        "--offense",
        "-f",
        help="Path to the offense JSON file. Use '-' or omit to read from stdin.",
        default=None,
    )
    parser.add_argument("--kb-dir", help="Override the knowledge-base directory.")
    parser.add_argument("--model", help="Override the Ollama chat model (e.g. qwen2.5:14b).")
    parser.add_argument("--embedding-model", help="Override the Ollama embedding model.")
    parser.add_argument("--host", help="Override the Ollama host URL.")
    parser.add_argument("--top-k", type=int, help="Number of KB passages to retrieve.")
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run retrieval only and print the query + retrieved passages (no LLM call).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        offense_data = _read_offense(args.offense)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: could not read offense JSON: {exc}", file=sys.stderr)
        return 2

    settings = _build_settings(args)

    # Import here so `--help` and arg parsing never trigger backend construction.
    from .pipeline import OffenseSummarizer
    from .llm import OllamaConnectionError

    try:
        summarizer = OffenseSummarizer(settings)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"error: failed to initialise (is Ollama running and are the models pulled?): {exc}", file=sys.stderr)
        return 1

    print(
        f"[indexed {summarizer.chunk_count} chunks from "
        f"{summarizer.document_count} documents]",
        file=sys.stderr,
    )

    try:
        if args.dry_run:
            query, chunks = summarizer.retrieve(offense_data)
            if args.format == "json":
                payload = {"retrieval_query": query, "retrieved_chunks": [c.model_dump() for c in chunks]}
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                print(_render_dry_run(query, chunks))
            return 0

        result = summarizer.analyze(offense_data)
    except OllamaConnectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"error: analysis failed: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(result.model_dump_json(indent=2))
    else:
        print(_render_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
