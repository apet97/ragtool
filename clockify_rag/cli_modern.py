"""Modern Typer-based CLI for Clockify RAG system.

Provides commands:
- ragctl doctor: System diagnostics and configuration check
- ragctl ingest: Build index from knowledge base
- ragctl query: Single query (non-interactive)
- ragctl chat: Interactive REPL
- ragctl eval: Run RAGAS evaluation
"""

import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import config
from . import cli as legacy_cli
from .answer import answer_once
from .caching import get_query_cache
from .cli import ensure_index_ready, chat_repl
from .embedding import _load_st_encoder
from .indexing import build, load_index
from .logging_utils import log_query_event
from .utils import check_ollama_connectivity, check_pytorch_mps
from .exceptions import IndexLoadError

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="Clockify RAG Command-Line Interface")


# ============================================================================
# Doctor Command: System Diagnostics
# ============================================================================


def get_device_info() -> dict:
    """Detect and return device information."""
    try:
        import torch

        device = "cpu"
        reason = "default"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            reason = "Metal Performance Shaders (Apple Silicon)"
        elif torch.cuda.is_available():
            device = "cuda"
            reason = f"CUDA ({torch.cuda.get_device_name(0)})"

        return {
            "device": device,
            "reason": reason,
            "torch_version": torch.__version__,
            "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        }
    except ImportError:
        return {
            "device": "unknown",
            "reason": "torch not installed",
            "torch_version": "N/A",
            "mps_available": False,
        }
    except Exception as e:
        return {
            "device": "error",
            "reason": str(e),
            "torch_version": "N/A",
            "mps_available": False,
        }


def get_dependency_info() -> dict:
    """Check for key dependencies."""
    deps = {}

    packages = [
        "numpy",
        "torch",
        "sentence_transformers",
        "faiss",
        "hnswlib",
        "rank_bm25",
        "fastapi",
        "uvicorn",
        "requests",
        "typer",
    ]

    for pkg in packages:
        try:
            mod = __import__(pkg)
            deps[pkg] = {
                "installed": True,
                "version": getattr(mod, "__version__", "unknown"),
            }
        except ImportError:
            deps[pkg] = {
                "installed": False,
                "version": "N/A",
            }

    return deps


def get_index_info() -> dict:
    """Check index files and their status."""
    info = {}
    required_files = [
        config.FILES["chunks"],
        config.FILES["emb"],
        config.FILES["meta"],
        config.FILES["bm25"],
        config.FILES["index_meta"],
    ]

    for key, fname in config.FILES.items():
        exists = os.path.exists(fname)
        size = os.path.getsize(fname) if exists else 0
        info[key] = {
            "file": fname,
            "exists": exists,
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2) if size > 0 else 0,
        }

    all_required = all(os.path.exists(fname) for fname in required_files)
    return {
        "files": info,
        "index_ready": all_required,
    }


@app.command()
def doctor(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Detailed output"),
    json_output: bool = typer.Option(False, "--json", help="JSON output for scripting"),
) -> None:
    """Run diagnostics on system and configuration.

    Checks:
    - Python version and platform
    - Device detection (CPU/MPS/CUDA)
    - Key dependencies
    - Index files and status
    - Configuration validation
    - Ollama connectivity
    """
    if json_output:
        # JSON mode for scripting
        output = {
            "system": {
                "platform": platform.system(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
                "python_executable": sys.executable,
            },
            "device": get_device_info(),
            "dependencies": get_dependency_info(),
            "index": get_index_info(),
            "config": {
                "ollama_url": config.OLLAMA_URL,
                "gen_model": config.GEN_MODEL,
                "emb_model": config.EMB_MODEL,
                "chunk_size": config.CHUNK_CHARS,
                "top_k": config.DEFAULT_TOP_K,
                "pack_top": config.DEFAULT_PACK_TOP,
            },
        }

        # Check Ollama connectivity
        try:
            check_ollama_connectivity(config.OLLAMA_URL, timeout=3)
            output["ollama"] = {"connected": True, "error": None}
        except Exception as e:
            output["ollama"] = {"connected": False, "error": str(e)}

        console.print(json.dumps(output, indent=2))
        return

    # Rich console output
    console.print(Panel("🔍 Clockify RAG System Diagnostics", style="bold blue"))
    console.print()

    # System Info
    table = Table(title="System Information", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Platform", platform.system())
    table.add_row("Architecture", platform.machine())
    table.add_row("Python Version", platform.python_version())
    table.add_row("Python Executable", sys.executable)
    console.print(table)
    console.print()

    # Device Info
    device_info = get_device_info()
    device_emoji = "🚀" if device_info["device"] != "cpu" else "📱"
    device_table = Table(title=f"{device_emoji} Device Detection", show_header=False)
    device_table.add_column("Key", style="cyan")
    device_table.add_column("Value", style="white")
    device_table.add_row("Device", device_info["device"].upper())
    device_table.add_row("Reason", device_info["reason"])
    device_table.add_row("PyTorch Version", device_info["torch_version"])
    device_table.add_row("MPS Available", "✅ Yes" if device_info["mps_available"] else "❌ No")
    console.print(device_table)
    console.print()

    # Dependencies
    deps = get_dependency_info()
    deps_table = Table(title="📦 Key Dependencies", show_header=True)
    deps_table.add_column("Package", style="cyan")
    deps_table.add_column("Status", style="white")
    deps_table.add_column("Version", style="green")
    for pkg, info in sorted(deps.items()):
        status = "✅" if info["installed"] else "❌"
        version = info["version"] if info["installed"] else "—"
        deps_table.add_row(pkg, status, version)
    console.print(deps_table)
    console.print()

    # Index Status
    index_info = get_index_info()
    index_ready = index_info["index_ready"]
    index_emoji = "✅" if index_ready else "❌"
    console.print(f"{index_emoji} Index Status: {'READY' if index_ready else 'NOT READY (run: ragctl ingest)'}")
    console.print()

    # Ollama Connectivity
    try:
        normalized_url = check_ollama_connectivity(config.OLLAMA_URL, timeout=3)
        console.print(f"✅ Ollama: Connected to {normalized_url}")
    except Exception as e:
        console.print(f"❌ Ollama: Connection failed - {e}")
    console.print()

    # Configuration Summary
    config_table = Table(title="⚙️ Configuration", show_header=False)
    config_table.add_column("Key", style="cyan")
    config_table.add_column("Value", style="white")
    config_table.add_row("Ollama URL", config.OLLAMA_URL)
    config_table.add_row("Generation Model", config.GEN_MODEL)
    config_table.add_row("Embedding Model", config.EMB_MODEL)
    config_table.add_row("Chunk Size", str(config.CHUNK_CHARS))
    config_table.add_row("Top-K Retrieval", str(config.DEFAULT_TOP_K))
    config_table.add_row("Pack Top", str(config.DEFAULT_PACK_TOP))
    console.print(config_table)
    console.print()

    if verbose:
        # Detailed index file listing
        console.print("[bold]📁 Index Files (Detailed):[/bold]")
        for key, file_info in index_info["files"].items():
            status = "✅" if file_info["exists"] else "❌"
            size_str = f"{file_info['size_mb']} MB" if file_info["size_mb"] > 0 else "—"
            console.print(f"  {status} {key:20} {file_info['file']:30} {size_str}")
        console.print()

    console.print("✨ Diagnostics complete!")


# ============================================================================
# Ingest Command: Build Index
# ============================================================================


@app.command()
def ingest(
    input: Optional[str] = typer.Option(
        None, "--input", "-i", help="Input markdown file or directory (default: knowledge_full.md)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for index (default: current directory)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force rebuild even if index exists"),
) -> None:
    """Build or rebuild the index from knowledge base.

    Performs:
    1. Chunking: Split markdown into semantic chunks
    2. Embedding: Generate vector embeddings
    3. Indexing: Build FAISS/HNSW indexes and BM25 index
    4. Validation: Verify all artifacts

    Example:
        ragctl ingest --input ./docs --output ./var/index
    """
    input_file = input or "knowledge_full.md"
    output_dir = output or "."

    if not os.path.exists(input_file):
        console.print(f"❌ Input file not found: {input_file}")
        raise typer.Exit(1)

    console.print(f"📥 Ingesting: {input_file}")
    console.print(f"📤 Output directory: {output_dir}")

    try:
        build(input_file, retries=2)
        console.print("✅ Index built successfully!")

        # Verify
        idx_info = get_index_info()
        if idx_info["index_ready"]:
            console.print("✅ All artifacts verified")
        else:
            console.print("⚠️ Some artifacts missing")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"❌ Build failed: {e}")
        logger.error(f"Build error: {e}", exc_info=True)
        raise typer.Exit(1)


# ============================================================================
# Query Command: Single Query
# ============================================================================


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(15, "--top-k", help="Number of chunks to retrieve"),
    pack_top: int = typer.Option(8, "--pack-top", help="Number of chunks to include in context"),
    threshold: float = typer.Option(0.25, "--threshold", help="Minimum similarity threshold"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
) -> None:
    """Ask a single question and get an answer.

    Example:
        ragctl query "How do I track time in Clockify?"
    """
    console.print(f"❓ Question: {question}")

    try:
        chunks, vecs_n, bm, hnsw = ensure_index_ready(retries=2)

        call_start = time.time()
        result = answer_once(
            question,
            chunks,
            vecs_n,
            bm,
            top_k=top_k,
            pack_top=pack_top,
            threshold=threshold,
            hnsw=hnsw,
        )
        latency_ms = (time.time() - call_start) * 1000

        answer_text = result.get("answer", "")
        selected_chunks = result.get("selected_chunks", [])
        metadata = result.get("metadata", {}) or {}

        if json_output:
            output = {
                "question": question,
                "answer": answer_text,
                "confidence": result.get("confidence"),
                "sources": selected_chunks,
                "num_sources": len(selected_chunks),
                "metadata": metadata,
            }
            console.print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            console.print()
            console.print(answer_text)
            if debug and selected_chunks:
                console.print()
                console.print(f"[dim]Sources: {selected_chunks[:3]}...[/dim]")
                if metadata:
                    console.print(f"[dim]Metadata: {metadata}[/dim]")

        log_query_event(
            question,
            result,
            chunks,
            latency_ms,
            channel="cli-modern.query",
            disabled=getattr(legacy_cli, "QUERY_LOG_DISABLED", False),
        )

    except IndexLoadError as exc:
        console.print(f"❌ Index load error: {exc}")
        logger.error("Index load error: %s", exc)
        raise typer.Exit(getattr(exc, "exit_code", 1))
    except Exception as e:
        console.print(f"❌ Error: {e}")
        logger.error(f"Query error: {e}", exc_info=True)
        raise typer.Exit(1)


# ============================================================================
# Chat Command: Interactive REPL
# ============================================================================


@app.command()
def chat(
    top_k: int = typer.Option(15, "--top-k", help="Number of chunks to retrieve"),
    pack_top: int = typer.Option(8, "--pack-top", help="Number of chunks to include in context"),
    threshold: float = typer.Option(0.25, "--threshold", help="Minimum similarity threshold"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
) -> None:
    """Start interactive chat REPL.

    Commands:
        :exit    - Quit
        :debug   - Toggle debug output
        :config  - Show configuration
        :help    - Show help

    Example:
        ragctl chat
        > What is Clockify?
        > How do I set up SSO?
        > :exit
    """
    console.print(Panel("💬 Clockify RAG Chat", style="bold green"))
    console.print("Type ':exit' to quit, ':debug' to toggle debug, ':help' for help")
    console.print()

    try:
        chat_repl(
            top_k=top_k,
            pack_top=pack_top,
            threshold=threshold,
            debug=debug,
        )
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!")
        raise typer.Exit(0)
    except IndexLoadError as exc:
        console.print(f"❌ Index load error: {exc}")
        logger.error("Index load error: %s", exc)
        raise typer.Exit(getattr(exc, "exit_code", 1))
    except Exception as e:
        console.print(f"❌ Error: {e}")
        logger.error(f"Chat error: {e}", exc_info=True)
        raise typer.Exit(1)


# ============================================================================
# Eval Command: RAGAS Evaluation
# ============================================================================


@app.command()
def eval(
    questions_file: str = typer.Option(
        "data/eval_questions.jsonl", "--questions", "-q", help="Questions JSONL file"
    ),
    output_dir: str = typer.Option("var/reports", "--output", "-o", help="Output directory for reports"),
    sample_size: Optional[int] = typer.Option(None, "--sample", "-s", help="Sample size (default: all)"),
    metrics: Optional[str] = typer.Option(
        "faithfulness,answer_relevancy",
        "--metrics",
        "-m",
        help="Comma-separated metrics to compute",
    ),
) -> None:
    """Run RAGAS evaluation on a set of questions.

    Computes metrics:
    - Faithfulness: Is the answer faithful to the context?
    - Answer Relevancy: Is the answer relevant to the question?
    - Context Precision: Is the context relevant to the question?
    - Context Recall: Does the context contain all relevant information?

    Example:
        ragctl eval --questions data/questions.jsonl --metrics faithfulness,answer_relevancy
    """
    console.print(f"📊 Running evaluation on {questions_file}")

    try:
        import ragas

        console.print(f"✅ RAGAS {ragas.__version__} loaded")
    except ImportError:
        console.print("❌ RAGAS not installed. Install with: pip install ragas datasets evaluate")
        raise typer.Exit(1)

    try:
        if not os.path.exists(questions_file):
            console.print(f"❌ Questions file not found: {questions_file}")
            raise typer.Exit(1)

        os.makedirs(output_dir, exist_ok=True)

        # Load index
        chunks, vecs_n, bm, hnsw = ensure_index_ready(retries=2)

        console.print(f"📥 Loaded index with {len(chunks)} chunks")
        console.print(f"📊 Metrics: {metrics}")

        # FUTURE FEATURE: RAGAS evaluation loop
        # Planned implementation:
        # 1. Load questions from JSONL
        # 2. Generate answers using answer_once()
        # 3. Compute RAGAS metrics (faithfulness, answer_relevancy, etc.)
        # 4. Generate report with scores and analysis
        # 5. Save to output_dir
        #
        # For now, use eval.py script for evaluation.
        console.print("⚠️  RAGAS evaluation via CLI not yet implemented")
        console.print("💡 Use eval.py script instead:")
        console.print(f"   python eval.py --dataset {questions_file}")
        raise typer.Exit(0)

    except IndexLoadError as exc:
        console.print(f"❌ Index load error: {exc}")
        logger.error("Eval index load error: %s", exc)
        raise typer.Exit(getattr(exc, "exit_code", 1))
    except Exception as e:
        console.print(f"❌ Evaluation failed: {e}")
        logger.error(f"Eval error: {e}", exc_info=True)
        raise typer.Exit(1)


# ============================================================================
# Entry Point
# ============================================================================


if __name__ == "__main__":
    app()
