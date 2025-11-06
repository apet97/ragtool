"""Utility functions for file I/O, validation, logging, and text processing."""

import atexit
import hashlib
import json
import logging
import os
import platform
import re
import tempfile
import time
import unicodedata
from contextlib import contextmanager
from typing import Any

import numpy as np

# Note: config will be imported at module level to avoid circular imports
# For functions that need dynamic config access, we'll import inside functions

logger = logging.getLogger(__name__)


# ====== CLEANUP HANDLERS ======
def _release_lock_if_owner():
    """Release build lock on exit if held by this process."""
    from .config import BUILD_LOCK

    try:
        if os.path.exists(BUILD_LOCK):
            with open(BUILD_LOCK) as f:
                data = json.loads(f.read())
            if data.get("pid") == os.getpid():
                os.remove(BUILD_LOCK)
                logger.debug("Cleaned up build lock")
    except (OSError, FileNotFoundError, json.JSONDecodeError, KeyError):
        # Cleanup failed - not critical, can ignore
        pass


atexit.register(_release_lock_if_owner)


# ====== PROCESS & LOCK MANAGEMENT ======
def _pid_alive(pid: int) -> bool:
    """Check if a process is alive. Cross-platform."""
    if pid <= 0:
        return False
    system = platform.system().lower()
    try:
        if system != "windows":
            # POSIX: use signal 0 check
            os.kill(pid, 0)
            return True
        else:
            # Windows: best-effort with optional psutil
            try:
                import psutil
                return psutil.pid_exists(pid)
            except Exception:
                # Fallback: treat as alive; bounded wait handles stale locks
                # Hint once for better DX
                try:
                    if not getattr(_pid_alive, "_hinted_psutil", False):
                        logger.debug("[build_lock] psutil not available on Windows; install 'psutil' for precise PID checks")
                        _pid_alive._hinted_psutil = True  # type: ignore
                except Exception:
                    pass
                return True
    except OSError:
        return False


@contextmanager
def build_lock():
    """Exclusive build lock with atomic create (O_EXCL) and stale-lock recovery.

    Uses atomic file creation to prevent partial writes. Detects stale locks via
    PID liveness check and TTL expiration.
    """
    from .config import BUILD_LOCK, BUILD_LOCK_TTL_SEC

    pid = os.getpid()
    hostname = platform.node() or "unknown"
    deadline = time.time() + 10.0  # 10s max wait

    while True:
        try:
            # Atomic create: fails if file exists (O_EXCL)
            fd = os.open(BUILD_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                with os.fdopen(fd, "w") as f:
                    started_at = time.time()
                    lock_data = {
                        "pid": pid,
                        "host": hostname,
                        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
                        "started_at_epoch": started_at,
                        "ttl_sec": BUILD_LOCK_TTL_SEC
                    }
                    f.write(json.dumps(lock_data))
                break  # Successfully acquired lock
            except Exception:
                try:
                    os.close(fd)
                except Exception:
                    pass
                raise
        except FileExistsError:
            # Lock file exists; check if it's stale
            try:
                with open(BUILD_LOCK, "r") as f:
                    lock_data = json.loads(f.read())
                stale_pid = lock_data.get("pid", 0)
                started_at_epoch = lock_data.get("started_at_epoch", 0)
                ttl_sec = lock_data.get("ttl_sec", BUILD_LOCK_TTL_SEC)

                # Check TTL expiration
                age = time.time() - started_at_epoch
                is_expired = age > ttl_sec
                pid_alive = _pid_alive(stale_pid)

                # If expired or dead owner, try to remove and retry
                if is_expired or not pid_alive:
                    reason = f"expired (age={age:.1f}s > ttl={ttl_sec}s)" if is_expired else f"dead PID {stale_pid}"
                    logger.warning(f"[build_lock] Recovering: {reason}")
                    try:
                        os.remove(BUILD_LOCK)
                        continue  # Retry atomic create
                    except Exception:
                        pass
            except FileNotFoundError:
                # Lock removed by another process between check and read, retry
                logger.debug("[build_lock] Lock removed during check, retrying...")
                continue
            except (json.JSONDecodeError, ValueError) as e:
                # Corrupt lock file, try to remove and retry
                logger.warning(f"[build_lock] Corrupt lock file: {e}")
                try:
                    os.remove(BUILD_LOCK)
                except Exception:
                    pass
                continue
            except Exception as e:
                logger.warning(f"[build_lock] Unexpected error reading lock: {e}")
                # Fall through to timeout logic

            # Still held by live process; wait and retry with 250 ms polling
            if time.time() > deadline:
                raise RuntimeError("Build already in progress; timed out waiting for lock release")
            while time.time() < deadline:  # Use deadline directly
                time.sleep(0.25)
                if not os.path.exists(BUILD_LOCK):
                    break
                if time.time() > deadline:  # Check deadline in loop
                    raise RuntimeError("Build already in progress; timed out waiting for lock release")
            continue

    try:
        yield
    finally:
        # Only remove lock if we still own it
        try:
            with open(BUILD_LOCK, "r") as f:
                lock_data = json.loads(f.read())
            if lock_data.get("pid") == os.getpid():
                os.remove(BUILD_LOCK)
        except Exception:
            pass


# ====== CONFIG VALIDATION ======
def validate_ollama_url(url: str) -> str:
    """Validate and normalize Ollama URL. Returns validated URL."""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            # Assume http if no scheme
            url = "http://" + url
            parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid scheme: {parsed.scheme}. Must be http or https.")
        if not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}. Must include host.")
        # Normalize: ensure no trailing slash
        url = f"{parsed.scheme}://{parsed.netloc}"
        if parsed.path and parsed.path != "/":
            url += parsed.path
        return url
    except Exception as e:
        raise ValueError(f"Invalid Ollama URL '{url}': {e}")


def validate_and_set_config(ollama_url=None, gen_model=None, emb_model=None, ctx_budget=None):
    """Validate and set global config from CLI args."""
    import clockify_rag.config as config

    if ollama_url:
        config.OLLAMA_URL = validate_ollama_url(ollama_url)
        logger.info(f"Ollama endpoint: {config.OLLAMA_URL}")

    if gen_model:
        config.GEN_MODEL = gen_model
        logger.info(f"Generation model: {config.GEN_MODEL}")

    if emb_model:
        config.EMB_MODEL = emb_model
        logger.info(f"Embedding model: {config.EMB_MODEL}")

    if ctx_budget:
        try:
            config.CTX_TOKEN_BUDGET = int(ctx_budget)
            if config.CTX_TOKEN_BUDGET < 256:
                raise ValueError("Context budget must be >= 256")
            logger.info(f"Context token budget: {config.CTX_TOKEN_BUDGET}")
        except ValueError as e:
            raise ValueError(f"Invalid context budget: {e}")


def validate_chunk_config():
    """Validate chunk parameters at startup."""
    from .config import CHUNK_CHARS, CHUNK_OVERLAP

    if CHUNK_OVERLAP >= CHUNK_CHARS:
        raise ValueError(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be < CHUNK_CHARS ({CHUNK_CHARS})")
    logger.debug(f"Chunk config: size={CHUNK_CHARS}, overlap={CHUNK_OVERLAP}")


def check_pytorch_mps():
    """Check PyTorch MPS availability on M1 Macs and log warnings (v4.1.2)."""
    is_macos_arm64 = platform.system() == "Darwin" and platform.machine() == "arm64"

    if not is_macos_arm64:
        return  # Only relevant for M1/M2/M3 Macs

    try:
        import torch
        mps_available = torch.backends.mps.is_available()

        if mps_available:
            logger.info("info: pytorch_mps=available platform=arm64 (GPU acceleration enabled)")
        else:
            logger.warning(
                "warning: pytorch_mps=unavailable platform=arm64 "
                "hint='Embeddings will use CPU (slower). Ensure macOS 12.3+ and PyTorch 1.12+'"
            )
            logger.warning("  To fix: pip install --upgrade torch or conda install -c pytorch pytorch")
    except ImportError:
        logger.debug("info: pytorch not imported, skipping MPS check")
    except Exception as e:
        logger.debug(f"info: pytorch_mps check failed: {e}")


def _log_config_summary(use_rerank=False, pack_top=None, seed=None, threshold=None,
                       top_k=None, num_ctx=None, num_predict=None, retries=0):
    """Log configuration summary at startup."""
    from .config import (GEN_MODEL, EMB_MODEL, DEFAULT_PACK_TOP, DEFAULT_SEED,
                        DEFAULT_THRESHOLD, DEFAULT_TOP_K, DEFAULT_NUM_CTX,
                        DEFAULT_NUM_PREDICT, EMB_READ_T, CHAT_READ_T,
                        RERANK_READ_T, REFUSAL_STR)

    # Use defaults if not provided
    pack_top = pack_top or DEFAULT_PACK_TOP
    seed = seed or DEFAULT_SEED
    threshold = threshold or DEFAULT_THRESHOLD
    top_k = top_k or DEFAULT_TOP_K
    num_ctx = num_ctx or DEFAULT_NUM_CTX
    num_predict = num_predict or DEFAULT_NUM_PREDICT

    proxy_trust = 1 if os.getenv("ALLOW_PROXIES") == "1" else 0
    # Single-line CONFIG banner
    logger.info(
        f"CONFIG model={GEN_MODEL} emb={EMB_MODEL} topk={top_k} pack={pack_top} thr={threshold} "
        f"seed={seed} ctx={num_ctx} pred={num_predict} retries={retries} "
        f"timeouts=(3,{int(EMB_READ_T)}/{int(CHAT_READ_T)}/{int(RERANK_READ_T)}) "
        f"trust_env={proxy_trust} rerank={1 if use_rerank else 0}"
    )
    # Print refusal string once for sanity
    logger.info(f'REFUSAL_STR="{REFUSAL_STR}"')


# ====== FILE I/O UTILITIES ======
def _fsync_dir(path: str) -> None:
    """Sync directory to ensure durability (best-effort, platform-dependent)."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    try:
        fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass  # Best-effort on platforms/filesystems without dir fsync


def atomic_write_bytes(path: str, data: bytes) -> None:
    """Atomically write bytes with fsync durability."""
    tmp = None
    try:
        d = os.path.dirname(os.path.abspath(path)) or "."
        with tempfile.NamedTemporaryFile(prefix=".tmp.", dir=d, delete=False) as f:
            tmp = f.name
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def atomic_write_text(path: str, text: str) -> None:
    """Atomically write text file with fsync durability."""
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str, obj: Any) -> None:
    """Atomically write JSON file."""
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False))


def atomic_write_jsonl(path: str, rows_list: list) -> None:
    """Atomically write JSONL file (list of dicts)."""
    # Build rows in memory as list of JSON strings
    lines = []
    for row in rows_list:
        if isinstance(row, dict):
            lines.append(json.dumps(row, ensure_ascii=False))
        else:
            lines.append(str(row))
    content = "\n".join(lines)
    if content and not content.endswith("\n"):
        content += "\n"
    atomic_write_text(path, content)


def atomic_save_npy(arr: np.ndarray, path: str) -> None:
    """Atomically save numpy array with fsync durability. Enforce float32."""
    # Enforce float32
    arr = arr.astype("float32")
    d = os.path.dirname(os.path.abspath(path)) or "."
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".tmp.", dir=d, delete=False) as f:
            tmp = f.name
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


# ====== LOGGING UTILITIES ======
def log_event(event: str, **fields):
    """Log a structured JSON event. Fallback to plain format if JSON serialization fails."""
    try:
        record = {"event": event, **fields}
        logger.info(json.dumps(record, ensure_ascii=False))
    except Exception:
        # Fallback to plain string if JSON encoding fails
        logger.info(f"{event} {fields}")


def log_kpi(topk: int, packed: int, used_tokens: int, rerank_applied: bool, rerank_reason: str = ""):
    """Log KPI metrics."""
    from .config import KPI

    logger.info(
        f"KPI retrieve={KPI.retrieve_ms}ms ann={KPI.ann_ms}ms "
        f"rerank={KPI.rerank_ms}ms ask={KPI.ask_ms}ms "
        f"topk={topk} packed={packed} used_tokens={used_tokens} "
        f"rerank={1 if rerank_applied else 0} reason={rerank_reason}"
    )


# ====== TEXT PROCESSING UTILITIES ======
def norm_ws(s: str) -> str:
    """Normalize whitespace."""
    return re.sub(r"[ \t]+", " ", s.strip())


def is_rtf(text: str) -> bool:
    """Check if text is RTF format."""
    # Check first 128 chars for RTF signature
    head_128 = text[:128]
    if "{\\rtf" in head_128 or "\\rtf" in head_128:
        return True

    # Check first 4096 chars for RTF control words (stricter)
    head_4k = text[:4096]
    rtf_commands = re.findall(r"\\(?:cf\d+|u[+-]?\d+\?|f\d+|pard)\b", head_4k)
    return len(rtf_commands) > 20


def strip_noise(text: str) -> str:
    """Drop scrape artifacts and normalize encoding."""
    # Guard: only apply RTF stripping if content is likely RTF
    if is_rtf(text):
        # Strip RTF escapes only for RTF content (more precise patterns)
        text = re.sub(r"\\cf\d+", "", text)  # \cfN (color)
        text = re.sub(r"\\u[+-]?\d+\?", "", text)  # \u1234? (unicode)
        text = re.sub(r"\\f\d+", "", text)  # \fN (font)
        text = re.sub(r"\{\\\*[^}]*\}", "", text)  # {\* ... } (special)
        text = re.sub(r"\\pard\b[^\n]*", "", text)  # \pard (paragraph)
    # Always remove chunk markers
    text = re.sub(r"^## +Chunk +\d+\s*$", "", text, flags=re.M)
    return text


def tokenize(s: str) -> list:
    """Simple tokenizer: lowercase [a-z0-9]+."""
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    return re.findall(r"[a-z0-9]+", s)


def approx_tokens(chars: int) -> int:
    """Estimate tokens: 1 token ≈ 4 chars."""
    return max(1, chars // 4)


def compute_sha256(filepath: str) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def truncate_to_token_budget(text: str, budget: int) -> str:
    """Truncate text to fit token budget, append ellipsis."""
    est_tokens = approx_tokens(len(text))
    if est_tokens <= budget:
        return text
    # Approximate char count for budget
    target_chars = budget * 4
    if len(text) <= target_chars:
        return text
    return text[:target_chars] + " […]"
