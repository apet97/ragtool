# Hardening Improvement Plan: Clockify Support CLI v3.2

**Date**: 2025-11-05
**Version**: 3.2
**Status**: Action Plan (15 Improvements Identified)

---

## Executive Summary

The hardened v3.1 implementation is solid but has 15 identified gaps that need fixing before full production deployment:

1. **Auth header redirect leak** (SECURITY) – Lines 155–177
2. **urllib3 compatibility incomplete** (ROBUSTNESS) – Lines 103–151
3. **Windows lock stall** (RELIABILITY) – Lines 197–273
4. **Determinism unproven** (CORRECTNESS) – Hashing/reproducibility claims unverified
5. **MMR signature mismatch** (CORRECTNESS) – Call site vs. definition
6. **Headroom enforcement missing** (CORRECTNESS) – HEADROOM_FACTOR=1.10 declared but not used
7. **Logging inconsistent** (OBSERVABILITY) – Mixed print() and logger calls
8. **Retry semantics unsafe** (RELIABILITY) – POST retries can double-bill
9. **Atomic write gaps** (DURABILITY) – Not all artifacts use atomic writes
10. **Timeout plumbing incomplete** (RELIABILITY) – Envs not fully propagated
11. **RTF guard too loose** (CORRECTNESS) – False-positives on non-RTF content
12. **Rerank fallback invisible** (OBSERVABILITY) – No logging on timeout/error
13. **Session proxy behavior undocumented** (SECURITY) – trust_env not set
14. **dtype consistency weak** (PERFORMANCE) – No end-to-end float32 guarantee
15. **Config validation timing** (OBSERVABILITY) – Effective config not printed at startup

---

## Detailed Issues & Fixes

### 1. AUTH HEADER REDIRECT LEAK (SECURITY) – HIGH

**Location**: Lines 153–177 in `_mount_retries()`

**Issue**:
```python
retry_strategy.remove_headers_on_redirect = frozenset()  # Preserves Authorization!
```

Setting `remove_headers_on_redirect = frozenset()` means *no* headers are removed on cross-origin redirects. This is the opposite of intended—it leaks the `Authorization` header to third parties.

**Intended Fix**:
```python
# Safe pattern: drop Authorization on cross-origin 30x
if orig_netloc != redir_netloc:
    # Cross-origin: drop Authorization to prevent credential leak
    retry_strategy.remove_headers_on_redirect = frozenset(["Authorization"])
else:
    # Same-origin: preserve all headers (safe)
    retry_strategy.remove_headers_on_redirect = frozenset()
```

**Current code already does this correctly** (lines 166–171). No change needed here, but verify the logic path is actually taken by adding a test.

**Action**: Add unit test to confirm auth header stripping on cross-origin redirects (Test #14 below).

---

### 2. URLLIB3 COMPATIBILITY INCOMPLETE – MEDIUM

**Location**: Lines 103–151 in `_mount_retries()`

**Issue**:
```python
retry_kwargs = {
    "total": retries,
    "connect": retries,
    "read": retries,
    "status": retries,
    "backoff_factor": 0.5,
    "raise_on_status": False,
}
# Try to add optional parameters that may not exist in older urllib3 versions
try:
    retry_strategy = Retry(**retry_kwargs, respect_retry_after_header=True)
except TypeError:
    retry_strategy = Retry(**retry_kwargs)  # ← Falls back but still uses old kwargs
```

**Problem**: If `status` or `backoff_factor` don't exist in ancient urllib3, the exception is caught but the same `retry_kwargs` is reused, causing the second attempt to fail identically. Also, `allowed_methods` vs. `method_whitelist` are added to `retry_kwargs` before the version check, so old urllib3 will reject them.

**Fix**:
```python
# Detect urllib3 version early
from urllib3.util.retry import Retry as RetryClass
retry_sig = RetryClass.__init__.__code__.co_varnames

# Build kwargs conditionally
retry_kwargs = {
    "total": retries,
    "connect": retries,
    "read": retries,
    "raise_on_status": False,
}

# Add optional parameters if supported
if "backoff_factor" in retry_sig:
    retry_kwargs["backoff_factor"] = 0.5
if "status_forcelist" in retry_sig:
    retry_kwargs["status_forcelist"] = [429, 500, 502, 503, 504]

# Detect and use correct method whitelist parameter
if "allowed_methods" in retry_sig:
    retry_kwargs["allowed_methods"] = frozenset(["POST", "GET"])
elif "method_whitelist" in retry_sig:
    retry_kwargs["method_whitelist"] = frozenset(["POST", "GET"])

# Try new parameter names first
if "respect_retry_after_header" in retry_sig:
    retry_kwargs["respect_retry_after_header"] = True

retry_strategy = Retry(**retry_kwargs)  # Single call, all params pre-validated
```

**Action**: Refactor `_mount_retries()` to detect and use correct parameter names per urllib3 version.

---

### 3. WINDOWS LOCK STALL (RELIABILITY) – MEDIUM

**Location**: Lines 197–273 in `_pid_alive()` and `build_lock()`

**Issue**:
```python
except Exception:
    # Fallback: assume alive; bounded wait will handle stale locks
    return True  # Always assumes alive on Windows if psutil missing
```

On Windows without `psutil`, `_pid_alive()` always returns `True`, even for dead PIDs. If a previous process crashed and left `.build.lock` with a dead PID, this code will:
1. Assume the PID is alive
2. Wait the full 10s deadline
3. Then raise `RuntimeError("Build already in progress...")` even though the lock holder is dead

**Fix**:
Add file mtime staleness as a backstop. Modify `build_lock()`:

```python
# If stale or dead owner, try to remove and retry
if not _pid_alive(stale_pid):
    try:
        os.remove(BUILD_LOCK)
        continue  # Retry atomic create
    except Exception:
        pass

# NEW: Check mtime as backup staleness indicator
try:
    mtime = os.path.getmtime(BUILD_LOCK)
    age_secs = time.time() - mtime
    if age_secs > 10.0:  # Lock older than 10 seconds is stale
        try:
            os.remove(BUILD_LOCK)
            continue
        except Exception:
            pass
except Exception:
    pass

# Still held by live process; wait and retry
if time.time() > deadline:
    raise RuntimeError("Build already in progress; timed out waiting for lock release")
time.sleep(0.25)
```

Also, write PID + timestamp in the lock file:
```python
lock_content = f"{os.getpid()}\n{time.time()}"  # PID + mtime
with os.fdopen(fd, "w") as f:
    f.write(lock_content)
```

Then read and parse both values in the check.

**Action**: Update `build_lock()` to write PID + timestamp and check mtime staleness as fallback.

---

### 4. DETERMINISM UNPROVEN (CORRECTNESS) – MEDIUM

**Location**: Lines 750–764 and 380–389 (rerank + generate)

**Issue**: The hardening document claims "deterministic: temperature=0, seed=42" but there is no proof that:
1. Two questions with identical input produce identical hashes
2. Two different questions produce different hashes
3. The system actually runs two distinct prompts and reports both hashes (as mentioned in hardening notes)

The visible code shows only `temperature=0, seed=42` in the payloads, which is correct, but there's no verification that the same question asked twice returns the same answer.

**Fix**: Add a determinism test to `main()` or a separate `test_determinism.sh`:

```python
def test_determinism(question: str, runs: int = 3):
    """Verify same question → same hash."""
    hashes = []
    for _ in range(runs):
        answer = answer_once(question, debug=False)
        h = hashlib.sha256(answer.encode()).hexdigest()
        hashes.append(h)

    if len(set(hashes)) == 1:
        logger.info(f"✓ Determinism OK: {len(runs)} runs → same hash")
        return True
    else:
        logger.warning(f"✗ Determinism FAILED: got {len(set(hashes))} unique hashes")
        return False
```

Add CLI flag:
```bash
python3 clockify_support_cli.py test-determinism "Your question"
```

**Action**: Add determinism test and validate reproducibility before marking as "fully deterministic."

---

### 5. MMR SIGNATURE MISMATCH (CORRECTNESS) – MEDIUM

**Location**: Line 670 (definition) vs. Line 101 (call)

**Issue**:
Definition (line 670):
```python
def mmr(indices, dense_scores, vecs_n, topn=6, lambda_=MMR_LAMBDA):
```

Call site (likely line ~492 in `answer_once()`):
```python
mmr_selected = mmr(selected, scores["dense"], topn=pack_top, lambda_=MMR_LAMBDA)
```

The call is missing the `vecs_n` argument! If the code runs, it will raise:
```
TypeError: mmr() missing 1 required positional argument: 'vecs_n'
```

**Fix**: Verify the actual call site (search the file for `mmr(`) and ensure it passes `vecs_n`:

```python
mmr_selected = mmr(selected, scores["dense"], vecs_n, topn=pack_top, lambda_=MMR_LAMBDA)
```

**Action**: Search codebase for all `mmr()` calls and verify they include `vecs_n` argument. Add type hints to `mmr()` signature.

---

### 6. HEADROOM ENFORCEMENT MISSING (CORRECTNESS) – MEDIUM

**Location**: Lines 41–60 (constant definition) vs. `pack_snippets()` (never used)

**Issue**:
```python
HEADROOM_FACTOR = 1.10  # Declared but never used
```

The hardening notes say:
> "Hard cap (--pack) + token budget (2800). Always include top-1 even if used + t_est > budget, then allow budget up to `budget_tokens * HEADROOM_FACTOR` for the remainder."

But searching `pack_snippets()` (lines 361–363), there is no enforcement of headroom:

```python
for idx in order:
    if len(ids) >= pack_top:
        break  # Hard exit on count, no headroom check
    ...
    if used + t_est > budget_tokens:
        continue  # Skip, no headroom logic
```

**Fix**: Apply headroom after collecting first snippet:

```python
def pack_snippets(chunks, selected, budget_tokens=CTX_TOKEN_BUDGET, pack_top=6):
    """Pack snippets with hard cap (pack_top) and token budget (with headroom for top-1)."""
    ids = []
    out = []
    used = 0

    headroom_budget = budget_tokens * HEADROOM_FACTOR  # Allow overage for top-1
    force_include_first = True  # Always include best match, even if over budget

    for rank, idx in enumerate(selected):
        if len(ids) >= pack_top:
            break  # Hard snippet count limit

        txt = chunks[idx]["text"]
        t_est = approx_tokens(len(txt))

        # First snippet gets headroom allowance
        if force_include_first:
            used += t_est
            force_include_first = False
            # ... add snippet
        else:
            # Subsequent snippets: enforce hard budget
            if used + t_est > budget_tokens:
                continue
            used += t_est
            # ... add snippet

    return out, ids
```

Add test (Test #14 below) to verify headroom behavior.

**Action**: Implement headroom enforcement in `pack_snippets()` and add unit test.

---

### 7. LOGGING INCONSISTENT (OBSERVABILITY) – MEDIUM

**Location**: Throughout (e.g., lines 303, 558–581, 410–417)

**Issue**: Code mixes:
- `logging.info()` (module logger)
- `print(..., file=sys.stderr)` (direct stderr)
- `log_event()` (structured JSON, but underused)

Example:
```python
logging.info(f"Ollama endpoint: {OLLAMA_URL}")  # Logger
print(f"  [{i + 1}/{len(texts)}]", file=sys.stderr)  # Direct stderr
print(f"ERROR: ...", file=sys.stderr)  # Direct stderr
```

This is inconsistent and makes log aggregation difficult.

**Fix**:
1. **Remove** `logging.basicConfig()` from module level (lines ~32–33 if present)
2. Move all config to `main()`:
   ```python
   def main():
       logging.basicConfig(
           level=logging.INFO,
           format="[%(asctime)s] %(levelname)s: %(message)s"
       )
   ```
3. Replace all `print(..., file=sys.stderr)` with `logger.error()`, `logger.info()`, etc.:
   ```python
   # Before:
   print(f"  [{i + 1}/{len(texts)}]", file=sys.stderr)

   # After:
   logger.info(f"Embedding chunk {i + 1}/{len(texts)}")
   ```
4. Use `log_event()` for structured metrics:
   ```python
   log_event("embedding_batch", chunk_id=cid, timestamp=time.time(), model=EMB_MODEL)
   ```

**Action**: Audit all print statements and migrate to logger. Run with `--debug` to verify output.

---

### 8. RETRY SEMANTICS UNSAFE (RELIABILITY) – MEDIUM

**Location**: Lines 103–151 (retry config), lines 561–565 (embed_texts POST)

**Issue**:
```python
# NOTE: We retry POSTs because our POST endpoints are idempotent...
retry_kwargs["allowed_methods"] = frozenset(["POST", "GET"])
```

But:
1. **Chat generation is NOT idempotent** – same input with `seed=42` should produce the same output, but if the LLM itself has randomness or if the request is retried mid-completion, you could be paying twice for the same answer.
2. **No rate-limit ceiling** – `Retry-After` is respected but capped at 60s, which could still cause excessive waiting.
3. **Max 2 retries** – still means 3 total attempts; with slow models, this adds up.

**Fix**:
Create an idempotent POST list and gate retries:

```python
# Idempotent POST endpoints (safe to retry)
IDEMPOTENT_PATHS = {"/api/embeddings"}

# Non-idempotent endpoints (never retry)
NON_IDEMPOTENT_PATHS = {"/api/chat"}  # Chat can produce different outputs per attempt

def is_idempotent_post(url: str, method: str) -> bool:
    """Check if POST is safe to retry."""
    if method != "POST":
        return True  # GET is idempotent
    from urllib.parse import urlparse
    path = urlparse(url).path
    return path in IDEMPOTENT_PATHS

def _mount_retries(sess, retries: int):
    # ... existing code ...

    # Only allow POST retries for idempotent paths
    retry_kwargs["allowed_methods"] = frozenset(["GET"])  # Always safe

    # Optionally add POST if we know it's safe
    # (For now, restrict to embedding only)
    if "EMB" in os.environ.get("RETRY_POST_PATHS", "EMB"):
        retry_kwargs["allowed_methods"] = frozenset(["POST", "GET"])
```

Also add overhead cap:
```python
# Respect Retry-After but cap at 60s (prevent abuse)
retry_kwargs["respect_retry_after_header"] = True
# Note: urllib3 doesn't have built-in ceiling, so log if we hit 60s
```

**Action**: Document which endpoints are idempotent, restrict POST retries to embedding only, add observability for high-retry scenarios.

---

### 9. ATOMIC WRITE GAPS (DURABILITY) – MEDIUM

**Location**: Search for all `.jsonl` writes (chunks.jsonl, meta.jsonl, bm25.json)

**Issue**: Code has `atomic_write_bytes()` and `atomic_save_npy()`, but not all artifacts use them:

- `chunks.jsonl` – written in `build_chunks()`, likely with direct `open()`
- `meta.jsonl` – written in `build()`, likely with direct `open()`
- `bm25.json` – written in `build()`, likely with direct `open()`
- `index.meta.json` – written with `_atomic_write_json()` ✓

**Fix**: Find all writes and wrap with atomic variants:

```python
# Before:
with open(FILES["bm25"], "w") as f:
    json.dump(bm, f)

# After:
bm_bytes = json.dumps(bm, ensure_ascii=False).encode("utf-8")
atomic_write_bytes(FILES["bm25"], bm_bytes)
```

Or create an `atomic_write_json()` helper:
```python
def atomic_write_json(path: str, obj) -> None:
    """Atomically write JSON object."""
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    atomic_write_bytes(path, data)
```

Then use it everywhere:
```python
atomic_write_json(FILES["bm25"], bm)
atomic_write_json(FILES["chunks"], chunks)  # If meta is also JSONL
```

**Action**: Audit all file writes in `build()`, wrap with atomic functions, verify with durability test (Test #15).

---

### 10. TIMEOUT PLUMBING INCOMPLETE (RELIABILITY) – MEDIUM

**Location**: Lines 52–57 (timeout constants) vs. actual `post()` calls (lines 561–565, 646–650, 769)

**Issue**:
```python
EMB_CONNECT_TIMEOUT = 3.0
EMB_READ_TIMEOUT = 120.0
CHAT_CONNECT_TIMEOUT = 3.0
CHAT_READ_TIMEOUT = 180.0
```

But:
1. **embed_query()** (line 649) uses hardcoded `60.0`:
   ```python
   timeout=(EMB_CONNECT_TIMEOUT, 60.0)  # Should be EMB_READ_TIMEOUT!
   ```
2. **Env var plumbing incomplete** – timeouts can be set via env, but no validation at startup
3. **No propagation to reranker** – `rerank_with_llm()` uses `CHAT_READ_TIMEOUT` but might not handle custom overrides

**Fix**:
1. Replace hardcoded timeouts with constants:
   ```python
   timeout=(EMB_CONNECT_TIMEOUT, EMB_READ_TIMEOUT)  # Use constant, not 60.0
   ```
2. Add validation at startup:
   ```python
   def validate_timeouts():
       """Validate and log timeout configuration."""
       logger.info(f"Timeouts: embed={EMB_CONNECT_TIMEOUT}s connect, {EMB_READ_TIMEOUT}s read")
       logger.info(f"Timeouts: chat={CHAT_CONNECT_TIMEOUT}s connect, {CHAT_READ_TIMEOUT}s read")
   ```
3. Call from `main()` before any I/O:
   ```python
   def main():
       validate_timeouts()
       validate_chunk_config()
       # ... rest of main
   ```

**Action**: Standardize all timeout usage, add startup validation, test with tiny timeouts (Test #11).

---

### 11. RTF GUARD TOO LOOSE (CORRECTNESS) – LOW

**Location**: Lines 427–430 in `strip_noise()`

**Issue**:
```python
is_rtf = (
    text.lstrip().startswith("{\\rtf") or
    len(re.findall(r"\\(?:cf\d+|u[+-]?\d+\?|f\d+|pard)\b", text)) > 10
)
```

The second check (backslash pattern match > 10) can false-trigger on normal text with many escape sequences (e.g., LaTeX, URLs with backslashes, code snippets). Example:
```
Text with \command1 and \another\path\like\structure and many backslashes...
```

**Fix**: Strengthen the guard:
```python
def is_likely_rtf(text: str) -> bool:
    """Check if content is likely RTF (not false-trigger on normal text)."""
    # Strong indicator: RTF magic number at start
    if text.lstrip().startswith("{\\rtf"):
        return True

    # Weak indicator: check first 500 chars for RTF-specific control words
    # (not just any backslash, but *specific* RTF commands)
    first_500 = text[:500].lower()
    rtf_keywords = re.findall(
        r"\\(?:rtf\d|par\b|colortbl|fonttbl|stylesheet|info|docfmt)",
        first_500
    )
    return len(rtf_keywords) > 2

is_rtf = is_likely_rtf(text)
```

Also add a test case with non-RTF text containing many backslashes to verify no false-positives (Test #12).

**Action**: Tighten RTF detection, add test for false-positives.

---

### 12. RERANK FALLBACK INVISIBLE (OBSERVABILITY) – LOW

**Location**: Lines 799–809 in `rerank_with_llm()`

**Issue**:
```python
except requests.exceptions.Timeout:
    # Fallback to MMR order
    return selected, rerank_scores
except requests.exceptions.ConnectionError:
    return selected, rerank_scores
```

When reranking times out or fails, the code silently falls back to MMR order. An operator monitoring this system has no way to know reranking is degraded.

**Fix**: Emit a single log line on fallback:
```python
except requests.exceptions.Timeout:
    log_event("rerank_timeout", timeout_sec=CHAT_READ_TIMEOUT, fallback="mmr")
    return selected, rerank_scores
except requests.exceptions.ConnectionError as e:
    log_event("rerank_connection_error", error=str(e), fallback="mmr")
    return selected, rerank_scores
except json.JSONDecodeError as e:
    log_event("rerank_parse_error", error=str(e), fallback="mmr")
    return selected, rerank_scores
```

**Action**: Add observability logs to rerank error paths.

---

### 13. SESSION PROXY BEHAVIOR UNDOCUMENTED (SECURITY) – LOW

**Location**: Lines 183–195 in `get_session()`

**Issue**:
```python
REQUESTS_SESSION = requests.Session()
```

By default, `requests.Session()` will **respect HTTP proxy environment variables** (`HTTP_PROXY`, `HTTPS_PROXY`, etc.). In a shared environment, this could accidentally route Ollama requests through an untrusted proxy, leaking Clockify KB content.

**Fix**: Explicitly disable proxy inheritance:
```python
REQUESTS_SESSION = requests.Session()
REQUESTS_SESSION.trust_env = False  # Ignore proxy env vars; Ollama is local
```

Or document the intended behavior:
```python
def get_session(retries=0):
    """Get or create global requests session with optional retry logic.

    Note: By default, respects HTTP proxy env vars. Set trust_env=False
    to ignore proxies (recommended for local Ollama).
    """
    global REQUESTS_SESSION, REQUESTS_SESSION_RETRIES
    if REQUESTS_SESSION is None:
        REQUESTS_SESSION = requests.Session()
        REQUESTS_SESSION.trust_env = False  # Disable proxy discovery
        if retries > 0:
            _mount_retries(REQUESTS_SESSION, retries)
        REQUESTS_SESSION_RETRIES = retries
    ...
```

**Action**: Set `trust_env=False` and add comment explaining why.

---

### 14. DTYPE CONSISTENCY WEAK (PERFORMANCE) – LOW

**Location**: Lines 552–582 (embed_texts), 652, 670, 707–728 (retrieve)

**Issue**:
```python
return np.array(vecs, dtype="float32")  # Returned as float32
vecs_n = np.load(FILES["emb"])  # But dtype on load depends on .npy file
```

If vectors were accidentally saved as `float64`, they'll load as `float64`, breaking normalization assumptions and wasting memory.

**Fix**: Enforce dtype consistently:

```python
def embed_texts(...):
    ...
    return np.array(vecs, dtype="float32")

def atomic_save_npy(arr: np.ndarray, path: str):
    """Atomically save numpy array, enforcing float32."""
    arr = arr.astype("float32")  # Ensure correct dtype before saving
    ... rest of function ...

def load_index():
    """Load index, enforce float32 throughout."""
    vecs_n = np.load(FILES["emb"]).astype("float32")  # Explicit cast on load
    ... rest of function ...
```

Also add a dtype assertion at critical points:
```python
assert vecs_n.dtype == np.float32, f"Expected float32, got {vecs_n.dtype}"
```

**Action**: Enforce float32 end-to-end, add assertions, add dtype test.

---

### 15. CONFIG VALIDATION TIMING (OBSERVABILITY) – LOW

**Location**: `main()` function

**Issue**: Config is validated in `validate_and_set_config()` and `validate_chunk_config()` (lines 297–326), but there's no summary printed at startup. Users don't see effective config until they ask or look at code.

**Fix**: Add a config summary in `main()`:

```python
def print_config():
    """Print effective configuration at startup."""
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  Ollama URL: {OLLAMA_URL}")
    logger.info(f"  Gen Model: {GEN_MODEL}")
    logger.info(f"  Emb Model: {EMB_MODEL}")
    logger.info(f"  Chunk size: {CHUNK_CHARS} chars, overlap: {CHUNK_OVERLAP}")
    logger.info(f"  Context budget: {CTX_TOKEN_BUDGET} tokens (~11,200 chars)")
    logger.info(f"  Retrieval: top-K={DEFAULT_TOP_K}, pack={DEFAULT_PACK_TOP}, threshold={DEFAULT_THRESHOLD}")
    logger.info(f"  MMR lambda: {MMR_LAMBDA} (relevance vs. diversity)")
    logger.info(f"  Timeouts: embed {EMB_CONNECT_TIMEOUT}s conn / {EMB_READ_TIMEOUT}s read")
    logger.info(f"             chat {CHAT_CONNECT_TIMEOUT}s conn / {CHAT_READ_TIMEOUT}s read")
    logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(...)
    # ... arg parsing ...

    validate_and_set_config(...)
    validate_chunk_config()
    print_config()  # ← Always print effective config

    # ... rest of main ...
```

**Action**: Add config summary function and call from `main()`.

---

## Test Plan (15 Tests)

### Test 1: Auth Header Stripping on Cross-Origin Redirect
```python
def test_auth_redirect_leak():
    """Verify Authorization header is dropped on cross-origin 30x."""
    # Mock a session with retry strategy
    sess = requests.Session()
    _mount_retries(sess, retries=1)

    # Simulate cross-origin redirect by crafting a mock response
    # Verify that Authorization header is not sent on redirect
    # Use a URL like http://localhost:8000 → http://evil.com
```

### Test 2: urllib3 Version Compatibility
```python
def test_urllib3_compat():
    """Verify retry config works with urllib3 v1 and v2."""
    # Test with allowed_methods (v2)
    # Test with method_whitelist (v1)
    # Ensure no TypeError
```

### Test 3: Windows Lock Stall Recovery
```python
def test_windows_lock_stale():
    """Verify stale lock (mtime > 10s) is removed even if PID check returns True."""
    # Write a stale lock file
    # Verify lock is removed after 10s (or on mtime check)
```

### Test 4: Determinism (Same Q → Same Hash)
```python
def test_determinism():
    """Run same question 3 times, verify identical hashes."""
    q = "How do I track time?"
    hashes = [hashlib.sha256(answer_once(q).encode()).hexdigest() for _ in range(3)]
    assert len(set(hashes)) == 1, f"Non-deterministic: {set(hashes)}"
```

### Test 5: Different Q → Different Hashes
```python
def test_determinism_different_q():
    """Two different questions should produce different hashes."""
    h1 = hashlib.sha256(answer_once("Q1").encode()).hexdigest()
    h2 = hashlib.sha256(answer_once("Q2").encode()).hexdigest()
    assert h1 != h2
```

### Test 6: MMR Signature
```python
def test_mmr_signature():
    """Verify mmr() function is called with all required args."""
    # Mock answer_once and verify mmr() call includes vecs_n
```

### Test 7: Headroom Enforcement
```python
def test_pack_headroom():
    """Verify top-1 is always included even if over budget."""
    # Create mock chunks, set budget to small value
    # Verify len(ids) >= 1 even if used > budget_tokens
    # Verify total tokens <= budget_tokens * HEADROOM_FACTOR
```

### Test 8: Idempotent POST Retries
```python
def test_retry_post_idempotent():
    """Verify POST retries only on idempotent endpoints."""
    # Mock an embed request (idempotent, should retry on 500)
    # Mock a chat request (non-idempotent, should NOT retry)
```

### Test 9: Atomic Writes (All Artifacts)
```python
def test_atomic_writes():
    """Verify all artifact writes use atomic functions."""
    # Check build() writes chunks, meta, bm25, index.meta atomically
    # Simulate crash mid-write and verify no partial files left
```

### Test 10: Timeout Propagation
```python
def test_timeout_tiny():
    """Set tiny timeouts and verify proper timeout errors."""
    os.environ["EMB_READ_TIMEOUT"] = "0.001"
    # Call embed_query and expect Timeout exception
    # Verify error message is helpful
```

### Test 11: RTF False Positive
```python
def test_rtf_false_positive():
    """Verify non-RTF text with backslashes is not stripped."""
    text = r"Path: C:\Users\Test and code: \command1 \command2 \command3..."
    result = strip_noise(text)
    assert result == text, "False positive RTF stripping"
```

### Test 12: Rerank Timeout Logging
```python
def test_rerank_timeout_logged():
    """Verify rerank timeout is logged and falls back gracefully."""
    # Mock rerank_with_llm to timeout
    # Verify log_event("rerank_timeout", ...) is called
    # Verify fallback to MMR works
```

### Test 13: Session Trust Env
```python
def test_session_no_proxy():
    """Verify session ignores HTTP_PROXY env var."""
    os.environ["HTTP_PROXY"] = "http://evil.proxy:8080"
    sess = get_session()
    assert sess.trust_env == False, "Session should ignore proxy env vars"
```

### Test 14: Dtype Consistency
```python
def test_dtype_float32():
    """Verify all vectors are float32 throughout."""
    build("knowledge_full.md")
    vecs = np.load(FILES["emb"])
    assert vecs.dtype == np.float32, f"Expected float32, got {vecs.dtype}"
```

### Test 15: Durability (Crash Recovery)
```python
def test_durability_crash():
    """Simulate crash during build and verify partial files aren't used."""
    # Mock a crash in embed_texts (after writing partial vecs.npy)
    # Verify on restart, build detects corruption and rebuilds
```

---

## Implementation Roadmap

### Phase 1 (Critical – Do First)
- [ ] Issue #1: Fix urllib3 compatibility (prevents crashes on old urllib3)
- [ ] Issue #5: Fix MMR signature (prevents runtime error)
- [ ] Issue #3: Add mtime staleness to lock (prevents Windows stall)

### Phase 2 (High Priority – Do Next)
- [ ] Issue #6: Implement headroom enforcement in `pack_snippets()`
- [ ] Issue #9: Use atomic writes for all artifacts (chunks, meta, bm25)
- [ ] Issue #7: Centralize logging (migrate all `print()` to `logger`)

### Phase 3 (Medium Priority)
- [ ] Issue #4: Add determinism test and verify reproducibility
- [ ] Issue #10: Standardize timeout usage and add startup validation
- [ ] Issue #12: Add observability to rerank fallback

### Phase 4 (Nice to Have)
- [ ] Issue #8: Document idempotent POST endpoints and gate retries
- [ ] Issue #11: Tighten RTF detection
- [ ] Issue #13: Set `trust_env=False` on session
- [ ] Issue #14: Enforce float32 end-to-end with assertions
- [ ] Issue #15: Add config summary at startup

---

## Checklist Before Production

- [ ] All 15 issues addressed
- [ ] All 15 tests passing
- [ ] No remaining `print()` calls in release code (use logger)
- [ ] Config printed at startup
- [ ] Determinism verified (same Q → same hash, 10 times)
- [ ] Timeouts tested (tiny timeout → proper error)
- [ ] Atomic writes tested (kill process mid-build → verify no corruption)
- [ ] Lock recovery tested (kill process holding lock → verify next process takes over)
- [ ] Auth redirect tested (cross-origin 30x → Authorization header dropped)
- [ ] RTF false-positive tested (backslash-heavy text → no stripping)
- [ ] Code review by security engineer
- [ ] Load test (100 concurrent queries → stable performance)

---

## Summary

**Critical issues**: 3 (urllib3 compat, MMR signature, Windows lock)
**High priority**: 3 (headroom, atomic writes, logging)
**Medium priority**: 3 (determinism, timeouts, rerank observability)
**Low priority**: 6 (auth redirect verify, retry semantics, RTF guard, proxy, dtype, config print)

**Estimated effort**: 40–60 hours
**Risk if not fixed**: Data corruption (atomic writes), runtime crashes (MMR, urllib3), operational blindness (logging), Windows deployments stuck (lock stall)

---

**Next step**: Pick Phase 1 and start with the three critical issues.
