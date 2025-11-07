# Next Claude Code Session Prompt

## Context

Continue implementing remaining improvements for the Clockify RAG CLI system. Previous session completed 22/30 improvements (73%). This session focuses on the remaining 8 moderate-to-complex improvements.

## Completed So Far

**Phase 1 (Quick Wins)**: 10/10 ✅ Complete
- Duplicate code removal, BM25 tuning, embedding cache, pytest, sanitization, EMB_DIM fix, MMR vectorization, logging, evaluation, rate limiting

**Phase 2 (High-Impact)**: 8/8 ✅ Complete
- God function refactor, exception handling, query expansion, query cache, sys.exit removal, mypy, CI/CD, pre-commit hooks

**Phase 3 (Advanced)**: 4/12 ✅ Partial
- Benchmark suite, few-shot prompting, import formatting, setup script

**Current State**:
- Branch: `claude/implement-30-rag-improvements-011CUrcTobd1TQBfKWCinpsr`
- Tests: 73 passing
- Type errors: 31 (down from 100+)
- Commits: 25

---

## Task: Implement Remaining 8 Improvements

### Priority 1: Moderate Complexity (1-2 weeks each)

#### Rank 22: FAISS IVFFlat Optimization for M1 Macs
**Goal**: Fix segfault on ARM64 macOS by using smaller nlist parameter

**Current State** (clockify_support_cli_final.py:220-258):
```python
# ARM64 macOS uses IndexFlatIP (linear scan) due to IVFFlat segfault
if is_macos_arm64:
    index = faiss.IndexFlatIP(dim)
```

**Implementation**:
1. Try IVFFlat with nlist=32 (instead of 256)
2. Train on small subset (1000 vectors)
3. Fallback to FlatIP if segfault occurs
4. Add logging for which index type succeeded
5. Test on M1/M2/M3 hardware if available

**Expected Gain**: 10-50x speedup on M1 for k-NN (1000ms → 20-100ms)

**Files to Modify**:
- `clockify_support_cli_final.py` (build_faiss_index function)

---

#### Rank 23: Sentence-Aware Chunking
**Goal**: Split chunks at sentence boundaries instead of mid-sentence

**Current State** (clockify_support_cli_final.py:791-811):
```python
def sliding_chunks(text: str, maxc=1600, overlap=200) -> list:
    # Currently splits at character boundaries
```

**Implementation**:
1. Add `nltk` to requirements.txt
2. Use `nltk.sent_tokenize()` for sentence splitting
3. Build chunks respecting sentence boundaries
4. Keep max_chars limit (1600)
5. Add overlap logic for sentence boundaries

**Expected Gain**: 5-10% retrieval quality improvement

**Files to Modify**:
- `clockify_support_cli_final.py` (sliding_chunks function)
- `requirements.txt` (add nltk)
- `tests/test_chunker.py` (add sentence boundary tests)

---

#### Rank 24: BM25 Early Termination (Wand Algorithm)
**Goal**: Optimize BM25 scoring to compute only top-k scores

**Current State** (clockify_support_cli_final.py:926-946):
```python
def bm25_scores(query: str, bm: dict) -> np.ndarray:
    # Computes scores for ALL documents
```

**Implementation**:
1. Research Wand/MaxScore algorithms
2. Implement early termination for top-k
3. Add unit tests for correctness
4. Benchmark performance improvement

**Expected Gain**: 2-3x speedup on large corpora (1000ms → 300ms for 100K docs)

**Files to Modify**:
- `clockify_support_cli_final.py` (bm25_scores function)
- `tests/test_bm25.py` (add early termination tests)

---

#### Rank 27: HTTP Connection Pooling Optimization
**Goal**: Explicitly set connection pool parameters for better concurrency

**Current State** (clockify_support_cli_final.py:147-161):
```python
def get_session(retries=0):
    global REQUESTS_SESSION
    if REQUESTS_SESSION is None:
        REQUESTS_SESSION = requests.Session()
```

**Implementation**:
1. Use `HTTPAdapter` with explicit pool_connections and pool_maxsize
2. Set pool_connections=10, pool_maxsize=20
3. Mount adapter for http:// and https://
4. Test concurrent query performance

**Expected Gain**: 10-20% latency reduction on concurrent queries

**Files to Modify**:
- `clockify_support_cli_final.py` (get_session function)

---

#### Rank 28: LLM Confidence Scoring
**Goal**: Add confidence field (0-100) to LLM output

**Current State** (clockify_support_cli_final.py:596-602):
```python
USER_WRAPPER = """SNIPPETS:
{snips}
QUESTION:
{q}
Answer with citations like [id1, id2]."""
```

**Implementation**:
1. Update USER_WRAPPER to request JSON output with confidence
2. Parse JSON response (handle parse errors)
3. Add confidence to metadata
4. Add confidence threshold filtering (e.g., reject < 50)
5. Update tests to handle new format

**Expected Gain**: Better user trust calibration, enable confidence-based filtering

**Files to Modify**:
- `clockify_support_cli_final.py` (USER_WRAPPER, ask_llm, answer_once)
- `tests/test_retriever.py` (test confidence parsing)

---

#### Rank 29: Profiling Support (--profile flag)
**Goal**: Add built-in performance profiling

**Current State**: No profiling support

**Implementation**:
1. Add `--profile` flag to argument parser
2. Wrap main execution with `cProfile.Profile()`
3. Output stats to stdout or file
4. Sort by cumulative time
5. Show top 20 functions

**Expected Gain**: Faster performance debugging, identify optimization targets

**Files to Modify**:
- `clockify_support_cli_final.py` (main function, argument parser)

---

### Priority 2: High Complexity (2-4 weeks each)

#### Rank 28: Modularize into Packages
**Goal**: Split monolithic file into logical packages

**Current State**: Single 2600+ line file

**Implementation**:
1. Create package structure:
   ```
   clockify_rag/
   ├── __init__.py
   ├── chunking.py
   ├── embedding.py
   ├── retrieval.py
   ├── reranking.py
   ├── packing.py
   ├── llm.py
   └── utils.py
   ```
2. Move functions to appropriate modules
3. Update imports throughout
4. Maintain backwards compatibility
5. Update tests to import from new structure

**Expected Gain**: Better maintainability, clearer code organization

**Files to Create**: 7+ new Python modules

---

#### Rank 29: Add Plugin Architecture
**Goal**: Enable extensibility via plugin system

**Current State**: Monolithic architecture

**Implementation**:
1. Define plugin interfaces (e.g., RetrieverPlugin, RerankPlugin)
2. Use entry points or importlib for plugin discovery
3. Add plugin registration system
4. Create example plugins (custom retriever, reranker)
5. Document plugin API

**Expected Gain**: Extensibility, community contributions, custom retrievers

**Files to Create**: Plugin framework + examples

---

## Success Criteria

For this session, aim to complete at least 3-4 moderate complexity improvements:
- [ ] FAISS M1 optimization (Rank 22)
- [ ] Sentence-aware chunking (Rank 23)
- [ ] HTTP connection pooling (Rank 27)
- [ ] Profiling support (Rank 29)

**Stretch Goals**:
- [ ] BM25 early termination (Rank 24)
- [ ] LLM confidence scoring (Rank 28)

**Leave for Future**:
- Modularization (Rank 28) - requires major refactoring
- Plugin architecture (Rank 29) - requires architecture redesign

---

## Testing Requirements

For each improvement:
1. **Add unit tests** - Maintain 100% test pass rate
2. **Run benchmarks** - `make benchmark-quick` before/after
3. **Type check** - `make typecheck` (accept current 31 errors)
4. **Integration test** - `make chat` and test manually

---

## Commit Strategy

Use clear commit messages following the pattern:
```
feat: add <improvement> (Rank <number>)

- Bullet point of what changed
- Expected gain: X% improvement
- All N tests passing
```

---

## Files Reference

**Main Implementation**: `clockify_support_cli_final.py` (2600+ lines)
**Tests**: `tests/test_*.py` (73 tests)
**Config**: `pyproject.toml`, `.pre-commit-config.yaml`
**CI/CD**: `.github/workflows/*.yml`
**Docs**: `CLAUDE.md`, `PR_SUMMARY.md`

---

## Important Notes

1. **Knowledge Base Question**: The system does NOT recrawl URLs. It reads from a static `knowledge_full.md` file that must be provided/updated manually. To rebuild: `make build`

2. **Type Errors**: Currently 31 type errors remaining (down from 100+). This is acceptable. Don't spend time on complete type safety.

3. **Branch**: Continue on `claude/implement-30-rag-improvements-011CUrcTobd1TQBfKWCinpsr`

4. **Don't Break Tests**: All 73 tests must pass after each change

5. **Performance Baseline**: Run `make benchmark-quick` before starting to establish baseline

---

## Prompt to Use

```
Continue implementing RAG improvements for the Clockify CLI. Previous session completed 22/30 improvements (73%).

Focus on these moderate complexity improvements:
1. Rank 22: FAISS IVFFlat for M1 Macs (fix segfault with nlist=32)
2. Rank 23: Sentence-aware chunking (use NLTK)
3. Rank 27: HTTP connection pooling (explicit HTTPAdapter params)
4. Rank 29: Add --profile flag (cProfile integration)

Current state:
- Branch: claude/implement-30-rag-improvements-011CUrcTobd1TQBfKWCinpsr
- Tests: 73 passing (maintain 100%)
- File: clockify_support_cli_final.py (2600+ lines)

See NEXT_SESSION_PROMPT.md for full context and implementation details.

Start with Rank 22 (FAISS M1 optimization) as it has clear implementation steps.
```

---

## Additional Context Files

Read these files for context:
- `PR_SUMMARY.md` - Summary of completed work
- `IMPROVEMENTS.jsonl` - Full list of 30 improvements with details
- `CLAUDE.md` - Project overview and architecture
- `clockify_support_cli_final.py` - Main implementation

---

**Good luck!** 🚀
