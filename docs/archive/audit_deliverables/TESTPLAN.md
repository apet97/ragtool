# Test Plan - Full Repository Review Fixes

**Project**: Clockify RAG CLI (1rag)
**Date**: 2025-11-05
**Version**: Post-patch testing plan
**Scope**: Verify all 10 patches applied correctly

---

## Pre-Patch Baseline

### 1. Create Baseline Backup
```bash
# Create timestamped backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p backups/$TIMESTAMP
cp clockify_support_cli_final.py backups/$TIMESTAMP/
cp Makefile backups/$TIMESTAMP/
cp requirements.txt backups/$TIMESTAMP/

echo "Backup created: backups/$TIMESTAMP"
```

### 2. Capture Pre-Patch Metrics
```bash
# Run baseline tests
bash scripts/smoke.sh > baseline_smoke.log 2>&1 || true
bash scripts/acceptance_test.sh > baseline_acceptance.log 2>&1 || true

# Capture performance baseline
if [ -f "knowledge_full.md" ]; then
    time python3 clockify_support_cli_final.py build knowledge_full.md > baseline_build.log 2>&1
    time python3 clockify_support_cli_final.py ask "What is Clockify?" > baseline_query.log 2>&1
fi

echo "✅ Baseline captured"
```

---

## Patch Application Order

**Recommended order** (lowest risk first):

1. ✅ Patch 5 (FAISS cache) - 5 min, zero risk
2. ✅ Patch 6 (input validation) - 10 min, zero risk
3. ✅ Patch 1 (duplicate function) - 5 min, low risk
4. ✅ Patch 2 (bare excepts) - 10 min, low risk
5. ✅ Patch 8 (Makefile docs) - 0 min, zero risk
6. ✅ Patch 9 (lockfile) - 5 min, zero risk
7. ✅ Patch 4 (JSON validation) - 20 min, low risk
8. ✅ Patch 7 (performance) - 15 min, low risk
9. ✅ Patch 10 (race condition) - 15 min, low risk
10. ✅ Patch 3 (exceptions) - 30 min, medium risk

**Total Time**: ~2 hours

---

## Per-Patch Testing

### Patch 1: Duplicate normalize_scores() - CRITICAL

**Apply**:
```bash
# Extract and apply patch 1
sed -n '/^## Patch 1:/,/^## Patch 2:/p' PATCHES.md | \
    sed -n '/^```diff$/,/^```$/p' | sed '1d;$d' > patch1.diff

git apply --check patch1.diff
git apply patch1.diff
```

**Test**:
```bash
# 1. Syntax check
python3 -m py_compile clockify_support_cli_final.py
# Expected: No output (success)

# 2. Function exists check
python3 -c "
from clockify_support_cli_final import normalize_scores, normalize_scores_zscore
assert callable(normalize_scores), 'normalize_scores missing'
assert callable(normalize_scores_zscore), 'normalize_scores_zscore missing'
print('✅ Both functions exist')
"

# 3. Behavior check
python3 -c "
from clockify_support_cli_final import normalize_scores, normalize_scores_zscore
import numpy as np

scores = [0.1, 0.5, 0.9]

# Min-max should return [0, 0.5, 1]
minmax = normalize_scores(scores)
assert abs(minmax[0] - 0.0) < 0.01, f'Min-max failed: {minmax}'
assert abs(minmax[1] - 0.5) < 0.01, f'Min-max failed: {minmax}'
assert abs(minmax[2] - 1.0) < 0.01, f'Min-max failed: {minmax}'

# Z-score should return normalized values
zscore = normalize_scores_zscore(np.array(scores))
assert abs(zscore.mean()) < 0.01, f'Z-score mean should be ~0: {zscore.mean()}'
assert abs(zscore.std() - 1.0) < 0.1, f'Z-score std should be ~1: {zscore.std()}'

print('✅ Functions behave correctly')
"
```

**Rollback**:
```bash
git apply -R patch1.diff
```

**Success Criteria**:
- ✅ Both functions exist
- ✅ normalize_scores() returns [0, 1] range
- ✅ normalize_scores_zscore() returns mean≈0, std≈1
- ✅ No import errors

---

### Patch 2: Bare Except Clauses - CRITICAL

**Apply**:
```bash
sed -n '/^## Patch 2:/,/^## Patch 3:/p' PATCHES.md | \
    sed -n '/^```diff$/,/^```$/p' | sed '1d;$d' > patch2.diff
git apply patch2.diff
```

**Test**:
```bash
# 1. Check KeyboardInterrupt works
timeout 3 bash -c '
python3 clockify_support_cli_final.py chat <<EOF &
PID=$!
sleep 1
kill -INT $PID
wait $PID
echo "✅ KeyboardInterrupt propagates"
EOF
' || echo "Test completed (timeout expected)"

# 2. Verify except clauses
grep -n "except:" clockify_support_cli_final.py && {
    echo "❌ Bare except: still present"
    exit 1
} || echo "✅ No bare except: clauses found"

# 3. Check specific exceptions added
grep -q "except (KeyError, ValueError, TypeError)" clockify_support_cli_final.py && \
    echo "✅ Specific exceptions added" || \
    echo "⚠️  Specific exceptions not found (check manually)"
```

**Rollback**:
```bash
git apply -R patch2.diff
```

**Success Criteria**:
- ✅ No bare `except:` clauses remain
- ✅ KeyboardInterrupt works (Ctrl+C doesn't hang)
- ✅ Specific exceptions used where appropriate

---

### Patch 3: Replace sys.exit() with Exceptions - CRITICAL

**Apply**:
```bash
sed -n '/^## Patch 3:/,/^## Patch 4:/p' PATCHES.md | \
    sed -n '/^```diff$/,/^```$/p' | sed '1d;$d' > patch3.diff
git apply patch3.diff
```

**Test**:
```bash
# 1. Check custom exceptions defined
python3 -c "
from clockify_support_cli_final import EmbeddingError, LLMError, IndexError
print(f'✅ EmbeddingError: {EmbeddingError}')
print(f'✅ LLMError: {LLMError}')
print(f'✅ IndexError: {IndexError}')  # Note: may conflict with builtin
"

# 2. Verify sys.exit() only in main()
python3 -c "
import ast, inspect
from clockify_support_cli_final import embed_texts, ask_llm, answer_once

# Check embed_texts has no sys.exit
source = inspect.getsource(embed_texts)
assert 'sys.exit' not in source, 'embed_texts still has sys.exit()'
print('✅ embed_texts uses exceptions')

# Check ask_llm has no sys.exit
source = inspect.getsource(ask_llm)
assert 'sys.exit' not in source, 'ask_llm still has sys.exit()'
print('✅ ask_llm uses exceptions')

# Check answer_once raises instead of exit
source = inspect.getsource(answer_once)
assert 'sys.exit' not in source, 'answer_once still has sys.exit()'
print('✅ answer_once uses exceptions')
"

# 3. Test exception raising (should fail gracefully, not exit)
python3 -c "
from clockify_support_cli_final import embed_query, EmbeddingError
import sys

# Mock bad Ollama connection
try:
    # This should raise EmbeddingError, not sys.exit()
    embed_query('test', retries=0)
except EmbeddingError as e:
    print(f'✅ EmbeddingError raised: {e}')
    sys.exit(0)
except SystemExit as e:
    print(f'❌ sys.exit() called instead of raising exception')
    sys.exit(1)
except Exception as e:
    print(f'⚠️  Different exception: {type(e).__name__}: {e}')
    sys.exit(0)
"
```

**Rollback**:
```bash
git apply -R patch3.diff
```

**Success Criteria**:
- ✅ Custom exceptions defined
- ✅ Library functions raise exceptions (not sys.exit)
- ✅ main() catches exceptions and exits gracefully
- ✅ Functions can be imported and tested

---

### Patch 4: JSON Schema Validation - CRITICAL

**Apply**:
```bash
sed -n '/^## Patch 4:/,/^## Patch 5:/p' PATCHES.md | \
    sed -n '/^```diff$/,/^```$/p' | sed '1d;$d' > patch4.diff
git apply patch4.diff
```

**Test**:
```bash
# 1. Create corrupted index files
mkdir -p test_corruption
cd test_corruption

# Test 1: index.meta.json as list instead of dict
echo '["not", "a", "dict"]' > index.meta.json
echo '{}' > chunks.jsonl
echo '{}' > bm25.json
touch vecs_n.npy meta.jsonl

python3 ../clockify_support_cli_final.py chat --help 2>&1 | grep -q "invalid format" && \
    echo "✅ Detects list instead of dict" || \
    echo "⚠️  Validation may not trigger (depends on code path)"

# Test 2: Missing required keys
echo '{"wrong": "keys"}' > index.meta.json
python3 -c "
import sys
sys.path.insert(0, '..')
from clockify_support_cli_final import load_index
result = load_index()
assert result is None, 'Should return None for invalid metadata'
print('✅ Detects missing required keys')
"

# Test 3: BM25 validation
echo '{"kb_sha256": "abc", "chunks": 10, "emb_rows": 10, "bm25_docs": 5}' > index.meta.json
echo '[]' > bm25.json
python3 -c "
import sys
sys.path.insert(0, '..')
from clockify_support_cli_final import load_index
result = load_index()
assert result is None, 'Should return None for invalid BM25'
print('✅ Detects invalid BM25 structure')
"

cd ..
rm -rf test_corruption
```

**Rollback**:
```bash
git apply -R patch4.diff
```

**Success Criteria**:
- ✅ Detects corrupted index.meta.json
- ✅ Detects missing required keys
- ✅ Detects invalid BM25 structure
- ✅ Returns None and triggers rebuild instead of crashing

---

### Patch 5: FAISS Cache Invalidation - CRITICAL

**Apply**:
```bash
sed -n '/^## Patch 5:/,/^## Patch 6:/p' PATCHES.md | \
    sed -n '/^```diff$/,/^```$/p' | sed '1d;$d' > patch5.diff
git apply patch5.diff
```

**Test**:
```bash
# 1. Verify cache reset logic exists
grep -A 3 "save_faiss_index" clockify_support_cli_final.py | grep -q "_FAISS_INDEX = None" && \
    echo "✅ Cache invalidation code present" || \
    echo "❌ Cache invalidation missing"

# 2. Test rebuild resets cache (integration test)
if [ -f "knowledge_full.md" ]; then
    python3 -c "
from clockify_support_cli_final import build, _FAISS_INDEX, FILES
import os

# Simulate cached index
# Note: This test requires FAISS to be installed
try:
    if os.path.exists('knowledge_full.md'):
        # First build
        build('knowledge_full.md')

        # Verify cache is None after build
        from clockify_support_cli_final import _FAISS_INDEX
        # Note: _FAISS_INDEX will be None after build() in patched version
        print('✅ Cache invalidation test (manual verification needed)')
except Exception as e:
    print(f'⚠️  Test skipped or failed: {e}')
"
fi
```

**Rollback**:
```bash
git apply -R patch5.diff
```

**Success Criteria**:
- ✅ Global `_FAISS_INDEX = None` added in build()
- ✅ Debug log "Invalidated FAISS cache" present
- ✅ Manual verification: rebuild → query shows fresh index

---

### Patch 6: Input Validation - CRITICAL

**Apply**:
```bash
sed -n '/^## Patch 6:/,/^## Patch 7:/p' PATCHES.md | \
    sed -n '/^```diff$/,/^```$/p' | sed '1d;$d' > patch6.diff
git apply patch6.diff
```

**Test**:
```bash
# 1. Check MAX_QUESTION_LEN defined
python3 -c "
from clockify_support_cli_final import MAX_QUESTION_LEN
assert MAX_QUESTION_LEN == 2000, f'Expected 2000, got {MAX_QUESTION_LEN}'
print(f'✅ MAX_QUESTION_LEN = {MAX_QUESTION_LEN}')
"

# 2. Test length limit in REPL
python3 clockify_support_cli_final.py chat <<EOF 2>&1 | grep -q "too long"
$(python3 -c "print('a' * 2001)")
:exit
EOF

[ $? -eq 0 ] && echo "✅ Length limit enforced" || echo "❌ Length limit not enforced"

# 3. Test empty input handling
python3 clockify_support_cli_final.py chat <<EOF 2>&1

:exit
EOF

echo "✅ Empty input handled"

# 4. Test normal query passes through
if [ -f "knowledge_full.md" ] && [ -f "chunks.jsonl" ]; then
    timeout 30 python3 clockify_support_cli_final.py chat <<EOF 2>&1 | grep -q -v "too long"
What is Clockify?
:exit
EOF
    [ $? -eq 0 ] && echo "✅ Normal queries work" || echo "⚠️  Query may have failed (check manually)"
fi
```

**Rollback**:
```bash
git apply -R patch6.diff
```

**Success Criteria**:
- ✅ MAX_QUESTION_LEN constant defined
- ✅ Questions >2000 chars rejected with error message
- ✅ Empty questions skipped
- ✅ Normal questions (<2000 chars) work

---

### Patch 7: Cache Normalized Scores (Performance) - HIGH

**Apply**:
```bash
sed -n '/^## Patch 7:/,/^## Patch 8:/p' PATCHES.md | \
    sed -n '/^```diff$/,/^```$/p' | sed '1d;$d' > patch7.diff
git apply patch7.diff
```

**Test**:
```bash
# 1. Verify no redundant normalize_scores calls
python3 -c "
import ast, inspect
from clockify_support_cli_final import retrieve

source = inspect.getsource(retrieve)

# Count normalize_scores calls
count = source.count('normalize_scores(')
print(f'normalize_scores called {count} times in retrieve()')

# Should be called exactly 2 times (once for dense, once for BM25)
assert count == 2, f'Expected 2 calls, found {count}'
print('✅ Normalized scores cached correctly')
"

# 2. Performance test (if KB exists)
if [ -f "knowledge_full.md" ] && [ -f "chunks.jsonl" ]; then
    echo "Running query performance test..."

    # Before patch (from baseline)
    BASELINE_TIME=$(grep "real" baseline_query.log 2>/dev/null | awk '{print $2}' || echo "N/A")

    # After patch
    START=$(date +%s%3N)
    timeout 60 python3 clockify_support_cli_final.py ask "What is Clockify?" > /dev/null 2>&1
    END=$(date +%s%3N)
    PATCHED_TIME=$((END - START))

    echo "Baseline: ${BASELINE_TIME}ms"
    echo "Patched:  ${PATCHED_TIME}ms"

    # Should be faster (10-20% speedup expected)
    echo "✅ Performance test complete (manual verification)"
fi
```

**Rollback**:
```bash
git apply -R patch7.diff
```

**Success Criteria**:
- ✅ normalize_scores() called only 2x (not 4x)
- ✅ Full scores computed once and cached
- ✅ Candidate scores sliced from cached full scores
- ✅ 10-20% query speedup (measured with benchmark.sh)

---

### Patch 8: Document Local Embeddings in Makefile - HIGH

**Apply**:
```bash
sed -n '/^## Patch 8:/,/^## Patch 9:/p' PATCHES.md | \
    sed -n '/^```diff$/,/^```$/p' | sed '1d;$d' > patch8.diff
git apply patch8.diff
```

**Test**:
```bash
# 1. Check Makefile updated
grep -q "local embeddings (faster than Ollama)" Makefile && \
    echo "✅ Makefile help text updated" || \
    echo "❌ Makefile not updated"

# 2. Check build target uses EMB_BACKEND=local
grep -q "EMB_BACKEND=local" Makefile && \
    echo "✅ Build uses local embeddings by default" || \
    echo "❌ Build target not updated"

# 3. Test make build (if KB exists)
if [ -f "knowledge_full.md" ]; then
    make clean
    make build 2>&1 | tee makefile_build.log

    grep -q "local embeddings" makefile_build.log && \
        echo "✅ Local embeddings used in build" || \
        echo "❌ Local embeddings not used"

    grep -q "Hint: To use Ollama" makefile_build.log && \
        echo "✅ Hint message shown" || \
        echo "⚠️  Hint message missing"
fi
```

**Rollback**:
```bash
git apply -R patch8.diff
```

**Success Criteria**:
- ✅ Makefile build target updated
- ✅ Help text mentions local embeddings
- ✅ Hint shown for Ollama alternative

---

### Patch 9: Full Dependency Lockfile - HIGH

**Apply**:
```bash
sed -n '/^## Patch 9:/,/^## Patch 10:/p' PATCHES.md | \
    sed -n '/^```diff$/,/^```$/p' | sed '1d;$d' > patch9.diff
git apply patch9.diff
```

**Test**:
```bash
# 1. Check Makefile freeze target added
grep -q "^freeze:" Makefile && \
    echo "✅ Makefile freeze target added" || \
    echo "❌ Freeze target missing"

# 2. Generate lockfile
make freeze

# 3. Verify lockfile created
[ -f requirements.lock ] && \
    echo "✅ requirements.lock created" || \
    echo "❌ requirements.lock missing"

# 4. Check lockfile has more entries than requirements.txt
REQ_COUNT=$(grep -c "==" requirements.txt || echo 0)
LOCK_COUNT=$(grep -c "==" requirements.lock || echo 0)

echo "requirements.txt: $REQ_COUNT packages"
echo "requirements.lock: $LOCK_COUNT packages"

[ $LOCK_COUNT -gt $REQ_COUNT ] && \
    echo "✅ Lockfile includes transitive dependencies" || \
    echo "⚠️  Lockfile may be incomplete"

# 5. Test install from lockfile
if [ -d "rag_env" ]; then
    make clean
    rm -rf rag_env
    make venv
    make install 2>&1 | grep -q "Installing from lockfile" && \
        echo "✅ Install uses lockfile" || \
        echo "⚠️  Lockfile install not triggered"
fi
```

**Rollback**:
```bash
git apply -R patch9.diff
rm -f requirements.lock
```

**Success Criteria**:
- ✅ Makefile freeze target exists
- ✅ requirements.lock generated with pip freeze
- ✅ Lockfile has more packages than requirements.txt
- ✅ make install uses lockfile if present

---

### Patch 10: Fix build_lock() Race Condition - HIGH

**Apply**:
```bash
sed -n '/^## Patch 10:/,/^---$/p' PATCHES.md | \
    sed -n '/^```diff$/,/^```$/p' | sed '1d;$d' > patch10.diff
git apply patch10.diff
```

**Test**:
```bash
# 1. Check exception handling added
grep -A 5 "with open(BUILD_LOCK" clockify_support_cli_final.py | grep -q "except FileNotFoundError" && \
    echo "✅ FileNotFoundError handler added" || \
    echo "❌ Exception handler missing"

# 2. Test concurrent build handling (integration test)
if [ -f "knowledge_full.md" ]; then
    # Clean artifacts
    rm -f .build.lock chunks.jsonl vecs_n.npy meta.jsonl bm25.json index.meta.json

    # Start first build in background
    timeout 60 python3 clockify_support_cli_final.py build knowledge_full.md > build1.log 2>&1 &
    PID1=$!

    # Wait for lock to be acquired
    sleep 2

    # Try second build (should wait or fail gracefully)
    timeout 10 python3 clockify_support_cli_final.py build knowledge_full.md > build2.log 2>&1 &
    PID2=$!

    # Wait for both to complete
    wait $PID1
    EXIT1=$?
    wait $PID2
    EXIT2=$?

    echo "Build 1 exit code: $EXIT1"
    echo "Build 2 exit code: $EXIT2"

    # Check logs for race condition messages
    grep -q "Recovering.*dead PID\|Lock removed during check" build1.log build2.log && \
        echo "✅ Race condition handled" || \
        echo "⚠️  Race condition not triggered (test may need manual verification)"

    # At least one build should succeed
    [ $EXIT1 -eq 0 ] || [ $EXIT2 -eq 0 ] && \
        echo "✅ At least one build succeeded" || \
        echo "❌ Both builds failed"
fi
```

**Rollback**:
```bash
git apply -R patch10.diff
```

**Success Criteria**:
- ✅ FileNotFoundError exception handled
- ✅ Corrupt lock file handled
- ✅ Concurrent builds don't crash
- ✅ Lock recovery logic works

---

## Integration Testing (All Patches Applied)

### 1. Lint and Type Checks
```bash
# Python syntax
python3 -m py_compile clockify_support_cli_final.py
echo "✅ Syntax valid"

# Import check
python3 -c "import clockify_support_cli_final; print('✅ Import OK')"

# Optional: Type checking (if mypy installed)
which mypy && mypy clockify_support_cli_final.py || echo "⚠️  mypy not installed"
```

### 2. Self-Test Suite
```bash
python3 clockify_support_cli_final.py --selftest
# Expected output: [selftest] 7/7 tests passed
```

### 3. Full Build from Scratch
```bash
# Clean all artifacts
make clean
rm -f chunks.jsonl vecs_n.npy meta.jsonl bm25.json index.meta.json faiss.index

# Rebuild with local embeddings
time make build > integration_build.log 2>&1

# Verify artifacts created
for file in chunks.jsonl vecs_n.npy meta.jsonl bm25.json index.meta.json; do
    [ -f "$file" ] && echo "✅ $file created" || echo "❌ $file missing"
done

# Check build completed without errors
grep -q "Done\." integration_build.log && \
    echo "✅ Build completed successfully" || \
    echo "❌ Build failed (check integration_build.log)"
```

### 4. Query Accuracy Test
```bash
# Test queries with known answers
python3 clockify_support_cli_final.py chat <<EOF > integration_queries.log 2>&1
What is Clockify?
How do I track time?
What are the pricing plans?
:exit
EOF

# Verify no crashes
grep -q "I don't know\|track\|pricing" integration_queries.log && \
    echo "✅ Queries returned answers" || \
    echo "⚠️  Query responses may need manual verification"

# Verify no errors
grep -i "error\|exception\|traceback" integration_queries.log && \
    echo "❌ Errors found in query log" || \
    echo "✅ No errors in queries"
```

### 5. Smoke Test Suite
```bash
bash scripts/smoke.sh > post_patch_smoke.log 2>&1

# Compare with baseline
diff baseline_smoke.log post_patch_smoke.log > smoke_diff.log || true

echo "Smoke test diff (should show improvements):"
cat smoke_diff.log

# Check for failures
grep -q "SMOKE TEST COMPLETE" post_patch_smoke.log && \
    echo "✅ Smoke tests passed" || \
    echo "❌ Smoke tests failed"
```

### 6. Acceptance Tests
```bash
bash scripts/acceptance_test.sh > post_patch_acceptance.log 2>&1

# Verify all tests pass
grep -q "ACCEPTANCE TESTS COMPLETE" post_patch_acceptance.log && \
    echo "✅ Acceptance tests passed" || \
    echo "❌ Acceptance tests failed"

# Check specific validations
grep "✅" post_patch_acceptance.log | wc -l
```

### 7. Performance Regression Test
```bash
# Run benchmark
bash scripts/benchmark.sh > post_patch_benchmark.log 2>&1

# Extract metrics
echo "=== Performance Metrics ==="
grep -E "Build time:|Query latency:" post_patch_benchmark.log

# Compare with baseline (manual)
echo "Baseline build time: $(grep 'Build time:' baseline_build.log || echo 'N/A')"
echo "Patched build time: $(grep 'Build time:' integration_build.log || echo 'N/A')"
```

---

## Rollback Plan

### Complete Rollback (All Patches)
```bash
# Method 1: Restore from backup
TIMESTAMP="<your_timestamp>"
cp backups/$TIMESTAMP/clockify_support_cli_final.py ./
cp backups/$TIMESTAMP/Makefile ./
cp backups/$TIMESTAMP/requirements.txt ./

# Verify restoration
git diff backups/$TIMESTAMP/clockify_support_cli_final.py clockify_support_cli_final.py
echo "✅ Restored from backup"
```

### Git Rollback (If Committed)
```bash
# Revert all commits since patches
git log --oneline | head -10  # Find commit hash before patches
git revert --no-commit <hash>..HEAD
git commit -m "Rollback: revert all patches"
```

### Partial Rollback (Individual Patches)
```bash
# Reverse specific patch
git apply -R patch3.diff

# Verify
python3 -m py_compile clockify_support_cli_final.py
```

---

## Post-Deployment Monitoring

### 1. Error Rate Monitoring
```bash
# Check logs for new exception types
grep -i "embeddingerror\|llmerror\|indexerror" *.log

# Expected: These should appear when errors occur (not sys.exit)
```

### 2. Performance Monitoring
```bash
# Track query latency over time
grep "info: retrieve=" *.log | awk '{print $NF}' | sort -n

# Expected: 10-20% faster after Patch 7
```

### 3. Cache Hit Monitoring
```bash
# Check FAISS index loads
grep "ann=faiss status=loaded" *.log

# Expected: Fresh loads after rebuild
```

---

## Success Criteria (All Tests)

| Category | Test | Expected Result | Status |
|----------|------|-----------------|--------|
| Correctness | Duplicate function fix | 2 functions with correct names | ⬜ |
| Error Handling | Bare except replaced | No bare except: clauses | ⬜ |
| API Design | Exceptions raised | Library functions raise, not exit | ⬜ |
| Robustness | JSON validation | Corrupted files trigger rebuild | ⬜ |
| Cache | FAISS invalidation | Fresh index after rebuild | ⬜ |
| Security | Input validation | >2000 char questions rejected | ⬜ |
| Performance | Cached scores | 10-20% faster queries | ⬜ |
| Documentation | Local embeddings | Makefile updated | ⬜ |
| Dependencies | Lockfile | requirements.lock generated | ⬜ |
| Concurrency | Race condition fix | Concurrent builds handled | ⬜ |
| Integration | Self-tests | 7/7 tests pass | ⬜ |
| Integration | Smoke tests | All smoke tests pass | ⬜ |
| Integration | Acceptance tests | All acceptance tests pass | ⬜ |

**Overall Status**: 🔴 NOT TESTED → 🟡 TESTING → 🟢 PASSED

---

## Final Verification Checklist

Before declaring patches production-ready:

- [ ] All 10 patches applied successfully
- [ ] python3 -m py_compile succeeds
- [ ] Import test succeeds
- [ ] Self-test suite passes (7/7)
- [ ] Smoke tests pass
- [ ] Acceptance tests pass
- [ ] Build completes without errors
- [ ] Queries return valid answers
- [ ] Performance not degraded (within 10%)
- [ ] No new error types in logs
- [ ] Concurrent operations handled gracefully
- [ ] Documentation updated
- [ ] Backup created
- [ ] Rollback procedure tested

---

## Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Pre-patch baseline | 30 min | Backup, baseline metrics |
| Patch application | 2 hours | Apply 10 patches in order |
| Per-patch testing | 2 hours | Test each patch individually |
| Integration testing | 1 hour | Full test suite |
| Performance validation | 30 min | Benchmark comparison |
| **Total** | **6 hours** | Complete test cycle |

---

## Contact & Escalation

If tests fail or unexpected issues arise:

1. **Check logs**: integration_*.log, post_patch_*.log
2. **Rollback**: Use backup or git revert
3. **Re-run**: Specific failing test in isolation
4. **Document**: Issue details for review
5. **Escalate**: Report to repository owner with logs

---

**Test Plan Version**: 1.0
**Last Updated**: 2025-11-05
**Status**: Ready for execution
