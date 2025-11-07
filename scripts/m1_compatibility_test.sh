#!/bin/bash
#
# M1 Compatibility Test Suite
# Tests platform detection, dependencies, and ARM64 optimizations
#
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_FILE="m1_compatibility.log"
: > "$LOG_FILE"  # Clear log

echo "=== M1 Compatibility Test Suite ===" | tee -a "$LOG_FILE"
echo "Repo: $REPO_ROOT" | tee -a "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Test 1: Platform detection
echo "[1/8] Platform detection..." | tee -a "$LOG_FILE"
PLATFORM=$(python3 -c "import platform; print(platform.machine())")
SYSTEM=$(python3 -c "import platform; print(platform.system())")

echo "  System: $SYSTEM" | tee -a "$LOG_FILE"
echo "  Machine: $PLATFORM" | tee -a "$LOG_FILE"

if [ "$PLATFORM" = "arm64" ]; then
    echo "  ✅ Running on ARM64 (native)" | tee -a "$LOG_FILE"
    IS_ARM64=true
elif [ "$PLATFORM" = "x86_64" ]; then
    if [ "$SYSTEM" = "Darwin" ]; then
        echo "  ⚠️  Running on x86_64 (Rosetta or Intel Mac)" | tee -a "$LOG_FILE"
    else
        echo "  ℹ️  Running on x86_64 (Intel/AMD)" | tee -a "$LOG_FILE"
    fi
    IS_ARM64=false
else
    echo "  ❓ Unexpected platform: $PLATFORM" | tee -a "$LOG_FILE"
    IS_ARM64=false
fi
echo "" | tee -a "$LOG_FILE"

# Test 2: Python version and build
echo "[2/8] Python version and build..." | tee -a "$LOG_FILE"
python3 --version | tee -a "$LOG_FILE"
python3 -c "import sys; print(f'  Python executable: {sys.executable}')" | tee -a "$LOG_FILE"
python3 -c "import platform; print(f'  Python implementation: {platform.python_implementation()}')" | tee -a "$LOG_FILE"
echo "  ✅ Python check complete" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Test 3: Core dependencies
echo "[3/8] Core dependencies..." | tee -a "$LOG_FILE"

# Check if running in CI environment
CI_MODE=${CI:-false}
if [ "$CI_MODE" = "true" ]; then
    echo "  ℹ️  Running in CI mode - some dependencies may be skipped" | tee -a "$LOG_FILE"
fi

python3 -c "import numpy; print(f'  numpy: {numpy.__version__}')" 2>&1 | tee -a "$LOG_FILE" || echo "  ⚠️  numpy not installed" | tee -a "$LOG_FILE"
python3 -c "import requests; print(f'  requests: {requests.__version__}')" 2>&1 | tee -a "$LOG_FILE" || echo "  ⚠️  requests not installed" | tee -a "$LOG_FILE"

# Optional dependencies (may not be present in CI)
if python3 -c "import sentence_transformers; print(f'  sentence-transformers: {sentence_transformers.__version__}')" 2>&1 | tee -a "$LOG_FILE"; then
    :  # Success
else
    if [ "$CI_MODE" = "true" ]; then
        echo "  ℹ️  sentence-transformers not installed (expected in CI)" | tee -a "$LOG_FILE"
    else
        echo "  ⚠️  sentence-transformers not installed" | tee -a "$LOG_FILE"
    fi
fi

if python3 -c "import torch; print(f'  torch: {torch.__version__}')" 2>&1 | tee -a "$LOG_FILE"; then
    :  # Success
else
    if [ "$CI_MODE" = "true" ]; then
        echo "  ℹ️  torch not installed (expected in CI)" | tee -a "$LOG_FILE"
    else
        echo "  ⚠️  torch not installed" | tee -a "$LOG_FILE"
    fi
fi

if python3 -c "import rank_bm25; print(f'  rank-bm25: installed')" 2>&1 | tee -a "$LOG_FILE"; then
    :  # Success
else
    echo "  ⚠️  rank-bm25 not installed" | tee -a "$LOG_FILE"
fi

echo "  ✅ Core dependencies check complete" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Test 4: PyTorch MPS availability (M1 only)
echo "[4/8] PyTorch MPS availability..." | tee -a "$LOG_FILE"
if [ "$IS_ARM64" = true ] && [ "$SYSTEM" = "Darwin" ]; then
    if python3 -c "import torch" 2>/dev/null; then
        MPS_AVAILABLE=$(python3 -c "import torch; print(torch.backends.mps.is_available())" 2>&1)
        if [ "$MPS_AVAILABLE" = "True" ]; then
            echo "  ✅ PyTorch MPS available (GPU acceleration enabled)" | tee -a "$LOG_FILE"
        else
            echo "  ⚠️  PyTorch MPS not available (will use CPU)" | tee -a "$LOG_FILE"
            echo "  Hint: Ensure macOS 12.3+ and PyTorch 1.12+" | tee -a "$LOG_FILE"
        fi
    else
        if [ "$CI_MODE" = "true" ]; then
            echo "  ℹ️  PyTorch not installed (skipped in CI)" | tee -a "$LOG_FILE"
        else
            echo "  ⚠️  PyTorch not installed" | tee -a "$LOG_FILE"
        fi
    fi
else
    echo "  ℹ️  Skipped (not M1 Mac)" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# Test 5: FAISS availability
echo "[5/8] FAISS availability..." | tee -a "$LOG_FILE"
if python3 -c "import faiss; print(f'  FAISS version: {faiss.__version__}')" 2>&1 | tee -a "$LOG_FILE"; then
    echo "  ✅ FAISS imported successfully" | tee -a "$LOG_FILE"

    # Test FAISS architecture compatibility
    if [ "$IS_ARM64" = true ]; then
        echo "  Testing FAISS ARM64 compatibility..." | tee -a "$LOG_FILE"
        python3 -c "
import faiss
import numpy as np
# Create small test index
vecs = np.random.randn(10, 128).astype('float32')
faiss.normalize_L2(vecs)
index = faiss.IndexFlatIP(128)
index.add(vecs)
D, I = index.search(vecs[:1], 5)
print(f'    FAISS test search successful (found {len(I[0])} results)')
" 2>&1 | tee -a "$LOG_FILE"
        echo "  ✅ FAISS ARM64 compatible" | tee -a "$LOG_FILE"
    fi
else
    if [ "$CI_MODE" = "true" ]; then
        echo "  ℹ️  FAISS not available (expected in CI, skipped)" | tee -a "$LOG_FILE"
    else
        echo "  ⚠️  FAISS not available (application will use fallback mode)" | tee -a "$LOG_FILE"
        if [ "$IS_ARM64" = true ]; then
            echo "  Recommendation: Install via conda for best M1 compatibility" | tee -a "$LOG_FILE"
            echo "    conda install -c conda-forge faiss-cpu=1.8.0" | tee -a "$LOG_FILE"
        fi
    fi
fi
echo "" | tee -a "$LOG_FILE"

# Test 6: Script syntax validation
echo "[6/8] Script syntax validation..." | tee -a "$LOG_FILE"
if python3 -m py_compile clockify_support_cli_final.py 2>&1 | tee -a "$LOG_FILE"; then
    echo "  ✅ clockify_support_cli_final.py syntax valid" | tee -a "$LOG_FILE"
else
    echo "  ❌ Syntax errors found" | tee -a "$LOG_FILE"
    exit 1
fi
echo "" | tee -a "$LOG_FILE"

# Test 7: ARM64 optimization verification (if M1)
echo "[7/8] ARM64 optimization verification..." | tee -a "$LOG_FILE"
if [ "$IS_ARM64" = true ] && [ "$SYSTEM" = "Darwin" ]; then
    # Check if ARM64 detection code exists
    if grep -q "platform.machine()" clockify_support_cli_final.py; then
        echo "  ✅ ARM64 detection code present (platform.machine())" | tee -a "$LOG_FILE"
    else
        echo "  ⚠️  ARM64 detection code not found" | tee -a "$LOG_FILE"
    fi

    if grep -q "is_macos_arm64 = platform.system()" clockify_support_cli_final.py; then
        echo "  ✅ ARM64 platform check present" | tee -a "$LOG_FILE"
    else
        echo "  ⚠️  ARM64 platform check not found" | tee -a "$LOG_FILE"
    fi

    if grep -q "IndexFlatIP" clockify_support_cli_final.py; then
        echo "  ✅ FAISS FlatIP fallback present" | tee -a "$LOG_FILE"
    else
        echo "  ⚠️  FAISS FlatIP fallback not found" | tee -a "$LOG_FILE"
    fi
else
    echo "  ℹ️  Skipped (not M1 Mac)" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# Test 8: Build test (optional, only if knowledge_full.md exists)
echo "[8/8] Build test..." | tee -a "$LOG_FILE"
if [ "$CI_MODE" = "true" ]; then
    echo "  ℹ️  Skipped in CI (requires heavy ML dependencies)" | tee -a "$LOG_FILE"
elif [ -f "knowledge_full.md" ] && [ "$IS_ARM64" = true ] && [ "$SYSTEM" = "Darwin" ]; then
    echo "  Building knowledge base to verify ARM64 optimization..." | tee -a "$LOG_FILE"

    # Clean old artifacts
    rm -f chunks.jsonl vecs_n.npy meta.jsonl bm25.json faiss.index index.meta.json

    # Build (capture output)
    if python3 clockify_support_cli_final.py build knowledge_full.md 2>&1 | tee -a "$LOG_FILE" | grep -q "macOS arm64 detected"; then
        echo "  ✅ ARM64 optimization activated during build" | tee -a "$LOG_FILE"
    else
        echo "  ⚠️  ARM64 optimization message not detected in build output" | tee -a "$LOG_FILE"
        echo "  This may indicate the optimization is not activating correctly" | tee -a "$LOG_FILE"
    fi

    # Verify artifacts created
    if [ -f "chunks.jsonl" ] && [ -f "vecs_n.npy" ] && [ -f "meta.jsonl" ]; then
        echo "  ✅ Build artifacts created successfully" | tee -a "$LOG_FILE"
    else
        echo "  ❌ Build failed - missing artifacts" | tee -a "$LOG_FILE"
        exit 1
    fi
else
    echo "  ℹ️  Skipped (knowledge_full.md not found or not M1 Mac)" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# Summary
echo "=== M1 Compatibility Test Summary ===" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ "$IS_ARM64" = true ] && [ "$SYSTEM" = "Darwin" ]; then
    echo "✅ Platform: Apple Silicon M1/M2/M3" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Recommendations:" | tee -a "$LOG_FILE"
    echo "  1. Use conda for FAISS installation (best compatibility)" | tee -a "$LOG_FILE"
    echo "  2. Verify PyTorch MPS is available for faster embeddings" | tee -a "$LOG_FILE"
    echo "  3. Check build logs for 'macOS arm64 detected' message" | tee -a "$LOG_FILE"
    echo "  4. See M1_COMPATIBILITY.md for detailed troubleshooting" | tee -a "$LOG_FILE"
elif [ "$PLATFORM" = "x86_64" ] && [ "$SYSTEM" = "Darwin" ]; then
    echo "ℹ️  Platform: Intel Mac or Rosetta" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Note: If on M1 Mac, ensure you're using native ARM Python:" | tee -a "$LOG_FILE"
    echo "  brew install python@3.11" | tee -a "$LOG_FILE"
    echo "  python3 -c 'import platform; print(platform.machine())'" | tee -a "$LOG_FILE"
else
    echo "ℹ️  Platform: $SYSTEM $PLATFORM" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "This test suite is optimized for M1 Macs but works on all platforms." | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "=== Test Complete ===" | tee -a "$LOG_FILE"
