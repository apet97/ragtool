#!/usr/bin/env bash
# CI Bootstrap Script
# Installs minimal dependencies required for testing before running pytest
# Usage: bash scripts/ci_bootstrap.sh [test|minimal|full]

set -e  # Exit on error

MODE="${1:-minimal}"

echo "======================================================================="
echo "CI Bootstrap - Installing dependencies (mode: $MODE)"
echo "======================================================================="

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Error: Python not found. Please install Python 3.7+"
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# Upgrade pip
echo ""
echo "[1/3] Upgrading pip..."
$PYTHON -m pip install --upgrade pip

# Install minimal numeric stack for tests
echo ""
echo "[2/3] Installing minimal dependencies..."

case "$MODE" in
    minimal)
        # Minimal deps for basic tests (no ML models)
        echo "Installing: numpy, requests, nltk, rank-bm25, pytest"
        $PYTHON -m pip install numpy requests nltk rank-bm25 pytest pytest-cov
        ;;

    test)
        # Test deps including ML but not FAISS
        echo "Installing: core deps + sentence-transformers (no FAISS)"
        $PYTHON -m pip install -r requirements.txt || {
            echo "⚠️  Warning: Full requirements.txt failed, falling back to minimal"
            $PYTHON -m pip install numpy requests nltk rank-bm25 pytest pytest-cov
            $PYTHON -m pip install sentence-transformers torch --no-deps || true
        }
        ;;

    full)
        # Full production deps
        echo "Installing: all production dependencies"
        $PYTHON -m pip install -r requirements.txt
        ;;

    *)
        echo "❌ Error: Invalid mode '$MODE'. Use: minimal, test, or full"
        exit 1
        ;;
esac

# Verify critical imports
echo ""
echo "[3/3] Verifying critical imports..."

$PYTHON -c "
import sys
import importlib

critical_imports = {
    'minimal': ['numpy', 'requests'],
    'test': ['numpy', 'requests', 'nltk', 'pytest'],
    'full': ['numpy', 'requests', 'nltk', 'pytest', 'sentence_transformers']
}

mode = '${MODE}'
imports = critical_imports.get(mode, critical_imports['minimal'])

failed = []
for module in imports:
    try:
        importlib.import_module(module)
        print(f'✓ {module}')
    except ImportError as e:
        print(f'✗ {module}: {e}')
        failed.append(module)

if failed:
    print(f'\n❌ Failed to import: {', '.join(failed)}')
    sys.exit(1)
else:
    print(f'\n✓ All critical imports successful ({len(imports)} modules)')
" || {
    echo "❌ Import verification failed"
    exit 1
}

echo ""
echo "======================================================================="
echo "✅ Bootstrap complete (mode: $MODE)"
echo "======================================================================="
echo ""
echo "Next steps:"
echo "  • Run tests: pytest tests/ -v"
echo "  • Run with coverage: pytest tests/ --cov=clockify_rag --cov-report=html"
echo "  • Run evaluation: python3 eval.py --dataset eval_datasets/clockify_v1.jsonl"
echo ""
