# Apple Silicon M1 Compatibility Guide

**Status**: ✅ Fully Compatible with M1 Pro/Max/Ultra
**Last Updated**: 2025-11-05
**Version**: v4.1.2

## Overview

The Clockify RAG CLI is fully compatible with Apple Silicon (M1/M2/M3) Macs. This guide covers installation instructions, known issues, and architecture-specific optimizations.

## Quick Start for M1 Macs

### Prerequisites
- macOS 12.3 or later (required for PyTorch MPS acceleration)
- Python 3.9+ (native ARM64 build recommended)
- Homebrew (ARM64 version)

### Verify Native ARM Python

Before installation, verify you're running native ARM Python (not Rosetta):

```bash
python3 -c "import platform; print(f'Machine: {platform.machine()}, System: {platform.system()}')"
```

Expected output: `Machine: arm64, System: Darwin`

If you see `x86_64`, you're running under Rosetta translation. Install native ARM Python:

```bash
# Install Python via Homebrew (ARM native)
brew install python@3.11
```

## Installation

### Option 1: Using Pip (Recommended for most users)

```bash
# Create and activate virtual environment
python3 -m venv rag_env
source rag_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Using Conda (Better for FAISS compatibility)

Conda is **strongly recommended** for M1 Macs due to better ARM64 package support, especially for FAISS.

```bash
# Create conda environment
conda create -n rag_env python=3.11
conda activate rag_env

# Install dependencies via conda-forge (ARM-optimized)
conda install -c conda-forge numpy requests
conda install -c pytorch sentence-transformers pytorch
conda install -c conda-forge faiss-cpu=1.8.0
conda install -c conda-forge rank-bm25

# Install remaining packages via pip
pip install urllib3==2.2.3
```

## Dependency Compatibility Matrix

| Package | Version | M1 Status | Installation Method | Notes |
|---------|---------|-----------|---------------------|-------|
| **numpy** | 2.3.4 | ✅ Full Support | pip/conda | ARM-optimized builds available |
| **requests** | 2.32.5 | ✅ Full Support | pip/conda | Pure Python, no architecture issues |
| **urllib3** | 2.2.3 | ✅ Full Support | pip/conda | Pure Python |
| **sentence-transformers** | 3.3.1 | ✅ Full Support | pip/conda | Depends on PyTorch |
| **torch** | 2.4.2 | ✅ Full Support | pip/conda | MPS acceleration available |
| **rank-bm25** | 0.2.2 | ✅ Full Support | pip/conda | Pure Python |
| **faiss-cpu** | 1.8.0.post1 | ⚠️ Conditional | **conda preferred** | See section below |

### FAISS on M1: Important Notes

**Issue**: The pip package `faiss-cpu==1.8.0.post1` may have compatibility issues on ARM64 macOS.

**Solutions**:

1. **Use Conda (Recommended)**:
   ```bash
   conda install -c conda-forge faiss-cpu=1.8.0
   ```

2. **Run without FAISS** (fallback mode):
   ```bash
   # The application automatically falls back to full-scan cosine similarity
   # if FAISS is not available or fails to load
   export USE_ANN=none
   python3 clockify_support_cli_final.py chat
   ```

3. **Build FAISS from source** (advanced):
   ```bash
   # Install dependencies
   brew install cmake swig

   # Clone and build FAISS
   git clone https://github.com/facebookresearch/faiss.git
   cd faiss
   cmake -B build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON .
   make -C build -j$(sysctl -n hw.ncpu)
   cd build/faiss/python && pip install .
   ```

## Architecture-Specific Optimizations

### Automatic ARM64 Detection

The code automatically detects Apple Silicon and applies optimizations:

```python
# clockify_support_cli_final.py:235
is_macos_arm64 = platform.system() == "Darwin" and platform.machine() == "arm64"

if is_macos_arm64:
    # Uses IndexFlatIP (linear search) instead of IVFFlat
    # Avoids segfault in IVF training on ARM64
    index = faiss.IndexFlatIP(dim)
```

**Why this matters**: FAISS IVFFlat training can cause segmentation faults on M1 Macs due to multiprocessing issues with Python 3.12+. The fallback to `IndexFlatIP` provides:
- ✅ Stability (no crashes)
- ✅ Accuracy (exact search, not approximate)
- ⚠️ Slower performance (linear O(N) vs. O(log N))

### Performance Considerations

| Operation | M1 Performance | Notes |
|-----------|---------------|-------|
| **Embedding generation** | Excellent | PyTorch MPS acceleration |
| **FAISS indexing** | Good | Uses FlatIP (no IVF training) |
| **FAISS search** | Good | Linear search O(N), acceptable for <100K chunks |
| **BM25 scoring** | Excellent | Pure Python, benefits from ARM efficiency |
| **LLM inference** | N/A | Handled by external Ollama server |

## Troubleshooting

### Issue 1: FAISS ImportError

**Symptom**:
```
ImportError: dlopen(..._swigfaiss.so, 0x0002): tried: ... (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64'))
```

**Solution**:
- You installed x86_64 FAISS under Rosetta
- Reinstall using native ARM Python and conda:
  ```bash
  conda install -c conda-forge faiss-cpu=1.8.0
  ```

### Issue 2: PyTorch MPS Not Available

**Symptom**:
```python
import torch
torch.backends.mps.is_available()  # Returns False
```

**Solution**:
- Ensure macOS 12.3+
- Reinstall PyTorch:
  ```bash
  pip install --upgrade --force-reinstall torch
  ```

### Issue 3: Slow Embedding Performance

**Symptom**: Embedding takes >5 seconds per query

**Possible Causes**:
1. Running under Rosetta (x86_64 translation)
2. PyTorch not using MPS acceleration

**Solution**:
```bash
# Check if running native ARM
python3 -c "import platform; print(platform.machine())"

# Verify PyTorch MPS
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Issue 4: Platform Detection Fails

**Symptom**: Code doesn't detect M1, uses wrong FAISS index type

**Debugging**:
```python
import platform
print(f"System: {platform.system()}")
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")

# Expected on M1:
# System: Darwin
# Machine: arm64
# Processor: arm (or empty string)
```

**Note**: Version v4.1.2 fixed a bug where `platform.processor()` was used instead of the more reliable `platform.machine()`.

## Verified Configurations

| Python | macOS | Conda/Pip | Status | Notes |
|--------|-------|-----------|--------|-------|
| 3.11.8 | 14.5 (Sonoma) | conda | ✅ Verified | Recommended |
| 3.11.8 | 14.5 (Sonoma) | pip | ⚠️ FAISS issues | Use conda for FAISS |
| 3.12.3 | 14.5 (Sonoma) | conda | ✅ Verified | Works with FlatIP fallback |
| 3.9.x | 13.x (Ventura) | conda | ✅ Verified | Older but stable |

## Performance Benchmarks (M1 Pro, 16GB)

### Build Phase (knowledge_full.md → indexes)
```
Chunking:         ~2 seconds
Embedding:        ~30 seconds (384 chunks, local SentenceTransformer)
FAISS indexing:   ~0.5 seconds (FlatIP)
BM25 indexing:    ~1 second
Total build:      ~35 seconds
```

### Query Phase (single question)
```
Query embedding:  ~0.3 seconds
FAISS retrieval:  ~0.05 seconds (384 chunks, FlatIP)
BM25 retrieval:   ~0.02 seconds
Hybrid scoring:   ~0.01 seconds
LLM inference:    ~5-10 seconds (Ollama, qwen2.5:32b)
Total latency:    ~6-11 seconds
```

## Environment Variables for M1 Optimization

```bash
# Force FAISS fallback if needed
export USE_ANN="none"

# Reduce FAISS clusters for smaller KBs (already set to 64 for ARM)
export ANN_NLIST="64"

# Enable PyTorch MPS acceleration (if available)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Increase timeout for slower M1 Air models
export EMB_READ_TIMEOUT="90"
export CHAT_READ_TIMEOUT="180"
```

## Known Limitations

1. **FAISS IVFFlat Disabled**: M1 uses `IndexFlatIP` instead of `IndexIVFFlat` due to segfault issues. This trades speed for stability.
2. **Conda Recommended**: pip installation may fail for FAISS on ARM64.
3. **MPS Acceleration**: Only available in PyTorch 1.12+, macOS 12.3+.
4. **Rosetta Compatibility**: The application runs under Rosetta but with degraded performance. Native ARM is strongly recommended.

## Migration from Intel Mac

If moving from an Intel Mac:

1. **Don't copy virtualenvs**: Recreate from scratch
2. **Rebuild indexes**: Old `faiss.index` files are architecture-specific
3. **Use conda**: Better ARM package support

```bash
# Clean old artifacts
make clean  # or: rm -f chunks.jsonl vecs_n.npy meta.jsonl bm25.json faiss.index

# Rebuild with ARM-native Python
source rag_env/bin/activate  # or conda activate
python3 clockify_support_cli_final.py build knowledge_full.md
```

## Testing on M1

```bash
# Run smoke tests
bash scripts/smoke.sh

# Run acceptance tests
bash scripts/acceptance_test.sh

# Manual test
python3 clockify_support_cli_final.py chat --debug
> How do I track time in Clockify?
# Should see: "macOS arm64 detected: using IndexFlatIP"
```

## VPN and Network Troubleshooting

### Issue 5: VPN Blocks Localhost Connections

**Symptom**:
```
ConnectionError: Failed to connect to http://127.0.0.1:11434
```

**Possible Causes**:
1. VPN software blocks localhost ports
2. Ollama running on different machine
3. Firewall rules blocking local connections

**Solution 1: Verify Ollama is Running**
```bash
curl http://127.0.0.1:11434/api/version
# Should return: {"version":"0.x.x"}
```

**Solution 2: Ollama on Remote Machine via VPN**
```bash
# Find Ollama machine IP (via VPN)
ping ollama-server.vpn  # or use VPN IP

# Configure endpoint
export OLLAMA_URL="http://10.x.x.x:11434"
python3 clockify_support_cli_final.py chat
```

**Solution 3: VPN Requires HTTP Proxy**
```bash
# Enable proxy support (disabled by default for security)
export ALLOW_PROXIES=1  # legacy USE_PROXY=1 also works
export http_proxy="http://proxy.company.com:8080"
export OLLAMA_URL="http://127.0.0.1:11434"
python3 clockify_support_cli_final.py chat
```

**Solution 4: Bind Ollama to Different Port**
```bash
# If VPN blocks port 11434
ollama serve --host 127.0.0.1:11435
export OLLAMA_URL="http://127.0.0.1:11435"
```

### Environment Variables Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint URL |
| `ALLOW_PROXIES` | `0` (disabled) | Enable HTTP proxy support (`USE_PROXY` alias supported) |
| `http_proxy` | (none) | HTTP proxy server |
| `https_proxy` | (none) | HTTPS proxy server |

### VPN Testing Checklist

```bash
# 1. Test with VPN connected (Ollama local)
export OLLAMA_URL="http://127.0.0.1:11434"
python3 clockify_support_cli_final.py chat --debug

# 2. Test with remote Ollama via VPN
export OLLAMA_URL="http://10.x.x.x:11434"  # Replace with actual VPN IP
curl "$OLLAMA_URL/api/version"  # Verify connectivity first
python3 clockify_support_cli_final.py chat

# 3. Test with VPN proxy (if required)
export ALLOW_PROXIES=1
export http_proxy="http://proxy:8080"
export https_proxy="http://proxy:8080"
python3 clockify_support_cli_final.py chat
```

**Note**: The default `127.0.0.1` (localhost) configuration is VPN-safe and works in most VPN environments. Only configure remote endpoints if Ollama runs on a different machine.

## Additional Resources

- **PyTorch M1 Guide**: https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/
- **FAISS Installation**: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
- **Conda M1 Setup**: https://towardsdatascience.com/python-conda-environments-for-both-arm64-and-x86-64-on-m1-apple-silicon-147b943ffa55

## Support

If you encounter M1-specific issues:

1. Check this guide's troubleshooting section
2. Verify native ARM Python: `platform.machine() == "arm64"`
3. Try conda instead of pip
4. Open an issue with:
   - Python version and architecture
   - macOS version
   - Installation method (pip/conda)
   - Full error traceback

---

**Maintained by**: Clockify RAG Team
**Last Tested**: 2025-11-05 on M1 Pro, macOS 14.5, Python 3.11.8
