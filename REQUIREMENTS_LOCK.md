# Requirements Lock File

## Overview

This project uses `pyproject.toml` as the source of truth for dependencies. For production deployments requiring reproducible builds, a `requirements.lock` file should be generated.

## Generating requirements.lock

**On Linux/macOS (recommended):**

```bash
# Create virtual environment
python3 -m venv rag_env
source rag_env/bin/activate

# Install from pyproject.toml
pip install -e .

# Generate lock file
pip freeze > requirements.lock
```

**On Apple Silicon (M1/M2/M3):**

```bash
# Use conda for best compatibility
conda create -n rag_env python=3.11
conda activate rag_env

# Install dependencies (see M1_COMPATIBILITY.md for details)
conda install -c conda-forge faiss-cpu=1.8.0 numpy
conda install -c pytorch sentence-transformers pytorch
pip install -e .

# Generate lock file
pip freeze > requirements.lock
```

**Using Make:**

```bash
make venv
make install
make freeze
```

## Why Use a Lock File?

### Benefits

1. **Reproducible Builds**: Ensures exact versions across deployments
2. **Security**: Pin tested versions to avoid supply chain attacks
3. **CI/CD Stability**: Prevents unexpected breaking changes
4. **Compliance**: Required for regulated environments

### When to Use

- **Production deployments**: Always use lock file
- **CI/CD pipelines**: Lock file ensures test consistency
- **Team collaboration**: Guarantees same environment
- **Docker images**: Pin versions for reproducibility

### When Not to Use

- **Development**: Use `pyproject.toml` for flexibility
- **Quick testing**: Install from requirements.txt is faster
- **Package distribution**: Rely on pyproject.toml version ranges

## CI/CD Usage

The project includes a `scripts/ci_bootstrap.sh` that handles dependency installation in CI without requiring a lock file:

```bash
# Minimal dependencies (fastest)
bash scripts/ci_bootstrap.sh minimal

# Test dependencies (includes ML, no FAISS)
bash scripts/ci_bootstrap.sh test

# Full production dependencies
bash scripts/ci_bootstrap.sh full
```

## Platform-Specific Notes

### Linux/Ubuntu

Standard pip installation works reliably:
```bash
pip install -r requirements.txt
pip freeze > requirements.lock
```

### macOS Intel

Same as Linux - no special considerations.

### macOS Apple Silicon (M1/M2/M3)

**CRITICAL**: Use conda for FAISS installation. See [M1_COMPATIBILITY.md](M1_COMPATIBILITY.md) for complete guide.

**Known Issues**:
- `torch==2.4.2` may not be available on all platforms - use range `torch>=2.3.0,<2.6.0`
- `faiss-cpu` requires conda on ARM64 - pip installation may fail
- Some wheels may need ARM64-specific builds

### Windows

Use WSL2 for best compatibility. Native Windows support is experimental.

## Verifying Lock File

After generating a lock file, verify it works:

```bash
# Create fresh environment
python3 -m venv test_env
source test_env/bin/activate

# Install from lock file
pip install -r requirements.lock

# Run tests
pytest tests/ -v

# Verify critical imports
python -c "import numpy, torch, sentence_transformers, faiss; print('✅ All imports successful')"
```

## Updating Lock File

Lock files should be regenerated when:
- Dependencies change in pyproject.toml
- Security patches are released
- Major version updates are tested and approved
- Platform requirements change (e.g., new Python version)

```bash
# Update dependencies
pip install -e . --upgrade

# Regenerate lock file
pip freeze > requirements.lock

# Test with new versions
make test

# Commit if tests pass
git add requirements.lock
git commit -m "chore: update requirements.lock"
```

## References

- **Source**: `pyproject.toml` - Dependency specifications
- **Lock**: `requirements.lock` - Pinned versions (generated)
- **Fallback**: `requirements.txt` - Minimum requirements
- **M1**: `requirements-m1.txt` - Apple Silicon specific
- **CI**: `scripts/ci_bootstrap.sh` - CI installation

## See Also

- [M1_COMPATIBILITY.md](M1_COMPATIBILITY.md) - Apple Silicon installation guide
- [CLAUDE.md](CLAUDE.md) - Project overview and architecture
- [README.md](README.md) - Quick start and usage
- GitHub Actions CI - `.github/workflows/ci.yml`
