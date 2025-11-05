# Clockify RAG CLI - Test & Demo Guide

This guide helps you verify the tool is working correctly and demonstrates its capabilities.

## Pre-Test Checklist

Before running tests, verify:

- [ ] Ollama is running: `ollama serve` in a terminal
- [ ] Required models are installed:
  ```bash
  ollama list | grep nomic-embed-text
  ollama list | grep qwen2.5:32b
  ```
- [ ] Virtual environment is activated:
  ```bash
  source rag_env/bin/activate
  ```
- [ ] Python dependencies are installed:
  ```bash
  python3 -c "import requests, numpy; print('OK')"
  ```

## Test Suite

### Test 1: Verify Installation

```bash
# Test Python syntax
python3 -m py_compile clockify_rag.py
echo "✓ Python syntax is valid"

# Test imports
python3 -c "import requests, numpy; print('✓ Dependencies installed')"

# Test CLI help
python3 clockify_rag.py --help
echo "✓ CLI is accessible"
```

**Expected Output**: No errors; help text displays with three subcommands (chunk, embed, ask)

---

### Test 2: Chunking (First-Time Setup)

Skip if `chunks.jsonl` already exists.

```bash
# Run chunking
python3 clockify_rag.py chunk

# Verify output
echo "Chunks created:"
wc -l chunks.jsonl
```

**Expected Output**:
```
Chunking complete: XXX chunks written to chunks.jsonl
Chunks created:
     XXX chunks.jsonl
```

**Troubleshooting**:
- "File not found": Ensure `knowledge_full.md` exists in current directory
- "Permission denied": Check file permissions (`chmod 644`)

---

### Test 3: Embedding (First-Time Setup)

Skip if `vecs.npy` and `meta.jsonl` already exist.

```bash
# Run embedding (this will take time - ensure Ollama is running)
python3 clockify_rag.py embed

# Verify output files
echo "Embeddings created:"
ls -lh vecs.npy meta.jsonl
```

**Expected Output**:
```
Embedding complete: XXX vectors saved to vecs.npy, metadata to meta.jsonl
Embeddings created:
-rw-r--r--  1 user  staff  2.9M  Nov  5 10:30 meta.jsonl
-rw-r--r--  1 user  staff  2.8M  Nov  5 10:30 vecs.npy
```

**Troubleshooting**:
- "Connection refused": Ollama is not running. Start with `ollama serve`
- "Model not found": Pull the model: `ollama pull nomic-embed-text`
- Slow progress: First-time downloads models; subsequent runs are faster

---

### Test 4: Basic Query

```bash
# Simple question
python3 clockify_rag.py ask "What is Clockify?"
```

**Expected Output**: Answer citing relevant snippets, e.g.:
```
Clockify is a time tracking software [12, 45] that allows users to...
```

Or if not found:
```
I don't know based on the MD.
```

---

### Test 5: Query Test Suite

Run these sample queries to verify different types of questions:

#### Category 1: General Information

```bash
echo "Test 1.1: General Information"
python3 clockify_rag.py ask "What is Clockify?"

echo "Test 1.2: Company Info"
python3 clockify_rag.py ask "Who created Clockify?"

echo "Test 1.3: Product Overview"
python3 clockify_rag.py ask "What are the main features of Clockify?"
```

#### Category 2: Features & Functionality

```bash
echo "Test 2.1: Time Tracking"
python3 clockify_rag.py ask "How do I track time in Clockify?"

echo "Test 2.2: Projects"
python3 clockify_rag.py ask "How do I create and manage projects?"

echo "Test 2.3: Reports"
python3 clockify_rag.py ask "What reports does Clockify offer?"

echo "Test 2.4: Integrations"
python3 clockify_rag.py ask "What third-party integrations does Clockify support?"
```

#### Category 3: Configuration

```bash
echo "Test 3.1: Settings"
python3 clockify_rag.py ask "How do I configure my profile settings?"

echo "Test 3.2: Billing"
python3 clockify_rag.py ask "What billing modes does Clockify support?"

echo "Test 3.3: Teams"
python3 clockify_rag.py ask "How do I manage team members?"
```

#### Category 4: Advanced Features

```bash
echo "Test 4.1: Rounding"
python3 clockify_rag.py ask "What is time rounding?"

echo "Test 4.2: Offline"
python3 clockify_rag.py ask "Can I track time offline?"

echo "Test 4.3: Export"
python3 clockify_rag.py ask "How do I export time data?"
```

#### Category 5: Edge Cases (Should Return "I don't know")

```bash
echo "Test 5.1: Unknown Topic"
python3 clockify_rag.py ask "What is the meaning of life?"

echo "Test 5.2: Empty Question"
python3 clockify_rag.py ask ""

echo "Test 5.3: Vague Question"
python3 clockify_rag.py ask "Tell me everything"
```

---

## Performance Baseline

### Timing Tests

Time each stage of the pipeline:

```bash
# Chunking (should be < 5 seconds)
time python3 clockify_rag.py chunk

# Embedding (should be 2-10 minutes depending on hardware)
time python3 clockify_rag.py embed

# Query (should be < 30 seconds, typically 5-15 seconds)
time python3 clockify_rag.py ask "How do I track time?"
```

**Expected Times**:
- Chunking: < 5 seconds
- Embedding: 2-10 minutes (first time; depends on knowledge base size)
- Query: 5-15 seconds (embedding query + LLM inference)

### Resource Monitoring

Monitor system resources during embedding:

```bash
# In another terminal, watch memory/CPU
top -o %MEM
# or on macOS
Activity Monitor
```

**Expected Resource Usage**:
- Memory: 2-4 GB peak during embedding
- CPU: 4-8 threads active
- Disk: 5-10 MB write speed

---

## Validation Tests

### Test 6: Output Format Validation

```bash
# Query and check output format
python3 clockify_rag.py ask "What is Clockify?" | head -20

# Should contain either:
# 1. Answer with citations like [1, 5] or [id1, id3]
# 2. Or the exact string: "I don't know based on the MD."
```

### Test 7: Consistency Test

```bash
# Run the same query twice and compare outputs
query="How do I track time in Clockify?"
echo "Run 1:"
python3 clockify_rag.py ask "$query"

sleep 2

echo "Run 2:"
python3 clockify_rag.py ask "$query"

# Should produce identical output (temperature=0)
```

### Test 8: File Integrity

```bash
# Verify files are valid
python3 << 'EOF'
import json
import numpy as np

print("Checking chunks.jsonl...")
with open('chunks.jsonl', 'r') as f:
    chunk_count = sum(1 for _ in f)
    print(f"  ✓ {chunk_count} chunks")

print("Checking vecs.npy...")
vecs = np.load('vecs.npy')
print(f"  ✓ Shape: {vecs.shape}")
print(f"  ✓ Dtype: {vecs.dtype}")

print("Checking meta.jsonl...")
with open('meta.jsonl', 'r') as f:
    meta_count = sum(1 for _ in f)
    print(f"  ✓ {meta_count} metadata entries")

if chunk_count == vecs.shape[0] == meta_count:
    print("✓ All files consistent!")
else:
    print("✗ File count mismatch!")
EOF
```

---

## Demo Script

Run this complete demo to showcase the tool:

```bash
#!/bin/bash
# demo.sh - Complete Clockify RAG demo

echo "=========================================="
echo "Clockify RAG CLI - Live Demo"
echo "=========================================="
echo ""

activate_env() {
    source rag_env/bin/activate 2>/dev/null || {
        echo "Please activate: source rag_env/bin/activate"
        exit 1
    }
}

check_ollama() {
    timeout 2 python3 -c "import requests; requests.post('http://127.0.0.1:11434/api/tags')" 2>/dev/null || {
        echo "ERROR: Ollama not running"
        echo "Start with: ollama serve"
        exit 1
    }
}

run_query() {
    local query="$1"
    echo "Query: \"$query\""
    python3 clockify_rag.py ask "$query"
    echo ""
}

activate_env
check_ollama

echo "1. Basic Feature Inquiry"
run_query "How do I track time in Clockify?"

echo "2. Configuration Question"
run_query "How do I create a project?"

echo "3. Advanced Feature"
run_query "What is time rounding?"

echo "4. Integration Question"
run_query "Does Clockify integrate with Slack?"

echo "5. Unknown Topic (should return 'I don't know')"
run_query "What is quantum computing?"

echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
```

Save this as `demo.sh`, make executable (`chmod +x demo.sh`), and run (`./demo.sh`).

---

## Common Issues & Solutions

### Issue: "Model not found" error during embedding

**Cause**: Required embedding model not downloaded
**Solution**:
```bash
ollama pull nomic-embed-text
python3 clockify_rag.py embed
```

### Issue: "Connection refused" during query

**Cause**: Ollama server not running or wrong URL
**Solution**:
```bash
# Start Ollama
ollama serve

# Or check URL in clockify_rag.py (line 11)
# Should match your Ollama instance
```

### Issue: Embedding takes too long

**Cause**: First-time model download or large knowledge base
**Solution**:
- Embedding is expected to take 2-10 minutes on first run
- Subsequent operations are much faster
- Ensure sufficient disk space for model cache (~2-5 GB)

### Issue: Queries return "I don't know" for valid topics

**Cause**: Low similarity threshold or poor chunk overlap
**Solution**:
1. Check if topic is actually in knowledge_full.md
2. Lower similarity threshold in `ask_question()` from 0.3 to 0.2
3. Re-run embedding with larger overlap

### Issue: Out of memory during embedding

**Cause**: System has insufficient RAM
**Solution**:
```python
# In clockify_rag.py, reduce chunk size:
CHUNK_SIZE = 800  # from 1600
CHUNK_OVERLAP = 100  # from 200
```
Then re-run: `python3 clockify_rag.py chunk && python3 clockify_rag.py embed`

---

## Performance Optimization Guide

### To Speed Up Embedding

```bash
# Use smaller model
EMBED_MODEL = "All-MiniLM-L6-v2"

# Reduce chunk size
CHUNK_SIZE = 800

# Fewer chunks = faster embedding
# But may reduce retrieval quality
```

### To Speed Up Queries

```bash
# Reduce top-K retrieval
top_k = 4  # from 6

# Use faster LLM
CHAT_MODEL = "neural-chat:7b"

# Increase temperature threshold
SIMILARITY_THRESHOLD = 0.5  # from 0.3
```

### To Improve Answer Quality

```bash
# Use larger, slower LLM
CHAT_MODEL = "qwen2.5:72b"  # if you have the VRAM

# Increase number of retrieved chunks
top_k = 10

# Lower similarity threshold
SIMILARITY_THRESHOLD = 0.2
```

---

## Verification Checklist

After running tests, verify:

- [ ] Chunking completes without errors
- [ ] Embedding completes without errors
- [ ] At least 3 test queries return relevant answers
- [ ] Answer format includes snippet citations
- [ ] "I don't know" responses work for unknown topics
- [ ] Performance is acceptable for your use case
- [ ] No memory/CPU issues observed
- [ ] File sizes are reasonable (vecs.npy ~3MB, meta.jsonl ~2-10MB)

---

## Reporting Issues

If tests fail:

1. Check Prerequisites (Ollama running, models installed)
2. Run individual tests to isolate the problem
3. Check error messages carefully
4. Review Troubleshooting section
5. Consult README_RAG.md for detailed explanations

For persistent issues:
- Verify Ollama is accessible: `curl http://127.0.0.1:11434/api/tags`
- Check model availability: `ollama list`
- Review Ollama logs for errors
- Test with smaller knowledge base if available

---

**Last Updated**: 2025-11-05
**Version**: 1.0
