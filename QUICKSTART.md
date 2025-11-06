# Clockify RAG CLI - Quick Start Guide

## One-Time Setup

### 1. Install Dependencies

The tool requires Python 3.7+ and a few packages. A virtual environment has already been created in `rag_env/`.

Activate the environment (MacOS/Linux):
```bash
source rag_env/bin/activate
```

Windows:
```bash
rag_env\Scripts\activate
```

Dependencies are already installed: `requests`, `numpy`.

### 2. Start Ollama Server

Ensure Ollama is running locally (default: `http://10.127.0.192:11434`):

```bash
ollama serve
```

Verify the required models are available:
```bash
ollama list
```

If missing, pull them:
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:32b
```

---

## Building the Knowledge Base (One-Time)

The production CLI handles chunking and embedding in a single build step. Run this after activating the virtual environment:

```bash
source rag_env/bin/activate
python3 clockify_support_cli_final.py build knowledge_full.md
```

The command validates existing artifacts, rebuilds them if needed, and creates the following files:

- `chunks.jsonl` – structured document chunks
- `vecs_n.npy` – normalized dense embeddings (float32)
- `meta.jsonl` – metadata for each chunk
- `bm25.json` – sparse retrieval index for keyword matching
- `index.meta.json` – build metadata (used for automatic rebuild detection)

The initial build can take several minutes depending on hardware; subsequent runs are faster thanks to incremental rebuild checks.

---

## Using the Tool (Repeatable)

### Start the Interactive Chat

After the build completes, launch the REPL interface:

```bash
source rag_env/bin/activate
python3 clockify_support_cli_final.py chat
```

The chat loop keeps the index in memory and lets you ask follow-up questions. Toggle diagnostics at any time with the `:debug` command.

### Run a Single Query

Use the `ask` subcommand for one-off questions without entering the REPL:

```bash
python3 clockify_support_cli_final.py ask "How do I track time in Clockify?"
```

Add `--json` for structured output that includes the selected chunk IDs and token usage:

```bash
python3 clockify_support_cli_final.py ask "How do I track time in Clockify?" --json
```

---

## Example Queries

Try these questions to test the system:

```bash
python3 clockify_support_cli_final.py ask "What is time rounding in Clockify?"
python3 clockify_support_cli_final.py ask "How do I set up projects?"
python3 clockify_support_cli_final.py ask "Can I track time offline?"
python3 clockify_support_cli_final.py ask "How do I export reports?"
python3 clockify_support_cli_final.py ask "What billing modes does Clockify support?"
```

---

## File Structure

After setup, your directory contains:

```
/Users/15x/Downloads/KBDOC/
├── clockify_support_cli_final.py  # Main CLI tool (build/chat/ask)
├── rag_env/                  # Virtual environment (activate with: source rag_env/bin/activate)
├── knowledge_full.md         # Source documentation (6.9 MB)
├── chunks.jsonl              # Generated: documentation chunks
├── vecs_n.npy                # Generated: normalized embedding vectors
├── meta.jsonl                # Generated: chunk metadata
├── README_RAG.md             # Full documentation
└── QUICKSTART.md             # This file
```

---

## Troubleshooting

### Connection Error: "Cannot connect to Ollama"

```
Embedding request failed for chunk 0: Connection refused
```

**Solution**: Ensure Ollama is running:
```bash
ollama serve
```

Check the `OLLAMA_URL` environment variable or override with `--ollama-url` when running the CLI.

### Model Not Found

```
Embedding API returned error 404
```

**Solution**: Pull the missing model:
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:32b
```

### Memory Issues

If you run out of memory during embedding:
- Reduce `CHUNK_SIZE` in `clockify_support_cli_final.py` (defaults to 1600)
- Ensure you have at least 8GB RAM available
- Close other applications

### Slow Responses

First-time queries may be slow as models load. Subsequent queries are faster.

---

## How It Works

1. **Chunking**: Splits `knowledge_full.md` by `##` sections with 1600-char max per chunk
2. **Embedding**: Converts each chunk to a semantic vector using `nomic-embed-text`
3. **Retrieval**: Computes cosine similarity between your question and all chunks
4. **Ranking**: Returns top 6 most relevant chunks
5. **QA**: Passes retrieved chunks + question to `qwen2.5:32b` LLM for answer generation
6. **Safety**: Rejects answers if fewer than 2 chunks are highly relevant (similarity ≥ 0.3)

---

## More Information

See `README_RAG.md` for:
- Detailed architecture explanation
- Advanced configuration options
- Customization examples
- Performance optimization tips
- Known limitations & future improvements

---

## Quick Command Reference

```bash
# Activate environment
source rag_env/bin/activate

# Run setup (one-time)
python3 clockify_support_cli_final.py build knowledge_full.md

# Ask questions (repeatable)
python3 clockify_support_cli_final.py ask "Your question here"

# Interactive mode
python3 clockify_support_cli_final.py chat

# Help
python3 clockify_support_cli_final.py --help
```

---

**Happy querying!** 🚀
