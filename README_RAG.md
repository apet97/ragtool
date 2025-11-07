# Clockify RAG CLI Tool

A Retrieval-Augmented Generation (RAG) CLI tool that uses Clockify's Markdown documentation as a knowledge base with Ollama LLM for Q&A.

## Overview

This tool (`clockify_support_cli_final.py`) implements a production-ready RAG pipeline:

1. **Chunking**: Split the markdown documentation into manageable text chunks based on document structure
2. **Embedding**: Generate vector embeddings for all chunks using embedding models (local or via Ollama)
3. **Hybrid Retrieval**: Combine BM25 keyword search with dense semantic search
4. **Answer Generation**: Use LLM to generate answers from retrieved context with citation validation

**Note**: This README documents the current production CLI. For the legacy v1.0 simple implementation, see `clockify_rag.py` (deprecated).

## Architecture

### Chunking Strategy

- Splits documentation by second-level Markdown headings (`##`)
- Enforces a maximum chunk size of 1600 characters
- For oversized chunks, applies 200-character overlap between sub-chunks to preserve context at boundaries
- Assigns unique IDs to each chunk for reference

### Embedding & Storage

- Supports two embedding backends:
  - **Local**: SentenceTransformer models (offline, no external dependency)
  - **Ollama**: `nomic-embed-text` via Ollama API (configurable endpoint)
- Stores normalized embeddings in `vecs_n.npy` (NumPy array format)
- Stores metadata (chunk IDs, titles, sections) in `meta.jsonl`
- Optional FAISS ANN index for fast retrieval on large knowledge bases
- BM25 index for keyword-based retrieval (`bm25.json`)

### Hybrid Retrieval & Answering

- **Hybrid retrieval**: Combines BM25 (keyword) + dense semantic search with configurable blending
- **Query expansion**: Automatic synonym expansion for better recall
- **MMR reranking**: Maximal Marginal Relevance to diversify results and reduce redundancy
- **Optional LLM reranking**: Use LLM to rerank candidates for improved relevance
- Retrieves top-K candidates (default: 12), packs top-N snippets (default: 6)
- Applies similarity threshold (≥ 0.30) to ensure answer quality
- **Citation validation**: Validates that all citations reference actual context chunks
- **Confidence scoring**: LLM provides confidence score (1-5) for each answer
- Uses `qwen2.5:32b` model via Ollama's chat API for final answer generation
- Refuses to answer with "I don't know based on the MD." when context is insufficient

## Prerequisites

### System Requirements

- Python 3.7+
- Ollama instance (local or remote) - default: `http://127.0.0.1:11434`
- Required Python packages: `requests`, `numpy`, `sentence-transformers` (optional, for local embeddings)

### Ollama Setup

**Option 1: Local Ollama**

1. Install Ollama from https://ollama.ai
2. Pull required models:
   ```bash
   ollama pull nomic-embed-text    # For embedding generation (if using Ollama backend)
   ollama pull qwen2.5:32b          # For LLM-based answering
   ```
3. Run Ollama server:
   ```bash
   ollama serve
   ```

**Option 2: Remote Ollama**

Set the `OLLAMA_URL` environment variable to point to your remote instance:
```bash
export OLLAMA_URL="http://your-ollama-host:11434"
```

Ensure the remote Ollama instance has the required models installed.

### Python Dependencies

Install required packages:

```bash
pip install -r requirements.txt
# Or manually:
pip install requests numpy sentence-transformers
```

**Note**: `sentence-transformers` is only needed if using local embedding backend (`--emb-backend local`).

## Usage

### 1. Build the Knowledge Base

Build the complete index (chunks, embeddings, BM25, FAISS) in one command:

```bash
python3 clockify_support_cli_final.py build knowledge_full.md
```

**Output**: Creates all required index files:
- `chunks.jsonl`: Text chunks with metadata
- `vecs_n.npy`: Normalized embedding vectors
- `meta.jsonl`: Chunk metadata (IDs, titles, sections, URLs)
- `bm25.json`: BM25 keyword index
- `index.meta.json`: Build metadata and versioning
- `faiss_ivf.index`: FAISS ANN index (if enabled)

### 2. Ask Questions (Single Query)

Query the knowledge base with a single question:

```bash
python3 clockify_support_cli_final.py ask "How do I track time in Clockify?"
```

**Options**:
- `--debug`: Show retrieval diagnostics and scores
- `--rerank`: Enable LLM-based reranking for better relevance
- `--topk N`: Retrieve top-K candidates (default: 12)
- `--pack N`: Pack top-N snippets into context (default: 6)
- `--threshold T`: Minimum similarity threshold (default: 0.30)
- `--json`: Output answer as JSON with metadata
- `--emb-backend {local,ollama}`: Choose embedding backend
- `--ann {faiss,none}`: Enable/disable FAISS ANN index

### 3. Interactive Chat Mode

Start an interactive REPL session:

```bash
python3 clockify_support_cli_final.py chat
```

Interactive commands:
- `:exit` - Exit the session
- `:debug` - Toggle debug mode
- `:help` - Show available commands

## Configuration

### Environment Variables

Configure the system via environment variables:

```bash
# Ollama endpoint (default: http://127.0.0.1:11434)
export OLLAMA_URL="http://your-ollama-host:11434"

# Model configuration
export GEN_MODEL="qwen2.5:32b"              # LLM for answer generation
export EMB_MODEL="nomic-embed-text"         # Embedding model (if using Ollama backend)

# Embedding backend: "local" (SentenceTransformer) or "ollama"
export EMB_BACKEND="local"

# Context budget in tokens
export CTX_BUDGET="2800"

# BM25 tuning
export BM25_K1="1.0"
export BM25_B="0.65"
```

### Script Constants

Advanced configuration in `clockify_support_cli_final.py`:

```python
# Chunking
CHUNK_CHARS = 1600           # Maximum characters per chunk
CHUNK_OVERLAP = 200          # Character overlap between sub-chunks

# Retrieval
DEFAULT_TOP_K = 12           # Candidates to retrieve
DEFAULT_PACK_TOP = 6         # Snippets to pack into context
DEFAULT_THRESHOLD = 0.30     # Minimum similarity score

# Hybrid search
ALPHA_HYBRID = 0.5           # BM25 vs dense blend (0.5 = equal weight)
MMR_LAMBDA = 0.7             # Relevance vs diversity (0.7 = favor relevance)

# LLM
DEFAULT_NUM_CTX = 8192       # Context window size
DEFAULT_NUM_PREDICT = 512    # Max generation tokens
```

## File Structure

After running all commands, your directory will contain:

```
/Users/15x/Downloads/KBDOC/
├── clockify_rag.py          # Main CLI tool
├── knowledge_full.md        # Source documentation (6.9 MB)
├── chunks.jsonl             # Generated: chunked documentation
├── vecs.npy                 # Generated: embedding vectors
├── meta.jsonl               # Generated: chunk metadata
└── README_RAG.md            # This file
```

## How It Works

### Chunking Example

Input (knowledge_full.md):
```markdown
## Getting Started
This section covers initial setup...
...content...

## Time Tracking
Time tracking enables you to...
...more content...
```

Output (chunks.jsonl):
```json
{"id": 0, "text": "## Getting Started\nThis section covers initial setup...\n..."}
{"id": 1, "text": "## Time Tracking\nTime tracking enables you to...\n..."}
```

### Embedding & Retrieval Example

1. User asks: "How do I track time in Clockify?"
2. Question is embedded into a vector
3. Cosine similarity computed against all chunk vectors
4. Top 6 chunks retrieved (e.g., IDs: 15, 23, 8, 45, 12, 56)
5. Relevance check: If at least 2 have similarity ≥ 0.3, proceed
6. LLM receives system prompt + top chunks + question
7. LLM generates answer citing snippet IDs: "You can track time by [15, 23]..."

### Quality Assurance

The system implements several safety mechanisms:

- **Closed-Book Design**: LLM is instructed to use only provided snippets
- **Citation Requirements**: Model is prompted to cite sources with snippet IDs
- **Relevance Thresholding**: Low-confidence answers are rejected upfront
- **Offline Operation**: No external API calls; everything runs locally

## Troubleshooting

### Error: "Connection refused" when embedding

- Ensure Ollama is running: `ollama serve`
- Check URL configuration in the script
- Verify network connectivity to the Ollama endpoint

### Error: "Model not found"

```bash
# Check available models
ollama list

# Pull missing models
ollama pull nomic-embed-text
ollama pull qwen2.5:32b
```

### Memory Issues

- Reduce `CHUNK_SIZE` if memory is limited
- Ensure sufficient RAM for loading embeddings (typically a few hundred MB for moderate-size knowledge bases)

### Slow Performance

- Ensure Ollama models are cached locally (first run may download)
- Consider using a faster embedding model if available
- Reduce the number of top chunks retrieved (currently set to 6)

## Advanced Customization

### Adjusting Relevance Threshold

Change the threshold in `ask_question()`:

```python
high_score_count = sum(1 for score in top_scores if score >= 0.5)  # Increase from 0.3
```

Higher values require stricter relevance matching.

### Changing Retrieved Chunk Count

Modify the `ask_question()` function to retrieve more/fewer chunks:

```python
# Change from 6 to a different number
if sims.shape[0] < 12:
    top_indices = np.argsort(sims)[::-1]
else:
    top_indices = np.argpartition(sims, -12)[-12:]
```

### Custom System Prompt

Edit the `system_message` in `ask_question()` to modify LLM behavior:

```python
system_message = (
    "You are an expert Clockify documentation assistant. "
    "Always use the provided SNIPPETS. If unclear, say: 'I don't know based on the MD.' "
    "Format answers with snippet citations [id1, id2]."
)
```

## Dataset Regeneration

When refreshing evaluation datasets after rerunning the chunk builder:

1. Regenerate chunk metadata and produce a title → chunk index:

   ```bash
   python clockify_rag/chunking/build_chunks.py  # existing chunk build step
   ./scripts/generate_chunk_title_map.py         # writes chunk_title_map.json
   ```

   The helper script parses `chunks.jsonl` and outputs `chunk_title_map.json` so you can quickly search for chunks that mention a topic or article title.

2. Update `eval_dataset.jsonl` and `eval_datasets/clockify_v1.jsonl` so every query references at least one relevant UUID.

   ```bash
   # manually edit the JSONL files with the UUIDs surfaced in chunk_title_map.json
   # ensure no query is left with an empty relevant_chunk_ids list
   ```

3. Rebuild the knowledge base artifacts and run the evaluation suite to confirm the new UUIDs work end-to-end:

   ```bash
   make build     # refresh embeddings so eval.py can read the knowledge base
   python eval.py
   ```

   Running `python eval.py` without rebuilding first results in the "Knowledge base not built" error.

## Limitations & Future Improvements

- **Fixed Chunk Size**: Currently uses fixed 1600-char chunks; could use dynamic sizing based on semantic boundaries
- **No Reranking**: Retrieved chunks are ranked only by cosine similarity; could add cross-encoder reranking for better relevance
- **Single Model**: Uses one embedding and one chat model; could support multiple models for comparison
- **No Caching**: Each query re-embeds the question; could cache frequently asked questions
- **No User Feedback**: No mechanism to learn from user feedback on answer quality

## References

- [Ollama Documentation](https://ollama.ai)
- [nomic-embed-text Model](https://huggingface.co/nomic-ai/nomic-embed-text-v1)
- [Qwen 2.5 Model](https://huggingface.co/Qwen/Qwen2.5-32B)
- [Cosine Similarity for NLP](https://en.wikipedia.org/wiki/Cosine_similarity)
- [RAG Pattern](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)

## License

This tool is provided as-is for use with Clockify documentation.
