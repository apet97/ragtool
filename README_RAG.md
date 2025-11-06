# Clockify RAG CLI Tool

A local, offline Retrieval-Augmented Generation (RAG) CLI tool that uses Clockify's Markdown documentation as a knowledge base with a local Ollama LLM instance for Q&A.

## Overview

This tool (`clockify_rag.py`) implements a three-stage pipeline:

1. **Chunking**: Split the markdown documentation into manageable text chunks based on document structure
2. **Embedding**: Generate vector embeddings for all chunks using a local embedding model (nomic-embed-text)
3. **Retrieval & QA**: Answer user questions by retrieving relevant chunks and querying a local LLM

## Architecture

### Chunking Strategy

- Splits documentation by second-level Markdown headings (`##`)
- Enforces a maximum chunk size of 1600 characters
- For oversized chunks, applies 200-character overlap between sub-chunks to preserve context at boundaries
- Assigns unique IDs to each chunk for reference

### Embedding & Storage

- Uses Ollama's `nomic-embed-text` model for semantic vector generation
- Stores embeddings in `vecs.npy` (NumPy array format)
- Stores metadata (chunk IDs and text) in `meta.jsonl` for retrieval

### Retrieval & Answering

- Embeds user questions with the same embedding model
- Uses cosine similarity to rank chunks by relevance
- Retrieves top 6 most relevant chunks
- Applies a relevance threshold (similarity ≥ 0.3) to ensure answer quality
- If fewer than 2 chunks meet the threshold, responds with "I don't know based on the MD."
- Constructs a system prompt instructing the LLM to be a "closed-book assistant" using only retrieved snippets
- Uses local `qwen2.5:32b` model via Ollama's chat API for final answer generation

## Prerequisites

### System Requirements

- Python 3.7+
- Local Ollama instance running on `http://10.127.0.192:11434` (adjust URL in script if different)
- Required Python packages: `requests`, `numpy`

### Ollama Setup

Ensure the following models are available locally:

```bash
ollama pull nomic-embed-text    # For embedding generation
ollama pull qwen2.5:32b          # For LLM-based answering
```

Run Ollama server:

```bash
ollama serve
```

### Python Dependencies

```bash
pip install requests numpy
```

## Usage

### 1. Chunk the Documentation

Split the `knowledge_full.md` file into chunks:

```bash
python3 clockify_rag.py chunk
```

**Output**: Creates `chunks.jsonl` with one JSON object per line, each containing:
- `id`: Unique chunk identifier
- `text`: Chunk content

### 2. Generate Embeddings

Embed all chunks using the local embedding model:

```bash
python3 clockify_rag.py embed
```

**Output**:
- `vecs.npy`: NumPy array of embedding vectors
- `meta.jsonl`: Metadata file with chunk IDs and text content

### 3. Ask Questions

Query the knowledge base:

```bash
python3 clockify_rag.py ask "How do I track time in Clockify?"
```

**Output**: An answer from the LLM with citations to relevant snippets, or "I don't know based on the MD." if the information isn't found.

## Configuration

Edit the following constants in `clockify_rag.py` to customize behavior:

```python
CHUNK_SIZE = 1600          # Maximum characters per chunk
CHUNK_OVERLAP = 200        # Character overlap between sub-chunks
OLLAMA_URL = "http://10.127.0.192:11434"  # Ollama server URL
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:32b"
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
