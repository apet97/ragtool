# Clockify Support CLI – Hybrid Retrieval Edition

**Status**: ✅ Production Ready
**Version**: 2.0 (Enhanced from v1.0 with hybrid retrieval)
**Location**: `/Users/15x/Downloads/KBDOC/clockify_support_cli.py`

---

## Overview

A **single-file, stateless support chatbot** for Clockify's internal knowledge base. Runs entirely offline against a local Ollama endpoint with **hybrid retrieval** (BM25 + semantic embeddings) for improved accuracy.

### Key Improvements Over v1.0

| Feature | v1.0 (clockify_rag.py) | v2.0 (clockify_support_cli.py) |
|---------|------------------------|------------------------------|
| Retrieval | Cosine similarity only | Hybrid (BM25 + dense) + MMR |
| Deduplication | None | Smart (title + section) |
| Diversification | None | MMR (Maximal Marginal Relevance) |
| State | Multi-turn history | Stateless (no memory) |
| CLI | 3 separate commands | 2 unified commands |
| Format | Multi-file setup | Single file |
| Snippets | Basic text | Structured (title + URL + section) |

---

## Architecture

### Pipeline

```
INPUT (user question)
  ↓
[RETRIEVE]
  ├─ Dense embedding (nomic-embed-text)
  ├─ BM25 sparse search (term frequency)
  ├─ Hybrid scoring (60% dense + 40% BM25, z-score normalized)
  ├─ Top-12 candidates
  ├─ Dedupe by (title, section)
  ├─ MMR diversification → Top-6
  └─ Coverage check (min 2 chunks @ cosine ≥ 0.30)
  ↓
[PACK SNIPPETS]
  ├─ Respect 2800-token context budget
  ├─ Format: [id | title | section] + URL + text
  └─ Return chunk IDs for citation
  ↓
[CALL LLM]
  ├─ System prompt: "Closed-book support assistant"
  ├─ Message: SNIPPETS + QUESTION
  └─ Model: qwen2.5:32b (temp=0, deterministic)
  ↓
OUTPUT (answer with citations)
  ├─ If coverage OK: LLM answer [id1, id2, ...]
  └─ If coverage low: "I don't know based on the MD."
  ↓
FORGET (stateless)
  └─ No history kept; next turn starts fresh
```

### Why Hybrid Retrieval?

- **BM25 (sparse)**: Catches exact keyword matches (e.g., "Bundle seats", "API endpoint")
- **Dense (semantic)**: Catches paraphrased questions and concept matches
- **Combined z-score**: Normalizes both signals; 60/40 weighted toward dense
- **MMR**: Reduces redundant near-duplicate snippets, improves diversity
- **Coverage guard**: Refuses low-confidence answers upfront

---

## Installation & Setup

### 0. Activate Environment

```bash
source rag_env/bin/activate
# or manually:
python3 -m venv venv && source venv/bin/activate
pip install numpy requests
```

### 1. Build Index (One-Time, ~5–10 minutes)

```bash
python3 clockify_support_cli.py build knowledge_full.md
```

**Output**:
- `chunks.jsonl` – Chunked documents (id, title, url, section, text)
- `vecs.npy` – Dense embeddings (numpy float32 array)
- `meta.jsonl` – Metadata for each chunk
- `bm25.json` – BM25 statistics (idf, doc lengths, term frequencies)

**Progress**:
```
======================================================================
BUILDING KNOWLEDGE BASE
======================================================================

[1/4] Parsing and chunking...
  Created 1247 chunks

[2/4] Embedding with Ollama...
  Embedded 50/1247...
  Embedded 100/1247...
  ...
  Saved (1247, 768) embeddings

[3/4] Building BM25 index...
  Indexed 12840 unique terms

[4/4] Done.
======================================================================
```

### 2. Start Chat Loop

```bash
python3 clockify_support_cli.py chat
# or with debug output:
python3 clockify_support_cli.py chat --debug
```

**REPL Commands**:
- `<question>` → Retrieve, answer, forget
- `:debug` → Toggle retrieval diagnostics on/off
- `:exit` → Quit

---

## Example Session

```
======================================================================
CLOCKIFY SUPPORT – Local, Stateless
======================================================================
Type a question, press Enter. Commands:
  :exit    – Quit
  :debug   – Toggle retrieval diagnostics
======================================================================

> How do I manage team members?

To manage team members in Clockify, you can:

1) Invite members: Go to Project Settings → Team → Invite via email
2) Set permissions: Assign roles (Admin, Manager, User) with appropriate access levels
3) Remove members: Click the "x" icon next to their name in the team list
4) View activity: Monitor time entries and project participation in the Admin Panel

Notes by role:
- Admins can create/edit projects and manage all team settings.
- Managers can oversee their assigned projects and team members.
- Users can only track time and view their own data.

For API integrations with your team endpoint, refer to documentation on region-specific API endpoints [id_123, id_456].

---
[DEBUG] Selected chunks:
[
  {
    "id": "a7f2c...",
    "title": "Team Management",
    "section": "## Team Settings",
    "url": "https://internal.kb/clockify/team-mgmt",
    "dense_score": 0.85
  },
  {
    "id": "b3e1d...",
    "title": "Roles and Permissions",
    "section": "## Permission Levels",
    "url": "https://internal.kb/clockify/roles",
    "dense_score": 0.78
  },
  ...
]

> How do I track time offline?

Clockify supports offline time tracking:

1) Mobile App: Enable offline mode in settings. Entries sync when reconnected.
2) Web: Use local storage caching (if available in your plan).
3) Desktop Timer: The timer continues running even if your network disconnects.

Recovery: When online again, all offline entries auto-sync without data loss.

Plan-specific notes:
- Free plan: Limited offline history
- Pro/Enterprise: Full offline support with unlimited sync

[id_789, id_234]

> :exit
```

---

## Configuration

### Environment Variables

Set before running to override defaults:

```bash
export OLLAMA_URL="http://10.127.0.192:11434"
export GEN_MODEL="qwen2.5:32b"
export EMB_MODEL="nomic-embed-text"
```

### Query Expansion Overrides (Domain Synonyms)

Query expansion terms now live in `config/query_expansions.json`. The file is a JSON object where each key is a lower-case term and each value is a list of synonyms that should be appended to the BM25 query.

```json
{
  "track": ["log", "record", "enter", "add"],
  "offline": ["no internet", "no connection"]
}
```

**How to override:**

* **CLI flag** – pass a path to a custom JSON file when launching the tool:

  ```bash
  python3 clockify_support_cli_final.py chat --query-expansions /path/to/team_expansions.json
  ```

* **Environment variable** – point `CLOCKIFY_QUERY_EXPANSIONS` to your JSON file:

  ```bash
  export CLOCKIFY_QUERY_EXPANSIONS=/path/to/team_expansions.json
  python3 clockify_support_cli_final.py chat
  ```

Each launch validates the JSON (structure + readability). If validation fails the CLI exits with a clear `CONFIG ERROR`, so dom
ain teams can fix issues before queries run.

### Code Constants (Edit in clockify_support_cli.py)

```python
CHUNK_CHARS = 1600          # Max chars per chunk
CHUNK_OVERLAP = 200         # Overlap between sub-chunks
TOP_K = 12                  # Top candidates before dedup
PACK_TOP = 6                # Final snippets to send to LLM
MMR_LAMBDA = 0.7            # 0.7 = favor relevance, 0.3 = favor diversity
LOW_COVERAGE_MIN = 2        # Min relevant snippets required
LOW_COS_THRESH = 0.30       # Min cosine similarity score
CTX_TOKEN_BUDGET = 2800     # Max tokens for snippets block
```

### System Prompts

**System Prompt** (controls agent behavior):
```
You are CAKE.com Internal Support for Clockify.
Closed-book. Only use SNIPPETS. If info is missing, reply exactly:
"I don't know based on the MD."
Rules:
- Answer in the user's language.
- Be precise. No speculation. No external info. No web search.
- Structure:
  1) Direct answer
  2) Steps
  3) Notes by role/plan/region if relevant
  4) Citations: list the snippet IDs you used, like [id1, id2], and include URLs in-line if present.
- If SNIPPETS disagree, state the conflict and offer safest interpretation.
```

**User Message Template**:
```
SNIPPETS:
{packed_snippets}

QUESTION:
{user_question}

Answer with citations like [id1, id2].
```

---

## Performance & Tuning

### Baseline Metrics

| Operation | Time | Notes |
|-----------|------|-------|
| Build (chunk + embed + BM25) | 5–10 min | Depends on KB size; first embed slower |
| Query (retrieve + LLM) | 10–20 sec | Includes Ollama network latency |
| Memory (loaded index) | ~500 MB | vecs.npy + chunks + models in Ollama |

### Optimization Tips

**Faster queries**:
- Reduce `PACK_TOP` from 6 to 4
- Lower `CTX_TOKEN_BUDGET` from 2800 to 2000
- Use `EMB_MODEL = "all-minilm-l6-v2"` (smaller, faster)

**Better accuracy**:
- Increase `MMR_LAMBDA` from 0.7 to 0.8 (more relevance, less diversity)
- Lower `LOW_COS_THRESH` from 0.30 to 0.20 (accept more marginal matches)
- Increase `PACK_TOP` from 6 to 8

**Memory-efficient**:
- Reduce `CHUNK_CHARS` from 1600 to 800
- Fewer chunks = smaller vecs.npy

---

## Troubleshooting

### "Connection refused" Error

```
Embedding request failed: Connection refused
```

**Solution**:
```bash
ollama serve  # Start Ollama in another terminal
# Verify:
curl http://10.127.0.192:11434/api/tags
```

### "Model not found" Error

```
Embedding API returned error 404
```

**Solution**:
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:32b
ollama list  # Verify both are present
```

### Out of Memory During Build

**Solution**:
1. Reduce chunk size: `CHUNK_CHARS = 800`
2. Re-run build
3. Or: close other applications, ensure 8 GB+ RAM

### Slow Queries

**Likely causes**:
- First query after start: models loading (normal, ~20 sec)
- Embedding service lagging: check Ollama logs
- Large KB: consider splitting knowledge_full.md

**Mitigation**:
- Pre-warm Ollama: send a dummy request after start
- Reduce `CTX_TOKEN_BUDGET` to pack fewer snippets

### Low-Quality Answers

**Symptoms**: Answers are vague or unrelated.

**Causes**:
- Low coverage threshold reached ("I don't know..." replies)
- Snippets don't match the question well

**Solutions**:
1. Check `:debug` output – are selected chunks relevant?
2. Lower `LOW_COS_THRESH` from 0.30 to 0.20
3. Increase `MMR_LAMBDA` from 0.7 to 0.8 (favor relevance over diversity)
4. Verify KB contains the information (search `knowledge_full.md` manually)

---

## Comparison: v1.0 vs v2.0

### v1.0 (clockify_rag.py)

**Pros**:
- Simple, easy to understand
- Works well for direct keyword + semantic matches

**Cons**:
- Cosine-only retrieval misses exact-match keywords
- No deduplication of similar snippets
- Multi-file setup with separate commands
- Multi-turn history (not always desired for support bot)

### v2.0 (clockify_support_cli.py)

**Pros**:
- Hybrid retrieval (BM25 + dense) catches both keywords and concepts
- MMR diversification reduces redundant snippets
- Single file, simpler deployment
- Stateless REPL (fresh context per question)
- Better for internal support agent use case
- Structured snippet format (title + URL + section)

**Cons**:
- Slightly more complex to understand
- Slightly longer startup time (loads BM25 stats)

**Recommendation**: Use v2.0 for support applications; v1.0 if simplicity is more important.

---

## Files Generated

After running `build`:

```
/Users/15x/Downloads/KBDOC/
├── clockify_support_cli.py   (main script)
├── knowledge_full.md         (source KB)
├── chunks.jsonl              (chunked docs, one JSON per line)
├── vecs.npy                  (numpy embedding array)
├── meta.jsonl                (chunk metadata)
└── bm25.json                 (BM25 index)
```

**Sizes**:
- `chunks.jsonl` – ~5–10 MB (depends on KB)
- `vecs.npy` – ~3–5 MB (n_chunks × 768 dimensions)
- `meta.jsonl` – ~2–5 MB
- `bm25.json` – ~1–2 MB

**Cleanup**:
```bash
rm chunks.jsonl vecs.npy meta.jsonl bm25.json
python3 clockify_support_cli.py build knowledge_full.md  # Rebuild
```

---

## Development & Extension

### Adding Reranking (Optional)

To add LLM-based reranking after initial retrieval:

```python
def rerank_with_llm(question, selected_chunks, vecs):
    """Optional: Score top-12 with LLM for precision."""
    # Build passage list
    passages = [
        {"id": chunks[i]["id"], "text": chunks[i]["text"]}
        for i in selected
    ]
    # Send to LLM with rerank prompt
    # Parse JSON response with scores
    # Return top-6 by score
    # (Implementation left as exercise)
    pass
```

### Custom System Prompts

Edit `SYSTEM_PROMPT` in the script to change agent behavior:

```python
SYSTEM_PROMPT = """You are a Clockify troubleshooting expert.
...[your prompt]..."""
```

### Different LLM Models

```bash
export GEN_MODEL="mistral:7b"
python3 clockify_support_cli.py chat
```

---

## Testing & Validation

### Acceptance Criteria

After build + chat, verify:

- [ ] Query returns answer with snippet citations [id1, id2]
- [ ] `:debug` shows selected chunks with scores
- [ ] Questions not in KB return exactly: "I don't know based on the MD."
- [ ] No prior-turn memory: same question asked twice → same answer, fresh context
- [ ] Stateless: error on turn 1 does not affect turn 2

### Sample Test Queries

```
> How do I set up a project?
> What are the different pricing plans?
> Can I track time offline?
> How do I invite team members?
> What is time rounding?
> Do you support SAML/SSO?
> [Unknown topic]
```

---

## Rollout & Deployment

### For Internal Support

1. Build index once:
   ```bash
   python3 clockify_support_cli.py build knowledge_full.md
   ```

2. Share script + built artifacts with team:
   ```bash
   # All files needed for chat:
   clockify_support_cli.py
   chunks.jsonl
   vecs.npy
   meta.jsonl
   bm25.json
   ```

3. Each user runs:
   ```bash
   source venv/bin/activate
   python3 clockify_support_cli.py chat
   ```

### Requirements for Rollout

- Python 3.7+
- numpy, requests (in venv)
- Access to internal Ollama (http://10.127.0.192:11434)
- ~500 MB disk space for built artifacts

---

## References

- **Ollama API**: https://ollama.ai
- **nomic-embed-text**: Semantic embeddings model
- **qwen2.5:32b**: Instruction-tuned LLM
- **BM25**: Classical information retrieval ranking
- **MMR (Maximal Marginal Relevance)**: Diversity-aware ranking (Carbonell & Goldstein, 1998)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2025-11-05 | Hybrid retrieval, MMR, stateless REPL, single file |
| 1.0 | 2025-11-05 | Initial cosine-similarity RAG with multi-file setup |

---

**Status**: ✅ Production Ready
**Maintained by**: Clockify Internal
**Contact**: [Support]
**Last Updated**: 2025-11-05
