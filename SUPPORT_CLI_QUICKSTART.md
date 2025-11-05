# Clockify Support CLI – Quick Start (v2.0)

**Single-file, stateless support chatbot with hybrid retrieval**

---

## 60-Second Setup

### 1. Activate Environment
```bash
source rag_env/bin/activate
```

### 2. Build Index (one-time, ~5–10 min)
```bash
python3 clockify_support_cli.py build knowledge_full.md
```

**You'll see**:
```
======================================================================
BUILDING KNOWLEDGE BASE
======================================================================

[1/4] Parsing and chunking...
  Created 1247 chunks

[2/4] Embedding with Ollama...
  Embedded 50/1247...
  ...
  Saved (1247, 768) embeddings

[3/4] Building BM25 index...
  Indexed 12840 unique terms

[4/4] Done.
======================================================================
```

### 3. Start Chat
```bash
python3 clockify_support_cli.py chat
```

### 4. Ask Questions
```
> How do I manage team members?
[Answer with citations]

> How do I track time offline?
[Answer with citations]

> :debug
[DEBUG=ON]

> What are the pricing plans?
[Answer with debug diagnostics]

> :exit
[Done]
```

---

## What's Better Than v1.0?

✅ **Hybrid Retrieval**: BM25 (keywords) + Dense (semantics) = better accuracy
✅ **Single File**: All code in `clockify_support_cli.py`
✅ **Stateless REPL**: No history; each question starts fresh
✅ **Better Snippets**: Title + URL + section for context
✅ **MMR Diversification**: Reduces duplicate near-sections
✅ **Smart Dedup**: Groups similar content, avoids redundant answers

---

## Files Generated

| File | Purpose |
|------|---------|
| `chunks.jsonl` | Chunked knowledge base |
| `vecs.npy` | Dense embeddings |
| `bm25.json` | BM25 index (keywords) |
| `meta.jsonl` | Chunk metadata |

**Cleanup & rebuild**: `rm chunks.jsonl vecs.npy meta.jsonl bm25.json && python3 clockify_support_cli.py build knowledge_full.md`

---

## Common Tasks

### Debug Retrieval

```bash
python3 clockify_support_cli.py chat --debug

> How do I invite team members?

[Answer...]

---
[DEBUG] Selected chunks:
[
  {"id": "a7f2c...", "title": "Team Mgmt", "dense_score": 0.85},
  {"id": "b3e1d...", "title": "Invitations", "dense_score": 0.78},
  ...
]
```

### Change LLM Model

```bash
export GEN_MODEL="mistral:7b"
python3 clockify_support_cli.py chat
```

### Tune Retrieval

Edit constants in `clockify_support_cli.py`:

```python
MMR_LAMBDA = 0.8          # Higher = more relevance, less diversity
LOW_COS_THRESH = 0.20     # Lower = accept more marginal matches
PACK_TOP = 8              # More snippets = longer context
```

Then rebuild: `python3 clockify_support_cli.py build knowledge_full.md`

---

## Troubleshooting

### "Connection refused"

```bash
# In another terminal:
ollama serve

# Verify:
curl http://127.0.0.1:11434/api/tags
```

### "Model not found"

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:32b
```

### "I don't know based on the MD."

This is correct behavior when:
- The KB doesn't contain the information
- Less than 2 snippets matched with high confidence (cosine ≥ 0.30)

Check `:debug` to see what was retrieved.

### Low-Quality Answers

1. Try `:debug` to see selected chunks
2. Lower `LOW_COS_THRESH` from 0.30 to 0.20
3. Increase `MMR_LAMBDA` from 0.7 to 0.8
4. Verify the KB contains the info (search `knowledge_full.md`)

---

## Commands Reference

| Command | Meaning |
|---------|---------|
| `python3 clockify_support_cli.py build <path>` | Build/rebuild index |
| `python3 clockify_support_cli.py chat` | Start chat loop |
| `python3 clockify_support_cli.py chat --debug` | Chat with diagnostics |
| `:exit` | Quit chat loop |
| `:debug` | Toggle diagnostics on/off |

---

## Example Q&A

```
> What's the difference between Pro and Enterprise plans?

Pro plan includes:
- Up to 10 team members
- Advanced reporting
- Custom integrations

Enterprise plan adds:
- Unlimited team members
- Dedicated support
- SSO/SAML authentication
- White-label options

See plans & pricing [id_123, id_234].

> How do I set up integrations?

1. Go to Settings → Integrations
2. Choose your app (Slack, Jira, Google Calendar, etc.)
3. Click Connect
4. Authorize in the app's login screen
5. Confirm in Clockify

For custom API integrations, see the REST API documentation [id_456, id_567].

> :exit
```

---

## Architecture

```
INPUT
  ↓
RETRIEVE (hybrid: BM25 + dense embeddings)
  ├─ Embed query
  ├─ Score with BM25 (keywords)
  ├─ Score with cosine (semantics)
  ├─ Combine via z-score (60% dense, 40% BM25)
  ├─ Top-12 candidates
  ├─ Dedupe (title, section)
  └─ MMR diversify → Top-6
  ↓
CHECK COVERAGE
  ├─ At least 2 snippets with cosine ≥ 0.30?
  └─ If no → answer "I don't know based on the MD."
  ↓
PACK SNIPPETS
  └─ Format: [id | title | section]\nURL\ntext
  ↓
CALL LLM
  ├─ System: "Closed-book support assistant"
  ├─ User: "SNIPPETS:\n{...}\n\nQUESTION:\n{user_q}"
  └─ Model: qwen2.5:32b (temp=0)
  ↓
OUTPUT
  └─ Answer with citations [id1, id2, ...]
  ↓
FORGET
  └─ No history; next turn starts fresh
```

---

## Performance

| Task | Time |
|------|------|
| Build (chunk + embed + BM25) | 5–10 min |
| Load index | <1 sec |
| Per query | 10–20 sec |
| Memory (loaded) | ~500 MB |

---

## Next Steps

- Read full docs: `CLOCKIFY_SUPPORT_CLI_README.md`
- Explore system prompts: Edit `SYSTEM_PROMPT` in the script
- Add reranking: See "Development & Extension" in full docs
- Deploy to team: Share script + built artifacts

---

**Status**: ✅ Production Ready
**Version**: 2.0
**Date**: 2025-11-05
