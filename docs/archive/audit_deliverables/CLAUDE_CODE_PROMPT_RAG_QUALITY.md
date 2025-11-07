# Claude Code Prompt: RAG Quality Deep Dive

**Copy this for specialized RAG effectiveness analysis**

---

## RAG-FOCUSED PROMPT

```
Role: RAG Systems Specialist conducting deep analysis of retrieval-augmented generation quality.

Objective: Analyze this RAG tool's effectiveness end-to-end and identify improvements to retrieval accuracy, answer quality, and user satisfaction.

## Analysis Scope

Read and trace the complete RAG pipeline:
1. Document ingestion → chunking
2. Chunk embedding → indexing
3. Query processing → retrieval
4. Context packing → LLM prompting
5. Answer generation → citation

## Key Files to Analyze
- Main implementation (clockify_support_cli*.py)
- Knowledge base (knowledge_full.md) - sample only
- Test/benchmark scripts (scripts/*.sh)
- Documentation (CLAUDE.md, README*.md)

## RAG Quality Checklist

### 1. Chunking Strategy
- [ ] Chunk size appropriate for content type?
- [ ] Overlap preserves context at boundaries?
- [ ] Respects semantic boundaries (paragraphs, sections)?
- [ ] Metadata preserved (section titles, hierarchy)?
- [ ] Special handling for code/tables/lists?

### 2. Embedding Quality
- [ ] Model choice appropriate (domain, dimension)?
- [ ] Normalization applied correctly?
- [ ] Embedding cache for unchanged content?
- [ ] Handles edge cases (empty text, special chars)?

### 3. Retrieval Effectiveness
- [ ] Hybrid search implemented (sparse + dense)?
- [ ] BM25 parameters tuned for domain (k1, b)?
- [ ] Dense scoring uses appropriate metric (cosine vs. dot)?
- [ ] Diversity mechanism (MMR) prevents redundancy?
- [ ] Retrieval count (top_k) well-calibrated?
- [ ] Score normalization consistent?

### 4. Reranking
- [ ] Reranker model appropriate?
- [ ] Fallback if reranking fails?
- [ ] Cost-benefit of reranking justified?

### 5. Context Packing
- [ ] Token budget management accurate?
- [ ] Prioritization strategy clear (top-K, threshold)?
- [ ] Context fits within LLM window?
- [ ] Truncation strategy preserves meaning?
- [ ] Formatting aids LLM comprehension?

### 6. Prompt Engineering
- [ ] System prompt clear and specific?
- [ ] Instructions to stay grounded in context?
- [ ] Citation format enforced?
- [ ] Refusal logic for low-confidence?
- [ ] Few-shot examples if needed?

### 7. Answer Quality
- [ ] Hallucination prevention mechanisms?
- [ ] Citation accuracy verified?
- [ ] Coverage check (min chunks required)?
- [ ] Answer completeness vs. context?
- [ ] Formatting user-friendly?

### 8. Evaluation & Metrics
- [ ] Ground truth Q&A pairs?
- [ ] Retrieval metrics (MRR, NDCG, Recall@K)?
- [ ] Answer quality metrics (correctness, citation)?
- [ ] Latency/cost tracking?
- [ ] A/B test framework?

## Deliverables

### A) RAG_QUALITY_REPORT.md

```markdown
# RAG Quality Analysis

## Executive Summary
- Overall RAG effectiveness: X/10
- Top 3 strengths
- Top 5 quality improvements needed

## Pipeline Trace
[Diagram or text trace of data flow from question → answer]

## Component Scores
1. Chunking: X/10
2. Embedding: X/10
3. Retrieval: X/10
4. Reranking: X/10
5. Context Packing: X/10
6. Prompting: X/10
7. Answer Quality: X/10
8. Evaluation: X/10

## Detailed Findings

### Chunking Analysis
- Current strategy: [description]
- Strengths: [list]
- Weaknesses: [list]
- Recommended changes: [specific]

[Repeat for each component]

## RAG Best Practices Comparison
| Practice | Implemented? | Quality | Notes |
|----------|--------------|---------|-------|
| Semantic chunking | ❌ | - | Uses fixed char count |
| Hybrid search | ✅ | 8/10 | BM25 + dense + MMR |
| Query expansion | ❌ | - | Missing |
| Cross-encoder rerank | ✅ | 7/10 | LLM-based, expensive |
| Citation tracking | ✅ | 9/10 | IDs preserved |
| Hallucination detection | ⚠️ | 5/10 | Simple coverage check |
| Evaluation suite | ❌ | - | No ground truth |

## Top 10 RAG Quality Improvements
1. [Specific improvement with code example]
2. [...]

## Suggested Experiments
1. Tune BM25 k1 parameter (test k1 ∈ {1.0, 1.2, 1.5, 2.0})
2. Increase chunk overlap from 200 to 400 chars
3. Add query expansion (synonyms, acronyms)
4. Test sentence-level chunking vs. fixed-char
5. Implement cross-encoder for final rerank
6. Add answer self-consistency check (3x generation)

## Evaluation Recommendations
- Create 50 ground truth Q&A pairs across difficulty levels
- Measure Recall@10, MRR@10, NDCG@10 for retrieval
- Measure answer correctness (human eval or LLM-as-judge)
- Measure citation accuracy (precision/recall)
- Track latency P50/P95/P99
```

### B) RAG_EXPERIMENTS.jsonl

Suggested experiments in structured format:

```jsonl
{"experiment":"tune_bm25_k1","hypothesis":"Lower k1 improves precision for technical docs","param":"BM25_K1","values":[1.0,1.2,1.5,2.0],"metric":"NDCG@10","baseline":0.75,"expected_improvement":0.05,"effort_hours":2,"risk":"LOW"}
{"experiment":"increase_chunk_overlap","hypothesis":"More overlap reduces boundary information loss","param":"CHUNK_OVERLAP","values":[200,300,400],"metric":"answer_completeness","baseline":"?","expected_improvement":"10%","effort_hours":1,"risk":"LOW"}
{"experiment":"semantic_chunking","hypothesis":"Sentence boundaries preserve context better","implementation":"Use sentence tokenizer instead of char count","metric":"chunk_coherence","baseline":"?","expected_improvement":"20%","effort_hours":8,"risk":"MEDIUM"}
```

### C) RAG_QUICK_WINS.md

Top 5 improvements with <2 hour implementation:
1. Tune BM25 k1 parameter
2. Increase chunk overlap
3. Add query preprocessing (lowercase, strip)
4. Improve citation formatting
5. Add confidence scores to answers

## Analysis Approach

1. **Trace a Query End-to-End**
   - Pick example: "How do I track time in Clockify?"
   - Trace through chunk → embed → retrieve → pack → prompt → answer
   - Identify bottlenecks and failure points

2. **Sample Evaluation**
   - Create 5-10 test questions manually
   - Run through system
   - Evaluate retrieved chunks (relevance, coverage)
   - Evaluate answers (correctness, citations)

3. **Code Deep Dive**
   - Read retrieval logic carefully (BM25, dense, hybrid, MMR)
   - Read prompting logic (system prompt, context format)
   - Read validation logic (coverage check, refusal)

4. **Best Practices Comparison**
   - Compare to LlamaIndex patterns
   - Compare to LangChain patterns
   - Compare to research papers (RAG, retrieval, prompting)

5. **Gap Analysis**
   - What's missing vs. SOTA?
   - What's implemented but suboptimal?
   - What's over-engineered?

## Success Criteria

✅ Complete RAG pipeline traced and documented
✅ Each component scored (1-10) with justification
✅ Top 10 quality improvements identified
✅ 5 quick wins with code snippets
✅ Suggested experiments with success metrics
✅ Comparison to best practices

Begin analysis. Start by reading the main RAG implementation file and tracing one query end-to-end.
```

## PROMPT END

---

## When to Use This Prompt

- 🎯 Focused on **RAG effectiveness**, not general code quality
- 🎯 Want to improve **retrieval accuracy** and **answer quality**
- 🎯 Need **experiment suggestions** with metrics
- 🎯 Building **evaluation framework**
- 🎯 Comparing against **RAG best practices**

## What You'll Get

1. **RAG_QUALITY_REPORT.md**: Deep dive into each pipeline component
2. **RAG_EXPERIMENTS.jsonl**: Structured A/B test suggestions
3. **RAG_QUICK_WINS.md**: Fast quality improvements

## Example Follow-Up Actions

After receiving the report:

```bash
# Run suggested experiments
python3 clockify_support_cli_final.py --bm25-k1 1.2 ask "test query"

# Create ground truth dataset
# (Use findings to identify edge cases)

# Implement top quick win
# (Usually tuning hyperparameters)
```

---

## Complementary Prompts

- **General Quality**: Use `CLAUDE_CODE_PROMPT_COMPREHENSIVE.md`
- **Quick Check**: Use `CLAUDE_CODE_PROMPT_SIMPLE.md`
- **RAG Focus**: Use this prompt (RAG_QUALITY.md) ✅

---

**Time**: ~45 minutes
**Output**: 3 documents focused on RAG quality
**Best For**: ML/RAG engineers optimizing retrieval and answer quality
