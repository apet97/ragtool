# Claude Code Analysis Prompts

**Three optimized prompts for analyzing this RAG tool codebase in new Claude Code sessions**

---

## 📋 Quick Selection Guide

| Prompt | Time | Focus | Output | Best For |
|--------|------|-------|--------|----------|
| **[SIMPLE](./CLAUDE_CODE_PROMPT_SIMPLE.md)** | 30 min | Top 10 actionable improvements | 1 doc | Quick health check, fast iteration |
| **[COMPREHENSIVE](./CLAUDE_CODE_PROMPT_COMPREHENSIVE.md)** | 1-2 hours | Full codebase audit (all aspects) | 4 docs | Production readiness, deep review |
| **[RAG_QUALITY](./CLAUDE_CODE_PROMPT_RAG_QUALITY.md)** | 45 min | RAG effectiveness & ML pipeline | 3 docs | Retrieval accuracy, answer quality |

---

## 🚀 Prompt Details

### 1. Simple Analysis (CLAUDE_CODE_PROMPT_SIMPLE.md)

**⏱️ Time**: ~30 minutes
**📊 Output**: 1 document (TOP_10_IMPROVEMENTS.md)
**🎯 Focus**: Quick wins, critical issues

**Use when you want**:
- ✅ Fast feedback cycle
- ✅ Top 10 actionable items only
- ✅ No need for comprehensive docs
- ✅ Minimal time investment

**You'll get**:
- Top 10 improvements ranked by ROI
- File/line references
- Impact & effort estimates
- Code snippets for fixes
- 3-5 sentence quality summary

**Example output**:
```markdown
# Top 10 Improvements

1. **Cache embeddings for incremental builds** (HIGH impact, 2 hours)
   - File: clockify_support_cli_final.py:850
   - Impact: 50% faster rebuilds
   - Fix: Add emb_cache.jsonl with content hashing
   ...
```

---

### 2. Comprehensive Analysis (CLAUDE_CODE_PROMPT_COMPREHENSIVE.md)

**⏱️ Time**: ~1-2 hours
**📊 Output**: 4 documents (ANALYSIS_REPORT.md, IMPROVEMENTS.jsonl, QUICK_WINS.md, ARCHITECTURE_VISION.md)
**🎯 Focus**: Everything - correctness, performance, security, architecture, RAG quality, DX

**Use when you want**:
- ✅ Complete production readiness assessment
- ✅ File-by-file analysis
- ✅ Detailed documentation of all findings
- ✅ Long-term improvement roadmap
- ✅ Programmatic data (JSONL format)

**You'll get**:
- **ANALYSIS_REPORT.md**: File-by-file review, category scores, top 20 improvements
- **IMPROVEMENTS.jsonl**: Structured data (rank, category, impact, effort, code, rationale)
- **QUICK_WINS.md**: Top 10 improvements under 30 min each
- **ARCHITECTURE_VISION.md**: Long-term refactoring roadmap

**Analysis areas**:
1. RAG-specific quality (retrieval, chunking, prompting)
2. Performance & scalability
3. Correctness & reliability
4. Code quality & maintainability
5. Security & safety
6. Developer experience
7. RAG pipeline improvements (advanced techniques)

**Example IMPROVEMENTS.jsonl entry**:
```json
{
  "rank": 1,
  "category": "RAG",
  "subcategory": "retrieval",
  "issue": "BM25 k1 parameter not tuned for domain",
  "impact": "HIGH",
  "effort": "LOW",
  "file": "clockify_support_cli_final.py",
  "line": 935,
  "current": "k1=1.5 (default)",
  "proposed": "k1=1.2 (better for technical docs)",
  "rationale": "Technical documentation has different term frequency...",
  "implementation": "Add BM25_K1 env var, tune on sample queries",
  "expected_gain": "5-10% retrieval accuracy improvement",
  "references": ["https://arxiv.org/abs/1803.08988"]
}
```

---

### 3. RAG Quality Deep Dive (CLAUDE_CODE_PROMPT_RAG_QUALITY.md)

**⏱️ Time**: ~45 minutes
**📊 Output**: 3 documents (RAG_QUALITY_REPORT.md, RAG_EXPERIMENTS.jsonl, RAG_QUICK_WINS.md)
**🎯 Focus**: RAG effectiveness - retrieval accuracy, answer quality, pipeline optimization

**Use when you want**:
- ✅ Improve retrieval precision/recall
- ✅ Optimize answer quality
- ✅ Experiment suggestions with metrics
- ✅ RAG best practices comparison
- ✅ Evaluation framework design

**You'll get**:
- **RAG_QUALITY_REPORT.md**:
  - Pipeline trace (question → answer)
  - Component scores (chunking: 8/10, retrieval: 7/10, etc.)
  - Best practices comparison table
  - Top 10 RAG-specific improvements
  - Suggested experiments

- **RAG_EXPERIMENTS.jsonl**:
  - A/B test suggestions
  - Hypothesis, metrics, expected gains
  - Risk assessment

- **RAG_QUICK_WINS.md**:
  - 5 improvements under 2 hours
  - Hyperparameter tuning suggestions

**RAG Quality Checklist**:
- ✅ Chunking strategy (size, overlap, boundaries)
- ✅ Embedding quality (model, normalization, caching)
- ✅ Retrieval effectiveness (BM25 tuning, hybrid search, MMR)
- ✅ Reranking (cross-encoder, fallback)
- ✅ Context packing (token budget, prioritization)
- ✅ Prompt engineering (grounding, citations, refusal)
- ✅ Answer quality (hallucination prevention, completeness)
- ✅ Evaluation (metrics, ground truth, A/B tests)

**Example experiment suggestion**:
```json
{
  "experiment": "tune_bm25_k1",
  "hypothesis": "Lower k1 improves precision for technical docs",
  "param": "BM25_K1",
  "values": [1.0, 1.2, 1.5, 2.0],
  "metric": "NDCG@10",
  "baseline": 0.75,
  "expected_improvement": 0.05,
  "effort_hours": 2,
  "risk": "LOW"
}
```

---

## 🎯 Decision Tree

**Start here**:

```
What's your primary goal?

┌─ Quick health check, fast iteration
│  └─► Use SIMPLE prompt
│     Output: TOP_10_IMPROVEMENTS.md
│     Time: 30 min
│
├─ Improve RAG/ML quality (retrieval, answers)
│  └─► Use RAG_QUALITY prompt
│     Output: RAG_QUALITY_REPORT.md + experiments
│     Time: 45 min
│
└─ Full production audit (all aspects)
   └─► Use COMPREHENSIVE prompt
      Output: 4 docs (report, jsonl, quick wins, roadmap)
      Time: 1-2 hours
```

---

## 📖 How to Use

### Step 1: Choose Your Prompt
Select based on your goals (see Decision Tree above)

### Step 2: Start New Claude Code Session
Open a fresh session (don't reuse old sessions)

### Step 3: Navigate to Repository
```bash
cd /path/to/1rag
```

### Step 4: Copy Entire Prompt
Open the selected prompt file and copy everything from "Role:" to "Begin analysis..."

### Step 5: Paste and Run
Paste into Claude Code and press Enter

### Step 6: Wait for Completion
- SIMPLE: ~30 min
- RAG_QUALITY: ~45 min
- COMPREHENSIVE: ~1-2 hours

### Step 7: Review Deliverables
Check the generated markdown/jsonl files

### Step 8: Apply Improvements
Start with quick wins or highest ROI items

---

## 🔄 Combining Prompts

**Recommended workflow**:

1. **Week 1**: Run **SIMPLE** prompt
   - Get quick wins
   - Apply top 5 improvements

2. **Week 2**: Run **RAG_QUALITY** prompt
   - Focus on retrieval accuracy
   - Run suggested experiments
   - Build evaluation framework

3. **Month 1**: Run **COMPREHENSIVE** prompt
   - Full audit before production
   - Address all CRITICAL/HIGH issues
   - Plan long-term architecture

---

## 📊 Comparison Matrix

| Aspect | SIMPLE | COMPREHENSIVE | RAG_QUALITY |
|--------|--------|---------------|-------------|
| **Time** | 30 min | 1-2 hours | 45 min |
| **Depth** | Surface | Deep | Deep (RAG only) |
| **Scope** | Top issues | Everything | RAG pipeline |
| **Output Files** | 1 | 4 | 3 |
| **Structured Data** | ❌ | ✅ (JSONL) | ✅ (JSONL) |
| **Code Quality** | ✅ | ✅✅✅ | ⚠️ |
| **RAG Quality** | ✅ | ✅✅ | ✅✅✅ |
| **Performance** | ✅ | ✅✅✅ | ✅✅ |
| **Security** | ✅ | ✅✅✅ | ❌ |
| **Architecture** | ❌ | ✅✅✅ | ⚠️ |
| **Experiments** | ❌ | ⚠️ | ✅✅✅ |
| **Quick Wins** | ✅ (top 10) | ✅ (separate doc) | ✅ (RAG only) |
| **Roadmap** | ❌ | ✅ | ❌ |

Legend: ✅✅✅ = Excellent, ✅✅ = Good, ✅ = Basic, ⚠️ = Partial, ❌ = Not covered

---

## 💡 Pro Tips

### For Best Results

1. **One prompt per session**: Don't mix prompts in same session
2. **Fresh session**: Start new session for each analysis
3. **Let it complete**: Don't interrupt analysis
4. **Review all deliverables**: Each has unique insights
5. **Prioritize by ROI**: Focus on high-impact, low-effort first

### Common Follow-Ups

After receiving analysis:

```bash
# Apply a specific improvement
"Implement improvement #3 from TOP_10_IMPROVEMENTS.md"

# Run an experiment
"Run the tune_bm25_k1 experiment from RAG_EXPERIMENTS.jsonl"

# Create tests
"Create unit tests for the issues found in ANALYSIS_REPORT.md section 3.2"

# Benchmark
"Create a benchmark script to measure the performance improvements"
```

### When to Re-run

- ✅ After major refactoring (to validate improvements)
- ✅ Before production deployment (COMPREHENSIVE)
- ✅ Monthly health check (SIMPLE)
- ✅ After adding new features (relevant prompt)
- ✅ When RAG quality degrades (RAG_QUALITY)

---

## 📁 File Listing

All prompts in this directory:

```
CLAUDE_PROMPTS_README.md              ← You are here
CLAUDE_CODE_PROMPT_SIMPLE.md          ← 30 min, top 10 improvements
CLAUDE_CODE_PROMPT_COMPREHENSIVE.md   ← 1-2 hrs, full audit
CLAUDE_CODE_PROMPT_RAG_QUALITY.md     ← 45 min, RAG effectiveness
```

---

## 🆚 vs. Previous Reviews

These prompts are **optimized successors** to the manual review process:

| Aspect | Manual Review (Previous) | These Prompts (New) |
|--------|-------------------------|---------------------|
| **Consistency** | Varies by session | Structured, repeatable |
| **Completeness** | Depends on memory | Comprehensive checklists |
| **Actionability** | Good | Excellent (code snippets, metrics) |
| **Prioritization** | Manual | Automated (ROI ranking) |
| **Reusability** | Low | High (copy-paste) |
| **Documentation** | Ad-hoc | 3-4 deliverables per run |
| **Data Format** | Markdown | Markdown + JSONL |
| **Time** | Variable | Predictable |

---

## 🚀 Quick Start

**Just want to try?** → Use SIMPLE prompt:

1. Copy `CLAUDE_CODE_PROMPT_SIMPLE.md`
2. New Claude Code session
3. `cd /path/to/1rag`
4. Paste prompt
5. Get TOP_10_IMPROVEMENTS.md in 30 min

**Want comprehensive audit?** → Use COMPREHENSIVE prompt

**Want to improve RAG quality?** → Use RAG_QUALITY prompt

---

**Created**: 2025-11-05
**Version**: 1.0
**Status**: ✅ Production Ready
**Maintained By**: 1rag project contributors
