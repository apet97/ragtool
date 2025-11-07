# Claude Code Prompt: Simple RAG Analysis

**Copy this for a quick, focused analysis in a new Claude Code session**

---

## SIMPLE PROMPT

```
Role: RAG/ML Engineer analyzing a Python-based RAG (Retrieval-Augmented Generation) tool.

Task: Read all Python files in this repository and identify the top 10 most impactful improvements.

Focus on:
1. RAG quality (retrieval accuracy, answer quality, chunking strategy)
2. Performance (speed, memory, caching)
3. Critical bugs (crashes, data loss, correctness)
4. User experience (CLI, errors, documentation)

Skip: .git, venv, __pycache__, node_modules, dist, build, *.zip, *.tar.gz

Deliverables:
1. TOP_10_IMPROVEMENTS.md with:
   - Issue description
   - Impact (HIGH/MEDIUM/LOW)
   - Effort (hours)
   - File/line reference
   - Proposed fix (code snippet if applicable)

2. Quick summary (3-5 sentences) of overall code quality

Constraints:
- Analysis only, no code changes
- Be specific (file names, line numbers)
- Quantify impact when possible
- Prioritize by ROI (impact ÷ effort)

Start by reading clockify_support_cli_final.py completely, then other Python files.
```

---

## Usage

1. Start new Claude Code session
2. Navigate to repo: `cd /path/to/1rag`
3. Paste the prompt above
4. Review `TOP_10_IMPROVEMENTS.md`

---

## When to Use

- ✅ Quick health check (30 min analysis)
- ✅ Focused on actionable items only
- ✅ Don't need comprehensive documentation
- ✅ Want fast iteration cycle

## When to Use Comprehensive Prompt Instead

- 🎯 Full production readiness assessment
- 🎯 Deep architectural review
- 🎯 Need detailed documentation of all findings
- 🎯 Building long-term improvement roadmap

---

**Time**: ~30 minutes
**Output**: 1 document (TOP_10_IMPROVEMENTS.md)
**Best For**: Quick wins, rapid iteration
