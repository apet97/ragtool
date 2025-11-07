# Claude Code Prompt: Comprehensive RAG Tool Analysis

**Copy and paste this prompt into a new Claude Code session for deep codebase analysis**

---

## PROMPT START

```
Role: Senior ML/RAG Engineer conducting comprehensive end-to-end codebase analysis and optimization.

Objective: Analyze the entire RAG tool codebase file-by-file to identify improvements in correctness, performance, architecture, RAG quality, and user experience.

## Scope

Recursively read and analyze EVERY file in the current repository:
- All Python files (*.py) - main implementation, scripts, utilities
- All shell scripts (*.sh) - test scripts, automation
- All documentation (*.md, CLAUDE.md) - accuracy, completeness
- All configuration (Makefile, requirements*.txt, .gitignore)
- All data files if small (<100KB)

**Skip**: .git/, .venv*, venv/, __pycache__/, node_modules/, dist/, build/, .mypy_cache/, .ruff_cache/, data/, logs/, *.tar.gz, *.zip, *.png/jpg/gif, large binaries (>2MB)

**Do not**: Run destructive commands, modify files during analysis phase, execute code that requires external services

## Analysis Focus Areas

### 1. RAG-Specific Quality
- **Retrieval accuracy**: Hybrid search implementation (BM25 + dense + MMR)
- **Chunking strategy**: Size, overlap, boundary detection, metadata preservation
- **Embedding quality**: Model choice, normalization, caching efficiency
- **Context packing**: Token budget utilization, snippet formatting, truncation
- **LLM prompting**: System prompts, few-shot examples, grounding techniques
- **Answer quality**: Citation accuracy, hallucination prevention, refusal handling
- **Evaluation**: Missing metrics (MRR, NDCG, answer relevance), no ground truth

### 2. Performance & Scalability
- **Indexing speed**: Build time optimization, parallel processing opportunities
- **Query latency**: Hot path analysis, unnecessary allocations, redundant computation
- **Memory efficiency**: Large array handling, mmap usage, cache sizes
- **I/O patterns**: File operations, network calls, batching opportunities
- **Algorithm complexity**: O(n²) loops, suboptimal data structures
- **Caching strategies**: What's cached, what should be cached, invalidation logic

### 3. Correctness & Reliability
- **Edge cases**: Empty inputs, malformed data, boundary conditions
- **Error handling**: Exception types, recovery strategies, error messages
- **Input validation**: Type checks, bounds checks, sanitization
- **Concurrency safety**: Race conditions, atomic operations, lock management
- **Data integrity**: Checksums, validation, corruption detection
- **Graceful degradation**: Fallback mechanisms, partial failure handling

### 4. Code Quality & Maintainability
- **Architecture**: Module boundaries, dependency graph, coupling
- **Design patterns**: Applied correctly, missing opportunities
- **Type safety**: Type hints coverage, mypy/pyright compatibility
- **Naming**: Clarity, consistency, domain alignment
- **Function size**: SRP violations, god functions, refactoring opportunities
- **Duplication**: Code smell detection, abstraction opportunities
- **Documentation**: Docstrings, inline comments, architectural docs
- **Testing**: Coverage gaps, missing unit/integration tests, test quality

### 5. Security & Safety
- **Input sanitization**: Prompt injection, path traversal, command injection
- **Secrets management**: Hardcoded keys, credential exposure
- **Dependency vulnerabilities**: Known CVEs, outdated packages
- **File operations**: Temp file security, atomic writes, permission checks
- **Logging**: PII leakage, sensitive data in logs/errors
- **Resource limits**: DoS prevention, rate limiting, memory bounds

### 6. Developer Experience
- **Setup difficulty**: Installation steps, dependency clarity, error messages
- **CLI usability**: Help text, argument validation, interactive prompts
- **Debugging**: Log levels, debug mode, diagnostic output
- **Configuration**: Environment variables, config files, sensible defaults
- **Error messages**: Actionable, specific, suggest fixes
- **Documentation**: Quickstart, examples, troubleshooting, API docs

### 7. RAG Pipeline Improvements
- **Advanced retrieval**: Cross-encoder reranking, query expansion, multi-hop
- **Chunk optimization**: Sentence-aware splitting, section boundaries, overlap tuning
- **Metadata enrichment**: Chunk provenance, confidence scores, timestamps
- **Answer formatting**: Structured output, citation formatting, markdown rendering
- **Streaming**: Support for streaming LLM responses
- **Multi-modal**: Image/table handling in documents
- **Multilingual**: Unicode handling, non-English support

## Deliverables

### A) ANALYSIS_REPORT.md

**Structure**:
```markdown
# Comprehensive RAG Tool Analysis

## Executive Summary
- Overall assessment (1-5 stars)
- Top 3 strengths
- Top 5 critical improvements needed
- Production readiness: YES/NO with justification

## File-by-File Analysis
For each file:
- Purpose & responsibility
- Lines of code
- Key findings (bugs, optimizations, improvements)
- Quality score (1-5)

## Findings by Category
1. RAG Quality (score: X/10)
   - Retrieval effectiveness
   - Answer quality
   - Prompt engineering

2. Performance (score: X/10)
   - Indexing speed
   - Query latency
   - Memory efficiency

3. Correctness (score: X/10)
   - Bug count & severity
   - Edge case handling
   - Data validation

4. Code Quality (score: X/10)
   - Architecture
   - Maintainability
   - Documentation

5. Security (score: X/10)
   - Vulnerabilities found
   - Best practices adherence

6. Developer Experience (score: X/10)
   - Setup ease
   - CLI usability
   - Debug capabilities

## Priority Improvements (Top 20)
| Rank | Category | Issue | Impact | Effort | ROI |
|------|----------|-------|--------|--------|-----|
| 1    | RAG      | ...   | HIGH   | LOW    | 9/10|

## RAG-Specific Recommendations
- Retrieval pipeline enhancements
- Chunking strategy improvements
- Prompt engineering optimizations
- Evaluation framework additions

## Architecture Recommendations
- Module restructuring suggestions
- Design pattern applications
- Dependency improvements

## Performance Hotspots
- Profiling results (if applicable)
- Top 5 optimization opportunities
- Expected speedup estimates

## Testing Strategy
- Missing test coverage areas
- Recommended test cases
- Integration test scenarios
- Benchmark suite design
```

### B) IMPROVEMENTS.jsonl

**Format**: One JSON object per line
```jsonl
{"rank":1,"category":"RAG","subcategory":"retrieval","issue":"BM25 k1 parameter not tuned for domain","impact":"HIGH","effort":"LOW","file":"clockify_support_cli_final.py","line":935,"current":"k1=1.5 (default)","proposed":"k1=1.2 (better for technical docs)","rationale":"Technical documentation has different term frequency characteristics than web text. Lower k1 reduces over-weighting of repeated terms.","implementation":"Add BM25_K1 env var, tune on sample queries","expected_gain":"5-10% retrieval accuracy improvement","references":["https://arxiv.org/abs/1803.08988"]}
{"rank":2,"category":"performance","subcategory":"caching","issue":"Embeddings recomputed on every build","impact":"HIGH","effort":"MEDIUM","file":"clockify_support_cli_final.py","line":850,"current":"No embedding cache","proposed":"Add persistent embedding cache with content hash","rationale":"Embeddings are deterministic for same input. 50%+ of chunks unchanged between builds.","implementation":"Add emb_cache.jsonl with {content_hash: embedding} mapping","expected_gain":"50% faster incremental builds","references":["task F in code comments"]}
```

### C) QUICK_WINS.md

**Top 10 improvements with**:
- <30 min implementation time
- High impact (user-visible or performance)
- Low risk (no breaking changes)
- Ready-to-apply code snippets

### D) ARCHITECTURE_VISION.md

**Long-term roadmap**:
- Modularization plan (if needed)
- Plugin architecture (if applicable)
- API design (if exposing programmatic interface)
- Scaling strategy (distributed indexing, query caching)

## Analysis Methodology

1. **File Discovery**
   - Use Glob to enumerate all files
   - Categorize by type (Python, shell, docs, config)
   - Prioritize by importance (main > utils > tests > docs)

2. **Code Reading**
   - Read each file completely (use offset/limit for large files)
   - Track imports, dependencies, data flow
   - Build mental model of architecture

3. **Pattern Detection**
   - Grep for anti-patterns (bare except, global state, magic numbers)
   - Grep for missing patterns (no type hints, no tests, no validation)
   - Grep for duplicated code

4. **Deep Analysis**
   - For each function: purpose, complexity, correctness, performance
   - For each class: SRP, cohesion, testability
   - For each module: coupling, boundary clarity

5. **Comparative Analysis**
   - Compare against RAG best practices (LangChain, LlamaIndex patterns)
   - Compare against production-grade implementations
   - Identify gaps vs. state-of-the-art

6. **Synthesis**
   - Aggregate findings into categories
   - Rank by impact × feasibility
   - Generate actionable recommendations

## Output Guidelines

- **Be specific**: File names, line numbers, exact code snippets
- **Be actionable**: Every issue should have a proposed fix
- **Be measurable**: Quantify impact when possible (% improvement, time saved)
- **Be pragmatic**: Consider effort vs. reward
- **Be comprehensive**: Don't skip files, even if they seem simple
- **Be critical**: This is a deep audit, not a rubber stamp
- **Be constructive**: Frame negatives as improvement opportunities

## Constraints

- Analysis only (no code modifications during analysis)
- Use Task (Explore agent) for codebase navigation where appropriate
- Read files completely, don't rely on snippets for critical analysis
- Validate claims by checking actual code, not assumptions
- If uncertain about behavior, read the code carefully or note assumption

## Success Criteria

✅ Every Python file analyzed and scored
✅ Every function >50 lines reviewed for refactoring
✅ RAG pipeline end-to-end traced and evaluated
✅ Top 20 improvements identified and ranked
✅ All deliverables generated (ANALYSIS_REPORT.md, IMPROVEMENTS.jsonl, QUICK_WINS.md, ARCHITECTURE_VISION.md)
✅ Findings are specific, actionable, and prioritized
✅ Analysis completed in <2 hours (Claude time)

Begin analysis now. Start with file enumeration and categorization.
```

## PROMPT END

---

## Usage Instructions

1. **Start a new Claude Code session**
2. **Navigate to the repository**: `cd /path/to/1rag`
3. **Copy and paste the entire prompt** (from "Role:" to "Begin analysis now.")
4. **Wait for analysis to complete** (~30-60 minutes)
5. **Review deliverables**:
   - `ANALYSIS_REPORT.md` - Main findings
   - `IMPROVEMENTS.jsonl` - Structured recommendations
   - `QUICK_WINS.md` - Fast improvements
   - `ARCHITECTURE_VISION.md` - Long-term roadmap

---

## Expected Outcomes

### Comprehensive Coverage
- ✅ All Python files analyzed (main, scripts, utils)
- ✅ All shell scripts reviewed (smoke, acceptance, benchmark)
- ✅ All documentation validated (CLAUDE.md, READMEs)
- ✅ All configuration checked (Makefile, requirements)

### Deep Insights
- 🎯 RAG-specific improvements (retrieval, chunking, prompting)
- ⚡ Performance optimizations (caching, algorithms, I/O)
- 🐛 Bug detection (edge cases, race conditions, validation)
- 🏗️ Architecture recommendations (modularity, patterns)
- 🔒 Security findings (input validation, secrets)
- 👨‍💻 DX improvements (CLI, errors, docs)

### Actionable Output
- Ranked by impact × effort (ROI optimization)
- Specific file/line references
- Code snippets for fixes
- Expected gains quantified
- Quick wins identified (<30 min each)

---

## Differences from Previous Review

| Aspect | Previous (Full-Repo Review) | This (RAG-Focused Analysis) |
|--------|---------------------------|---------------------------|
| **Focus** | General code quality, bugs | RAG quality, retrieval accuracy |
| **Deliverables** | 3 (REVIEW, PATCHES, TESTPLAN) | 4 (ANALYSIS, IMPROVEMENTS.jsonl, QUICK_WINS, ARCHITECTURE) |
| **Depth** | Correctness, security, performance | + RAG best practices, ML pipeline |
| **Output** | Markdown tables | + JSONL for programmatic use |
| **Scope** | All issues | Prioritized improvements only |
| **Emphasis** | Fix bugs | Improve RAG effectiveness |

---

## Advanced Variant (For Deeper Analysis)

If you want even more depth, add these sections:

```markdown
### Additional Analysis Areas

8. **RAG Evaluation**
   - Create ground truth Q&A pairs (10-20 samples)
   - Measure MRR@10, NDCG@10 for retrieval
   - Measure answer correctness, citation accuracy
   - Compare BM25 vs. dense vs. hybrid performance

9. **Competitive Analysis**
   - Compare vs. LlamaIndex baseline
   - Compare vs. LangChain RAG template
   - Identify unique strengths/weaknesses

10. **Scalability Analysis**
    - Project memory usage at 10x, 100x KB size
    - Estimate query latency at 1K, 10K QPS
    - Identify bottlenecks for horizontal scaling
```

---

## Tips for Best Results

1. **Let it run completely** - Don't interrupt the analysis
2. **Review IMPROVEMENTS.jsonl first** - Most actionable data
3. **Start with QUICK_WINS.md** - Fast improvements
4. **Use findings as backlog** - Each item becomes a task
5. **Re-run after major changes** - Track improvement over time

---

## Example Follow-Up Prompts

After analysis completes:

```
"Apply the top 5 quick wins from QUICK_WINS.md"

"Implement the #1 ranked improvement from IMPROVEMENTS.jsonl"

"Create a test suite based on missing coverage identified in ANALYSIS_REPORT.md"

"Generate a benchmark script to measure the performance improvements you identified"

"Refactor the retrieval pipeline based on your RAG-specific recommendations"
```

---

**Status**: ✅ Ready to use
**Last Updated**: 2025-11-05
**Recommended For**: Deep RAG quality analysis, ML pipeline optimization, production readiness assessment
