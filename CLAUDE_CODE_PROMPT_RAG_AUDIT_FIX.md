# Claude Code Prompt: Repair RAG Audit Deliverables

**Copy this entire prompt into a fresh Claude Code session to regenerate the RAG audit deliverables with full fidelity.**

---

## PROMPT START

```
Role: Senior ML/RAG engineer tasked with repairing the RAG audit deliverables after an unsatisfactory rewrite.

Primary Goal: Recreate ANALYSIS_REPORT1.md, IMPROVEMENTS1.jsonl, QUICK_WINS1.md, and ARCHITECTURE_VISION1.md so they again meet the original comprehensive-audit specification and address every regression introduced in commit 5169c8e2.

### Context
- The current HEAD commit (5169c8e2, "Refresh RAG audit deliverables") drastically shortened the four deliverables and introduced factual inaccuracies. Review feedback on that diff must be incorporated.
- The previous versions of these files (from commit 5169c8e2^, i.e., `git show HEAD^:<file>`) contained the desired level of depth—use them as the structural baseline but update any stale findings uncovered during re-analysis.
- The repository root contains numerous historical documents; only the four deliverables above should be overwritten.

### Resources & References
- Original requirements in `CLAUDE_CODE_PROMPT_COMPREHENSIVE.md` (this is the canonical spec for the analysis output format and scoring details).
- Repository code under `clockify_rag/`, `clockify_support_cli_final.py`, scripts, tests, and docs—re-read these to ensure every claim is evidence-based.
- Any inline PR review comments associated with commit 5169c8e2 must be satisfied. If a comment cites a specific section, update that section explicitly.

### Required Workflow
1. **Re-evaluate the codebase.** Skim every Python module, shell script, and relevant documentation to confirm whether the findings in the pre-regression versions are still accurate. Update conclusions where the code has changed since that snapshot.
2. **Use the previous deliverables as scaffolding.** Pull them via `git show HEAD^:ANALYSIS_REPORT1.md` (and the other files) to recover the comprehensive tables, category scores, and ranked improvements. Preserve their structure unless an inline comment demands otherwise.
3. **Validate every statement.** Do not copy generic claims. When you assert a bug or recommendation, reference the exact file path, function, or configuration knob that justifies it. Remove any entries that cannot be tied to a verified code location.
4. **Incorporate review feedback.** For each inline comment from the regression diff, edit the impacted section to resolve the reviewer’s concern (e.g., restore deleted detail, correct incorrect metrics, expand rationales).
5. **Regenerate the Top 20 improvements, Quick Wins, and Architecture Vision** so they align with the refreshed analysis. Ensure JSON records remain valid (double quotes, escaped characters) and Markdown tables render correctly.
6. **Self-review.** Before finishing, diff the regenerated files against HEAD^ to ensure the level of detail is comparable or better, and lint JSON using `python -m json.tool IMPROVEMENTS1.jsonl` (use streaming validation) to verify syntax.

### Deliverable-Specific Acceptance Criteria
- **ANALYSIS_REPORT1.md** must follow the canonical outline (Executive Summary, File-by-File table, category scores, Top 20 improvements table, etc.). Every file in the repo should still appear in the table with updated LOC and quality scores.
- **IMPROVEMENTS1.jsonl** requires exactly 20 JSON objects, each with `rank`, `category`, `subcategory`, `issue`, `impact`, `effort`, `file`, `line`, `current`, `proposed`, `rationale`, `implementation`, `expected_gain`, `references`.
- **QUICK_WINS1.md** must list ten <30-minute fixes with ready-to-apply code snippets that match current source files.
- **ARCHITECTURE_VISION1.md** should restore the detailed roadmap (modularization, plugin strategy, scaling) while updating timelines and milestones per the latest findings.

### Definition of Done
- All four files are regenerated with comprehensive, accurate content that addresses review comments.
- JSON validates and Markdown tables render without broken pipes or formatting issues.
- Every recommendation links to specific code locations or documentation sections.
- No other repository files are modified.

Output only the modified file contents; do not run `git commit`.
```

## PROMPT END
