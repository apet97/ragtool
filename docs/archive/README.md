# Archive Directory

This directory contains archived files for historical reference.

## Contents

### Code Snapshots
- `clockify_support_cli_final.py.bak_v41` - CLI snapshot from v4.1 release, retained for reference during major refactorings

### Audit Deliverables (audit_deliverables/)
**Archived**: 2025-11-07 (v5.2 cleanup - Priority #10)

Archive contains 59 legacy documentation files from prior audit cycles:
- Analysis reports and findings (ANALYSIS_REPORT1.md, COMPREHENSIVE_CODE_REVIEW.md)
- Architecture vision documents (ARCHITECTURE_VISION*.md)
- Final delivery documents (FINAL_*.md, V4_0_*.md)
- Quick wins and improvement plans (QUICK_WINS*.md, HARDENING_*.md, OPERATIONAL_IMPROVEMENTS_SUMMARY.md)
- Acceptance tests and proof documents (ACCEPTANCE_TESTS_PROOF.md, DEEPSEEK_*.md)
- Claude Code prompts and planning docs (CLAUDE_CODE_PROMPT*.md, PLAN_*.md)
- Legacy compatibility audits (COMPATIBILITY_AUDIT*.md, M1_*_AUDIT.md)
- Refactoring guides (OLLAMA_REFACTORING*.md, MODULARIZATION*.md)
- Test plans and checklists (TESTPLAN*.md, V3_5_VERIFICATION_CHECKLIST.md)

**Reason for archival**: These documents served their purpose during development and audit cycles.
They are retained for historical reference but are no longer active documentation.

## Purpose

Historical artifacts are moved here to:
1. Keep the repository root clean and focused on active development
2. Preserve important snapshots for comparison and rollback if needed
3. Make it clear which files are legacy vs. current
4. Reduce documentation maintenance overhead
5. Improve repository clarity for new contributors

## Active Documentation

The repository root now contains only active, maintained documentation:
- **README.md** - Main project documentation
- **CLAUDE.md** - Project instructions for Claude Code
- **ANALYSIS_REPORT.md** - Current audit findings
- **LOGGING_CONFIG.md** - Logging configuration guide
- **M1_COMPATIBILITY.md** - Apple Silicon compatibility guide
- **START_HERE.md** - Quick start guide
- **SUPPORT_CLI_QUICKSTART.md** - CLI quick reference
- **QUICKSTART.md**, **README_RAG.md** - v1.0 documentation
- **VERSION_COMPARISON.md** - v1 vs v2 comparison
- **PROJECT_STRUCTURE.md** - Repository structure
- **CHANGELOG_v4.1.md** - Version changelog
- **CI_CD_M1_RECOMMENDATIONS.md** - CI/CD guidance
- **knowledge_full.md** - Source knowledge base

## Guidelines

- Files in this directory should not be imported or executed
- Consider removing files older than 2 releases unless they serve a specific reference purpose
- Document the reason for archiving in commit messages
- Maintain this README when adding new archives
