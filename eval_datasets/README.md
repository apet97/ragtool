# RAG Evaluation Datasets

This directory contains ground truth evaluation datasets for measuring RAG system performance.

## clockify_v1.jsonl

**Status**: ⚠️ Needs chunk ID population
**Questions**: 20
**Format**: JSONL (one JSON object per line)

### Format

```json
{
  "query": "User question text",
  "relevant_chunk_ids": ["chunk_id_1", "chunk_id_2"],
  "difficulty": "easy|medium|hard",
  "tags": ["tag1", "tag2"],
  "language": "en",
  "notes": "Optional notes about expected chunks"
}
```

### Fields

- `query`: The user's question (required)
- `relevant_chunk_ids`: List of chunk IDs from chunks.jsonl that contain relevant information (required)
- `difficulty`: Question difficulty level (easy/medium/hard)
- `tags`: Categorization tags for analysis
- `language`: Query language code (ISO 639-1)
- `notes`: Helper notes for manual chunk ID identification

### How to Populate Chunk IDs

The dataset currently has empty `relevant_chunk_ids` arrays. To populate them:

#### Method 1: Manual Search (Recommended for accuracy)

```bash
# 1. Load chunks.jsonl
python3 -c "
import json
with open('chunks.jsonl') as f:
    chunks = [json.loads(line) for line in f if line.strip()]

# 2. Search for keywords
query = 'time tracking'
for i, chunk in enumerate(chunks):
    if query.lower() in chunk['text'].lower():
        print(f\"Chunk {i}: {chunk['id'][:8]}... - {chunk['title']}\")
        print(f\"  {chunk['text'][:100]}...\")
        print()
"
```

#### Method 2: Automated with Current System

```bash
# Run queries and capture which chunks were retrieved
python3 clockify_support_cli_final.py ask "How do I track time?" --debug

# Note the chunk IDs shown in debug output
# Manually verify they're actually relevant
# Add IDs to the dataset
```

#### Method 3: Helper Script

Create `populate_ground_truth.py`:

```python
import json

# Load chunks
with open('chunks.jsonl') as f:
    chunks = [json.loads(line) for line in f if line.strip()]

# Load evaluation dataset
with open('eval_datasets/clockify_v1.jsonl') as f:
    eval_data = [json.loads(line) for line in f if line.strip()]

# For each query, manually identify relevant chunks
for item in eval_data:
    print(f"\nQuery: {item['query']}")
    print(f"Notes: {item.get('notes', 'N/A')}")
    print("\nSearch chunks.jsonl for relevant content...")
    print("Enter chunk IDs (comma-separated): ", end='')

    chunk_ids = input().strip()
    if chunk_ids:
        item['relevant_chunk_ids'] = [cid.strip() for cid in chunk_ids.split(',')]

# Save updated dataset
with open('eval_datasets/clockify_v1_populated.jsonl', 'w') as f:
    for item in eval_data:
        f.write(json.dumps(item) + '\n')

print("\nSaved to eval_datasets/clockify_v1_populated.jsonl")
```

### Evaluation Metrics

Once chunk IDs are populated, evaluate with:

```bash
# Run evaluation
python3 eval.py --dataset eval_datasets/clockify_v1.jsonl

# Expected output:
# - MRR (Mean Reciprocal Rank): How high relevant chunks rank
# - NDCG@k: Normalized Discounted Cumulative Gain
# - Precision@k: Fraction of top-k that are relevant
# - Recall@k: Fraction of relevant docs in top-k
```

### Coverage by Category

| Category | Count | Difficulty | Notes |
|----------|-------|------------|-------|
| Time Tracking | 4 | Easy-Medium | Core functionality |
| Pricing | 3 | Easy-Medium | Common questions |
| Projects | 5 | Easy-Medium | Project management |
| Reports | 3 | Easy-Medium | Reporting features |
| Advanced | 5 | Medium-Hard | SSO, API, workflows |

### Expansion Strategy

To reach 50-100 questions:

1. **More basic questions** (10): Timer usage, mobile app basics, account setup
2. **Integration questions** (10): Specific integrations (Jira, Slack, Asana, etc.)
3. **Advanced features** (10): Custom fields, formulas, automation
4. **Troubleshooting** (10): Common errors, sync issues, login problems
5. **Mobile-specific** (5): iOS/Android specific features
6. **API questions** (5): Endpoints, authentication, webhooks

### Quality Criteria

For each entry:
- ✅ Question is natural and realistic
- ✅ Difficulty matches complexity
- ✅ Tags accurately categorize
- ✅ At least 1-3 relevant chunks identified
- ✅ Relevant chunks actually answer the question

### Usage

```python
# Load dataset
import json

with open('eval_datasets/clockify_v1.jsonl') as f:
    dataset = [json.loads(line) for line in f if line.strip()]

# Filter by difficulty
easy_questions = [d for d in dataset if d['difficulty'] == 'easy']

# Filter by tag
pricing_questions = [d for d in dataset if 'pricing' in d['tags']]
```

## Contributing

To add new questions:

1. Follow the JSON format above
2. Provide meaningful tags
3. Add helpful notes for chunk identification
4. Maintain diversity of difficulty levels
5. Cover different documentation areas
