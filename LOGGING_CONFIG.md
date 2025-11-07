# Logging Configuration Guide

This document explains all logging-related environment variables for the Clockify RAG system.

## Quick Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_LOG_FILE` | `rag_queries.jsonl` | Path to query log file |
| `RAG_NO_LOG` | `0` | Disable all query logging (`1` to disable) |
| `RAG_LOG_INCLUDE_ANSWER` | `1` | Include answer text in logs (`0` to redact) |
| `RAG_LOG_ANSWER_PLACEHOLDER` | `[REDACTED]` | Placeholder when answer is redacted |
| `RAG_LOG_INCLUDE_CHUNKS` | `0` | Include full chunk text in logs (`1` to enable) |

## Detailed Configuration

### RAG_LOG_FILE

**Default**: `rag_queries.jsonl`
**Format**: File path (relative or absolute)

Specifies where query logs are written. Each line is a JSON object containing:
- Query text and metadata
- Retrieved chunks (with or without text, depending on flags)
- Answer (with or without text, depending on flags)
- Timing metrics
- Confidence scores

**Example**:
```bash
export RAG_LOG_FILE="/var/log/rag/queries.jsonl"
python3 clockify_support_cli_final.py chat
```

### RAG_NO_LOG

**Default**: `0` (logging enabled)
**Values**: `0`, `1`, `false`, `true`, `yes`, `no`, `on`, `off`

Master switch to disable all query logging. When set to `1` (or equivalent), no logs are written regardless of other flags.

**Use case**: Disable logging in development or when testing without creating log files.

**Example**:
```bash
RAG_NO_LOG=1 python3 clockify_support_cli_final.py chat
```

### RAG_LOG_INCLUDE_ANSWER

**Default**: `1` (answers included)
**Values**: `0`, `1`, `false`, `true`, `yes`, `no`, `on`, `off`

Controls whether the generated answer text is included in logs.

- **`1` (enabled)**: Full answer text is logged
- **`0` (disabled)**: Answer is replaced with placeholder (see `RAG_LOG_ANSWER_PLACEHOLDER`)

**Privacy considerations**:
- If answers may contain PII or sensitive information, set to `0`
- Useful for compliance requirements (GDPR, HIPAA, etc.)
- Redacting answers still logs query text, chunk IDs, scores, and metadata

**Example**:
```bash
# Redact answers for privacy
export RAG_LOG_INCLUDE_ANSWER=0
python3 clockify_support_cli_final.py chat
```

### RAG_LOG_ANSWER_PLACEHOLDER

**Default**: `[REDACTED]`
**Format**: Any string

Specifies the placeholder text used when `RAG_LOG_INCLUDE_ANSWER=0`.

**Example**:
```bash
export RAG_LOG_INCLUDE_ANSWER=0
export RAG_LOG_ANSWER_PLACEHOLDER="<answer omitted for privacy>"
python3 clockify_support_cli_final.py chat
```

### RAG_LOG_INCLUDE_CHUNKS

**Default**: `0` (chunks redacted)
**Values**: `0`, `1`, `false`, `true`, `yes`, `no`, `on`, `off`

Controls whether the full text of retrieved chunks is included in logs.

- **`0` (disabled, default)**: Only chunk IDs and scores are logged; chunk text is redacted for security
- **`1` (enabled)**: Full chunk text is included in logs

**Security considerations**:
- **Default is OFF for security**: Chunk text may contain sensitive knowledge base content
- Independent of `RAG_LOG_INCLUDE_ANSWER`: You can log answers without chunk text, or vice versa
- Enabling chunk logging increases log file size significantly

**Use cases**:
- **Development/debugging**: Enable to see what content was retrieved
- **Production**: Keep disabled to prevent knowledge base leakage in logs

**Example**:
```bash
# Enable chunk text for debugging
export RAG_LOG_INCLUDE_CHUNKS=1
python3 clockify_support_cli_final.py chat --debug
```

## Common Configuration Scenarios

### 1. Full Logging (Development)

Log everything for debugging:

```bash
export RAG_LOG_FILE="dev_queries.jsonl"
export RAG_NO_LOG=0
export RAG_LOG_INCLUDE_ANSWER=1
export RAG_LOG_INCLUDE_CHUNKS=1
```

### 2. Privacy-Preserving (Production)

Log queries and metadata but redact sensitive content:

```bash
export RAG_LOG_FILE="/var/log/rag/queries.jsonl"
export RAG_NO_LOG=0
export RAG_LOG_INCLUDE_ANSWER=0          # Redact answers
export RAG_LOG_INCLUDE_CHUNKS=0          # Redact chunk text (default)
export RAG_LOG_ANSWER_PLACEHOLDER="[REDACTED FOR PRIVACY]"
```

### 3. Metrics Only

Log for monitoring but minimize content exposure:

```bash
export RAG_LOG_FILE="metrics.jsonl"
export RAG_NO_LOG=0
export RAG_LOG_INCLUDE_ANSWER=0          # No answer text
export RAG_LOG_INCLUDE_CHUNKS=0          # No chunk text (default)
# Still logs: query length, chunk IDs, scores, timings, confidence
```

### 4. No Logging (Testing)

Disable all logging:

```bash
export RAG_NO_LOG=1
# All other flags ignored when RAG_NO_LOG=1
```

## Log Entry Structure

When logging is enabled, each query produces a JSON log entry with this structure:

```json
{
  "timestamp": 1699564800.123,
  "timestamp_iso": "2023-11-09T20:00:00Z",
  "query": "How do I track time?",
  "query_length": 20,
  "answer": "To track time...",  // Or placeholder if RAG_LOG_INCLUDE_ANSWER=0
  "answer_length": 145,
  "num_chunks_retrieved": 6,
  "chunk_ids": ["uuid-1", "uuid-2", ...],
  "chunk_scores": {
    "dense": [0.85, 0.82, ...],
    "bm25": [0.78, 0.75, ...],
    "hybrid": [0.815, 0.785, ...]
  },
  "retrieved_chunks": [
    {
      "id": "uuid-1",
      "pack_rank": 0,
      "dense": 0.85,
      "bm25": 0.78,
      "hybrid": 0.815,
      "chunk": "Full text..."  // Only if RAG_LOG_INCLUDE_CHUNKS=1
    }
  ],
  "avg_chunk_score": 0.78,
  "max_chunk_score": 0.85,
  "latency_ms": 1234.5,
  "refused": false,
  "metadata": {
    "debug": false,
    "backend": "local",
    "coverage_pass": true,
    "rerank_applied": true,
    "rerank_reason": "success",
    "rerank_candidates": 12,
    "num_selected": 12,
    "num_packed": 6,
    "used_tokens": 2456,
    "timings": {
      "embed": 45.2,
      "retrieve": 23.1,
      "rerank": 156.8,
      "answer": 1234.5
    },
    "confidence": 85
  }
}
```

## Redaction Behavior

### When RAG_LOG_INCLUDE_ANSWER=0:
- `answer` field contains placeholder string
- `answer_length` still shows true length for metrics

### When RAG_LOG_INCLUDE_CHUNKS=0 (default):
- `retrieved_chunks` array present but `chunk` and `text` fields removed
- `chunk_ids` and scores still logged for metrics
- Knowledge base content is protected

## Environment Variable Precedence

1. **RAG_NO_LOG**: Master switch, overrides all other flags
2. **RAG_LOG_INCLUDE_ANSWER**: Controls answer text only
3. **RAG_LOG_INCLUDE_CHUNKS**: Controls chunk text only (independent of answer flag)

## Compliance Considerations

### GDPR / Privacy Regulations

- **Recommendation**: Set `RAG_LOG_INCLUDE_ANSWER=0` and keep `RAG_LOG_INCLUDE_CHUNKS=0`
- Still allows monitoring system performance without storing user content
- Consider rotating/purging log files regularly

### Security / Data Leakage Prevention

- **Default settings are secure**: Chunk text redacted by default
- Enable `RAG_LOG_INCLUDE_CHUNKS=1` only in development or secure environments
- Treat log files as sensitive data (restrict permissions, encrypt at rest)

### Audit Requirements

- Logs include all necessary audit fields (timestamp, query metrics, outcomes)
- No PII in logs when using privacy-preserving configuration
- Consider forwarding logs to SIEM for compliance monitoring

## Troubleshooting

### "No logs are being written"

Check:
1. `RAG_NO_LOG` is not set to `1`
2. Log file path is writable
3. Parent directory exists (created automatically but may fail on permission errors)

### "Logs are too large"

Solutions:
1. Set `RAG_LOG_INCLUDE_CHUNKS=0` (default) to reduce size
2. Implement log rotation (logrotate or application-level)
3. Consider sampling (log only 10% of queries)

### "Can't see what chunks were retrieved"

Enable chunk text logging:
```bash
RAG_LOG_INCLUDE_CHUNKS=1 python3 clockify_support_cli_final.py chat
```

**Warning**: Only do this in development; production logs should keep chunks redacted.

## Related Documentation

- [SUPPORT_CLI_QUICKSTART.md](SUPPORT_CLI_QUICKSTART.md) - Quick start guide
- [CLAUDE.md](CLAUDE.md) - Project overview
- [clockify_rag/config.py](clockify_rag/config.py) - Configuration constants

## Version History

- **v5.1**: Added `RAG_LOG_INCLUDE_CHUNKS` flag for independent chunk redaction
- **v4.1**: Initial logging framework with answer redaction
