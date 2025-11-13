# Clockify RAG API Authentication

This document explains how operators can secure the Clockify RAG API using the
new authentication controls.

## Overview

Sensitive endpoints such as `/v1/query`, `/v1/ingest`, and `/v1/metrics` now
require credentials when authentication is enabled. Two strategies are
supported:

1. **API Keys (recommended for internal deployments)** – compare an incoming
   header to a list of approved secrets.
2. **JWT Bearer tokens** – verify signed tokens issued by an identity
   provider. This option requires the optional `PyJWT` dependency.

If `RAG_AUTH_MODE` is unset or explicitly set to `none`, the query and metrics
endpoints remain open as before, but `/v1/ingest` is now blocked with `403`
responses until a secure mode (`api_key` or `jwt`) is configured. This prevents
unauthenticated rebuilds that could tamper with the knowledge base.

## Configuration

Authentication is configured through environment variables or files mounted via
secret storage (for example, Docker or Kubernetes secrets). The following table
summarises the available settings:

| Variable | Description | Example |
| --- | --- | --- |
| `RAG_AUTH_MODE` | `none`, `api_key`, or `jwt` | `api_key` |
| `RAG_API_KEY_HEADER` | HTTP header used for API keys | `X-Internal-Key` |
| `RAG_API_KEYS` | Comma separated API keys | `prod-key-1,prod-key-2` |
| `RAG_API_KEYS_FILE` | Path to file containing keys (newline or JSON list) | `/var/run/secrets/rag-api-keys` |
| `RAG_JWT_SECRET` | Shared secret for verifying JWTs | — |
| `RAG_JWT_SECRET_FILE` | Path to file containing the JWT secret | `/var/run/secrets/rag-jwt-secret` |
| `RAG_JWT_ALGORITHMS` | Comma separated list of JWT algorithms | `HS256,HS512` |

### Example: API Key Authentication

```bash
export RAG_AUTH_MODE=api_key
export RAG_API_KEY_HEADER=X-API-Key
export RAG_API_KEYS_FILE=/var/run/secrets/rag_keys
```

The file referenced above may contain either newline separated keys or a JSON
array:

```
prod-key-1
prod-key-2
```

or

```json
["prod-key-1", "prod-key-2"]
```

### Example: JWT Authentication

```bash
export RAG_AUTH_MODE=jwt
export RAG_JWT_SECRET_FILE=/var/run/secrets/rag_jwt_secret
export RAG_JWT_ALGORITHMS=HS256
```

Install `PyJWT` in the deployment environment to enable token validation:

```bash
pip install PyJWT
```

Clients must send a standard `Authorization: Bearer <token>` header.

## Operational Notes

- Missing or invalid credentials result in `401` and `403` responses
  respectively.
- Misconfiguration (for example enabling `api_key` without provisioning keys)
  returns `500` so that operators can detect the issue quickly.
- A successful authentication is exposed to the route handler via dependency
  injection, which enables future auditing or per-user logging.
