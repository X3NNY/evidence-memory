# EvidenceMemory

EvidenceMemory is a source-grounded text memory service for the Agent Memory Challenge. It implements authenticated synchronous Add/Search/Health APIs and returns original evidence rather than generating final answers.

The current submission is **text-http-hybrid-v8** for the **Textual Memory / Open-source Methods (Academic)** track. Code and multimodal HTTP adapters are not part of this submission.

## Method

- BM25 and `text-embedding-v4` (1024-dimensional) dense retrieval, combined with reciprocal rank fusion and MMR.
- Deterministic dialogue parsing, fine passage ranking and complete source-chunk retrieval.
- Source windows repair split utterances; session frames remove repeated headers and overlaps, preserving source offsets and marking gaps.
- Atomic, idempotent SQLite WAL ingestion with FULL synchronous commits, generation-scoped user indexes and durable user/model/type-isolated vectors.
- Bounded immutable response caching; evidence provenance expires with cached results.

No LLM is used in Add or Search. Search does not answer questions. The implementation has no configured person names, gold answers or benchmark question-ID rules. The declared 48,000-character budget governs evidence enrichment; it is not a hard cap on the full Search response.

## Frozen runtime

`runtime/` contains the exact 52-file deployment source bundle plus its `CODE_MANIFEST.json`. Its bytes are preserved in Git, including line endings. Runtime code is unchanged from the deployed v8 release. The root README, Compose wrapper, example configuration and synthetic smoke helper are publication material.

Original deployment archive SHA-256:

```text
ea2c9e1f1d1c1cdd42f7b67a1cdc24801f821bb00d071103e2febdc2285df2bb
```

The immutable Git commit submitted to the competition identifies this public source snapshot. `RELEASE.json` records the corresponding deployed image ID.

## Run

Use Python 3.11, or build the included Docker image. Copy `.env.example` to a private `.env`, create a separate random Memory System Key file and configure your own provider credentials. These two credentials have different purposes: the Memory System Key authenticates Add/Search callers; the provider key is used only by the embedding client.

```sh
docker compose up -d --build
curl http://127.0.0.1:8787/health
```

Compose reads the path in `AML_MEMORY_SYSTEM_KEY_FILE`, mounts it into the container and keeps the database in a persistent volume. Expose the loopback listener through an HTTPS reverse proxy. The participant deployment uses a dedicated `/agent-memory/` prefix; strip that prefix before forwarding to the service.

The sample configuration explicitly enables remote embeddings. New raw documents and uncached queries incur provider usage. `cache_only` fails on missing vectors; it does not silently switch to BM25. No benchmark vectors or credentials are distributed here.

## API

`GET /health` is public. Add and Search require `Authorization: Bearer <Memory System Key>` when the default auth scheme is used.

```json
{"request_id":"r1","user_id":"u1","session_id":"s1","messages":[{"role":"user","content":"Original remembered source text."}]}
```

Send the above to `POST /add`. A successful response echoes `success`, `request_id`, `user_id` and `session_id`. Identical retries are idempotent; conflicting request-ID reuse returns 409.

```json
{"query":"What was remembered?","user_id":"u1","top_k":10}
```

Send the above to `POST /search`. The response is `{"data":[{"id":"...","content":"...","score":0.1,"created_at":"..."}]}`. `top_k` is an integer from 1 to 100. Unknown users return an empty list without calling the embedding provider. Retryable storage/provider failures return 503.

## Offline synthetic smoke

```sh
python -m pip install -c runtime/requirements-service.txt './runtime[service,lexical]'
python scripts/smoke_runtime.py
```

This helper runs two local HTTP processes with synthetic vectors, checks Add/Search, auth, idempotency, incremental visibility, failed-write behavior and vector reuse after restart, and makes no model requests. The deployed Linux image passed its 18 checks with Docker networking disabled.

## Validation and limits

The development workspace passed 433 tests. Its repeatedly reused public-data sample contains 50 LoCoMo-Refined and 50 LongMemEval-S questions; it is not a blind set. v8 preserves all 100 complete v7 responses and source intervals. LoCoMo's 97.50% actual citation-turn token coverage at 48k is a retrieval metric, not a QA or competition score; the original 22/50 QA result was retained. These historical experimental files are not part of this runtime-only repository.

Public HTTPS, authentication, empty-scope Search, container restart and volume persistence were verified on the participant server. Provider DNS/TLS connectivity was checked without an HTTP model request. No real provider request or official platform Smoke/Full evaluation has been run as part of deployment.

The participant server allocates 1.5 CPU and 1536 MiB RAM and retains 2 user indexes. Start with Add/Search concurrency 1. Full-scale capacity is unverified; cold indexes still rebuild on generation changes. This repository does not claim a new QA score or official ranking.

See [provenance](PROVENANCE.md), [runtime architecture](runtime/docs/service_v8.md), [retrieval policy](runtime/docs/service_v7.md) and [deployment reference](runtime/docs/deploy_text.md). The runtime documents retain their original historical snapshots; this README records the later deployment and publication status.
