# EvidenceMemory

EvidenceMemory provides authenticated Add/Search/Health APIs and returns original source evidence. The current deployment profile is **text-http-lexical-v1**, a **zero-model-API** text memory service.

## Zero-model profile

- Add persists raw messages atomically and idempotently; no embedding work is performed.
- Search uses BM25. Ordinary text combines raw and body-stemmed lexical rankings with fixed 0.5/0.5 RRF. Explicit participant queries use deterministic dialogue, passage and continuity policies.
- Original source windows and session frames complete split utterances and deduplicate repeated headers, with source-offset provenance.
- No Embedding model, LLM, pretrained reranker, model credentials or old vector cache is required. New messages and new queries work without model calls.
- SQLite WAL/FULL commits, generation-based user indexes and bounded immutable response/provenance caches remain in place.

Model API fees are zero for this profile. Hosting, storage and bandwidth are still participant costs; platform Answer/Eval are handled by the competition. Search does not generate final answers or load benchmark labels and question-ID rules.

## Run

Generate a separate random Memory System Key file for authenticating service callers. Set its path, then build the image:

```sh
export AML_MEMORY_SYSTEM_KEY_FILE=/absolute/path/to/memory-system-key.txt
docker compose up -d --build
curl http://127.0.0.1:8788/health
```

The default Compose profile builds `runtime-lexical/`, sets `AML_RETRIEVER=lexical` and `AML_ALLOW_EMBEDDING_API=0`, and mounts no model credentials. Its persistent volume is separate from hybrid-v8 data. Put HTTPS in front of the loopback listener. The runtime CLI blocks external outbound connections and DNS while permitting incoming requests and local health checks.

Health reports `version=text-http-lexical-v1`, `retriever=lexical`, `embedding_mode=none`. Add/Search use `Authorization: Bearer <Memory System Key>`.

```json
{"request_id":"r1","user_id":"u1","session_id":"s1","messages":[{"role":"user","content":"Original remembered text."}]}
```

Send the above to `POST /add`. Successful writes echo success and the three envelope IDs. Identical retries are idempotent; conflicting retries return 409.

```json
{"query":"What was remembered?","user_id":"u1","top_k":10}
```

Send the above to `POST /search`. Responses contain only `id`, `content`, `score`, `created_at` inside `data`. `top_k` is 1–100; unknown users return an empty list. The 48,000-character policy budget governs evidence enrichment, not a hard cap on the whole HTTP response.

## Source and reproducibility

`runtime-lexical/` is the frozen zero-model source bundle, including `CODE_MANIFEST.json`. `runtime/` and tag `text-http-hybrid-v8` preserve the earlier hybrid snapshot. Both runtime trees retain exact deployment bytes and line endings in Git. The root README, Compose wrapper, example and helper scripts are publication material.

Zero-model release archive SHA-256:

```text
6dc05bcd28d3be02ee6d2364c9fbb907b6489b03f1f8468ee14ac762c7350cae
```

```sh
python -m pip install -c runtime-lexical/requirements-service.txt './runtime-lexical[service,lexical]'
python runtime-lexical/scripts/smoke_lexical_service.py
```

The helper checks authenticated HTTP ingestion, lexical retrieval, unseen queries without models, idempotency, storage failure and restart behavior. The service must not construct a model provider or store vectors in this mode.

## Development evaluation

The development workspace passed 439 tests. Only the existing public-data hard100 was used: 50 LoCoMo-Refined and 50 LongMemEval-S questions, including 3 abstention examples. This sample has been repeatedly used in development and is not blind evaluation.

| Actual citation-turn token coverage, same 48k budget | Frozen hybrid v8 | Zero-model lexical v1 |
|---|---:|---:|
| LoCoMo-Refined | 97.50% | 87.83% |
| LongMemEval-S, 47 evidence-bearing questions | 77.32% | 50.26% |

These are retrieval coverage metrics, not QA or official scores. The free profile has a real quality tradeoff. All 100 hybrid-v8 responses stayed identical in compatibility regression; its scores cannot be attributed to the new profile. A separate version binding and official Smoke are required.

The zero-model replay performed zero vector reads, stored zero vectors and constructed zero model clients. Gold labels are used only by the evaluator, not the service. Datasets, vector caches, user memories, credentials and private evaluation results are excluded from this repository.

The earlier deployment's server allocation is 1.5 CPU / 1536 MiB. Platform concurrency has a minimum of 16; use proxy queueing to leave backend connection headroom. Full private-data capacity is not guaranteed by small development checks, and Full has not been started by this change workflow.

See [method disclosure](PROVENANCE.md) and [zero-model service notes](runtime-lexical/docs/service_lexical_v1.md). Code and multimodal HTTP adapters are outside this submission. Project code is [MIT licensed](LICENSE); dependencies retain their own licenses.
