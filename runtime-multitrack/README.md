# EvidenceMemory: multitrack-lexical-v1

One zero-model Add/Search service for Textual, Coding and Multimodal memory. The service returns original evidence. The evaluation platform supplies answering, judging and code execution. This release does not require a model key, embedding cache or downloaded model weights.

Text retrieval preserves the frozen lexical-v1 pipeline. Code-like source records use BM25 with a fixed, equal-weight identifier-token channel. CamelCase, snake_case and file paths are indexed alongside their intact forms. Multimodal retrieval uses supplied text/captions, Porter stemming and bounded same-session adjacency; exact image-byte identity can match a query image to its source. This is not semantic image understanding. Ordered text/image parts are stored atomically and original image bytes are returned without modification.

## Run

Set `AML_MEMORY_SYSTEM_KEY_FILE` to an absolute path containing a dedicated Add/Search authentication token. Run `docker compose up -d --build`. Check `http://127.0.0.1:8789/health`. Use HTTPS in front of the service. POST `/add` and `/search` use the official AML contract and Bearer authentication.

SQLite persistence is in a named volume. Each user_id has an isolated scope and generation cache. Requests are idempotent by request_id and original payload; conflicting retries return409. Writes and original media commit together. The service accepts ordered inline JPEG/PNG/WebP parts, with10MiB per image and30MiB media per Add or Search response. Remote image URLs are rejected. Request bodies are limited to48MiB. Two requests may buffer/process concurrently; additional requests queue before their bodies are buffered. The proxy should cap backend connections at14 for the process's16-connection limit.

Run the bundled synthetic TCP/restart check with `python scripts/smoke_multitrack_service.py --output /tmp/fresh-smoke-dir`. It needs no external network and creates only synthetic records. Production data is never used by that test.

## Evidence and limitations

Textual hard100 compatibility:100/100 exact lexical-v1 responses. Original44-point local QA result and frozen artifacts preserved. Public fresh-question comparison froze code/selection before reference loading. SWE-ContextBench Recall@10:93% BM25 to95% identifier fusion, with98% budget source recall for both. Mem-Gallery Recall@10:73.96% BM25 to85.31% existing stemmed/adjacency method; budget image recall92.59% to100%. Each dataset has50 new questions, but uses previously seen repositories/scenarios. These are reference-source retrieval proxies, not official QA or coding success scores. No per-question rules or private evaluation labels are loaded by runtime.

This is independently implemented, informed by established techniques and the designs described by [Aider's repository map](https://aider.chat/docs/repomap.html), [LlamaIndex auto-merging retrieval](https://developers.llamaindex.ai/python/examples/retrievers/auto_merging_retriever/), [Graphiti](https://github.com/getzep/graphiti) and [Mem0](https://github.com/mem0ai/mem0). Their source code is not vendored. Their authors retain ownership of their projects. Code-symbol indexing, provenance and source-context retention are the relevant ideas; paid extraction/model components are not part of this release.

Media storage and response bandwidth can dominate total cost even with zero model API charges. Capacity must be checked against the actual track before Full. Credentials, benchmark data, vectors and private results are excluded from this source package. Project code is MIT licensed; dependencies retain their original licenses.
