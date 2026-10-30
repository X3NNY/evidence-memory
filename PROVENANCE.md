# Provenance and disclosure

The historical `text-http-lexical-v1` profile removes dense retrieval and all model-provider use. It reuses the shared source projection/evidence pipeline and adds fixed-weight raw/body-stemmed BM25 fusion for ordinary text. The earlier hybrid source remains frozen in `runtime/`; the new runtime is `runtime-lexical/`. No benchmark labels or question-ID conditions are introduced by this change.

EvidenceMemory is an independent project implementation developed with AI coding assistance. It combines established retrieval techniques with deterministic dialogue-aware source evidence composition. It is not submitted as a reproduction of a single external paper.

## Design references

- [Graphiti](https://github.com/getzep/graphiti), by its maintainers: hybrid lexical/dense retrieval, rank fusion and group isolation informed the design. The project does not vendor Graphiti source.
- [LlamaIndex sentence-window retrieval](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo/), by the LlamaIndex contributors: fine retrieval followed by returning surrounding original evidence informed the source-window approach. LlamaIndex source is not vendored.
- BM25, cosine similarity, reciprocal rank fusion and maximal marginal relevance are standard components implemented in the shared retrieval core. Dependencies are declared in `runtime/pyproject.toml` and constrained by `runtime/requirements-service.txt`; external dependencies retain their own licenses.

## Implementation changes

The service adds stable raw-message projection, incremental vector preparation, atomic idempotent ingestion, generation-scoped indexes, bounded response/provenance caches, deterministic speaker/turn recovery, split-turn continuity scoring, source-window completion and session-frame packing. The v8 change separates the journal repository and evidence policy and replaces per-vector SELECTs with bounded batches of up to 200.

The frozen shared core and text adapter remain byte-identical to the deployed release. Their presence in `runtime/` is for running and reviewing this service; other domain adapters were not included in that historical HTTP submission.

## Data and evaluation disclosure

Public LoCoMo-Refined and LongMemEval-S samples were repeatedly used during development. The reported small-sample retrieval figures are development results and must not be treated as blind generalization or official scores. Runtime selection does not load answer labels, benchmark question IDs, datasets or saved evaluation results.

Datasets, cached provider vectors, user memories, private evaluation results and credentials are intentionally excluded. No official private competition data was used. Methods and operational constraints are documented for organizer review and reproducibility.

Project code is distributed under the [MIT License](LICENSE). External dependencies retain their own licenses; no third-party framework source is vendored or relicensed by this disclosure.


## Three-track extension

`multitrack-lexical-v1` adds an independently implemented identifier-aware lexical channel, typed ordered multimodal content, atomic original-media storage, per-user media isolation and bounded media admission. [Aider repository maps](https://aider.chat/docs/repomap.html) motivate preserving code symbols and paths; [LlamaIndex auto-merging](https://developers.llamaindex.ai/python/examples/retrievers/auto_merging_retriever/) motivates keeping source context. No source from either project is vendored. The image path uses supplied captions and exact original-image identity, without learned visual embeddings or generated descriptions.

One fixed candidate was checked on50 fresh questions per public dataset after code and input selection were frozen. Labels were disabled until every candidate retrieval completed. Previously used repositories/scenarios remain represented; this is not an unseen-domain claim. The runtime does not import public-data loaders or evaluation labels. Official private Smoke results were not used for tuning and are not published here. The earlier text hard100 output is unchanged.
