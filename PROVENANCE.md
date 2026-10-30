# Provenance and disclosure

EvidenceMemory is an independent project implementation developed with AI coding assistance. It combines established retrieval techniques with deterministic dialogue-aware source evidence composition. It is not submitted as a reproduction of a single external paper.

## Design references

- [Graphiti](https://github.com/getzep/graphiti), by its maintainers: hybrid lexical/dense retrieval, rank fusion and group isolation informed the design. The project does not vendor Graphiti source.
- [LlamaIndex sentence-window retrieval](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo/), by the LlamaIndex contributors: fine retrieval followed by returning surrounding original evidence informed the source-window approach. LlamaIndex source is not vendored.
- BM25, cosine similarity, reciprocal rank fusion and maximal marginal relevance are standard components implemented in the shared retrieval core. Dependencies are declared in `runtime/pyproject.toml` and constrained by `runtime/requirements-service.txt`; external dependencies retain their own licenses.

## Implementation changes

The service adds stable raw-message projection, incremental vector preparation, atomic idempotent ingestion, generation-scoped indexes, bounded response/provenance caches, deterministic speaker/turn recovery, split-turn continuity scoring, source-window completion and session-frame packing. The v8 change separates the journal repository and evidence policy and replaces per-vector SELECTs with bounded batches of up to 200.

The frozen shared core and text adapter remain byte-identical to the deployed release. Their presence in `runtime/` is for running and reviewing this service; other domain adapters are not included in the HTTP submission.

## Data and evaluation disclosure

Public LoCoMo-Refined and LongMemEval-S samples were repeatedly used during development. The reported small-sample retrieval figures are development results and must not be treated as blind generalization or official scores. Runtime selection does not load answer labels, benchmark question IDs, datasets or saved evaluation results.

Datasets, cached provider vectors, user memories, private evaluation results and credentials are intentionally excluded. No official private competition data was used. Methods and operational constraints are documented for organizer review and reproducibility.

Project code is distributed under the [MIT License](LICENSE). External dependencies retain their own licenses; no third-party framework source is vendored or relicensed by this disclosure.
