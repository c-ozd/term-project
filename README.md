# Term Project

Multi-layer retrieval-augmented generation pipeline for cross-domain
knowledge isolation between Dota 2 and League of Legends wikis. MSc
non-thesis project.


## Hardware requirements

- GPU with ≥6 GB VRAM (developed on RTX 3060 Laptop, 6 GB)
- 16 GB RAM
- ~15 GB disk for model weights and ChromaDB index

The pipeline uses a GPU shuttle pattern (load → use → unload) because
no single layer fits in VRAM alongside another. On larger cards this
overhead is unnecessary but harmless.

## Repository structure

| Folder | Contents |
|---|---|
| `00-Scraping` | Wiki scrapers for Dota 2 and LoL |
| `01-Chunking` | JSONL chunker scripts |
| `02-Embedding` | ChromaDB injection script |
| `03-Retrieval` | hybrid BM25 + dense retrieval with RRF |
| `04-Reranker` | bge-reranker-large cross-encoder + threshold gating |
| `05-Generation` | Llama 3.1 8B generation|
| `06-Faithfulness Check` | NLI faithfulness scoring |
| `07-Confidence Orchestrator` | Tier assignment from weighted signals |
| `08-Helper Functions` | Shared GPU utilities, pipeline runner |
| `09-RAGAS` | Offline evaluation: RAGAS, Hit@k, MRR |

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Download model weights (not redistributed in this repo):
   - Llama 3.1 8B Q4_K_M GGUF
   - Qwen3-Embedding-4B Q8 GGUF (HuggingFace)
   - bge-reranker-large (HuggingFace)
   - deberta-v3-large-zeroshot-v2.0 (HuggingFace)
3. Update the hardcoded paths and the
   chunker scripts to point to your local model and data locations.
4. Build the BM25 indices and ChromaDB collection from
   `01-Chunking` and `02-Embedding`.


## Ablation variants

The 12 ablation variants reported are not
included in this repository; see the report for derivation
instructions from the main pipeline.

## License

Source code: MIT (see [LICENSE](LICENSE)).
Data artifacts: see [DATA_LICENSE.md](DATA_LICENSE.md).
