import json
import glob
import os
import gc
import time
import numpy as np
import chromadb
from llama_cpp import Llama


MODEL_PATH       = r"your_path"
COLLECTION_NAME  = "your_collection_name"
MODEL_LABEL      = "your_model_label"

DB_PATH          = r"your_path"
DATA_FOLDER      = r"your_path"

N_GPU_LAYERS     = -1      
N_CTX            = 8192     
BATCH_SIZE       = 32 


# PART 1: LOAD GGUF MODEL ON GPU


def load_gguf_embedder(model_path: str, n_gpu_layers: int, n_ctx: int) -> Llama:
    """Load a GGUF embedding model with GPU acceleration."""
    print(f"Loading GGUF embedding model: {model_path}")
    print(f"  GPU layers: {n_gpu_layers}, context: {n_ctx}")
    t0 = time.perf_counter()

    llm = Llama(
        model_path=model_path,
        embedding=True,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False,
    )

    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Model loaded in {load_ms:.0f}ms")
    return llm


def embed_texts(llm: Llama, texts: list[str], normalize: bool = True) -> list[list[float]]:
    raw_embeddings = []
    for text in texts:
        result = llm.create_embedding(text)
        raw_embeddings.append(result["data"][0]["embedding"])

    if normalize:
        emb_array = np.array(raw_embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        emb_array = emb_array / norms
        return emb_array.tolist()

    return raw_embeddings


def count_tokens(llm: Llama, text: str) -> int:
    """Count tokens using the GGUF model's built-in tokenizer."""
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=False)
    return len(tokens)


def free_vram():
    """Release all GPU memory back to the OS."""
    gc.collect()
    if hasattr(gc, "collect"):
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# PART 2: CHROMADB SETUP

def setup_collection(db_path: str, collection_name: str):
    """Create or open a ChromaDB collection."""
    print(f"Connecting to ChromaDB at {db_path}...")
    client = chromadb.PersistentClient(path=db_path)

    # Each GGUF model gets its own collection (different dimensions)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"  Collection '{collection_name}': {collection.count()} existing documents")
    return collection


# PART 3: INGESTION PIPELINE

def ingest_moba_data(
    llm:            Llama,
    collection,
    directory_path: str,
    batch_size:     int = 32,
):
    """
    Ingest JSONL chunks into ChromaDB using GGUF embeddings.

    """
    search_pattern = os.path.join(directory_path, "*.jsonl")
    jsonl_files = glob.glob(search_pattern)

    if not jsonl_files:
        print(f"No JSONL files found in: {directory_path}")
        return

    # Dimension check
    test_emb = embed_texts(llm, ["dimension probe"], normalize=False)
    embed_dim = len(test_emb[0])
    print(f"  Embedding dimension: {embed_dim}")

    documents = []
    metadatas = []
    ids = []
    total_processed = 0
    total_time_embed = 0.0

    for filepath in jsonl_files:
        print(f"\nProcessing file: {os.path.basename(filepath)}")

        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                chunk = json.loads(line)

                # 1. Text Content
                documents.append(chunk["page_content"])

                # 2. Metadata Sanitization
                raw_meta = chunk.get("metadata", {})
                sanitized_meta = {}

                for k, v in raw_meta.items():
                    if v is None:
                        continue
                    safe_key = str(k).lower()
                    if isinstance(v, (str, int, float, bool)):
                        sanitized_meta[safe_key] = v
                    else:
                        sanitized_meta[safe_key] = str(v)

                # Token count using GGUF model's tokenizer
                page_content = chunk["page_content"]
                computed_tokens = count_tokens(llm, page_content)
                sanitized_meta["token_count"] = computed_tokens

                metadatas.append(sanitized_meta)

                # 3. Unique ID
                game = sanitized_meta.get("game", "Unknown")
                full_path = sanitized_meta.get("full_path", "unknown_path")
                chunk_idx = sanitized_meta.get("chunk_index", 0)

                entity = (
                    sanitized_meta.get("hero") or
                    sanitized_meta.get("champion") or
                    sanitized_meta.get("item") or
                    sanitized_meta.get("mechanics") or
                    "general"
                )

                unique_id = f"{game}_{entity}_{full_path}_{chunk_idx}"
                ids.append(unique_id)

                # Batch upsert
                if len(documents) >= batch_size:
                    print(f"  Embedding & upserting batch of {len(documents)}... "
                          f"(Total so far: {total_processed + len(documents)})")

                    t0 = time.perf_counter()
                    embeddings = embed_texts(llm, documents, normalize=True)
                    embed_ms = (time.perf_counter() - t0) * 1000
                    total_time_embed += embed_ms

                    avg_ms = embed_ms / len(documents)
                    print(f"    Embedded in {embed_ms:.0f}ms ({avg_ms:.1f}ms/doc)")

                    collection.upsert(
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )

                    total_processed += len(documents)
                    documents.clear()
                    metadatas.clear()
                    ids.clear()

        # Flush leftovers
        if documents:
            print(f"  Embedding & upserting final {len(documents)} leftovers...")

            t0 = time.perf_counter()
            embeddings = embed_texts(llm, documents, normalize=True)
            embed_ms = (time.perf_counter() - t0) * 1000
            total_time_embed += embed_ms

            collection.upsert(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            total_processed += len(documents)
            documents.clear()
            metadatas.clear()
            ids.clear()

    print(f"\n{'='*60}")
    print(f"   Success! Model: {MODEL_LABEL}")
    print(f"   Total chunks processed: {total_processed}")
    print(f"   Collection '{collection.name}': {collection.count()} documents")
    print(f"   Total embedding time: {total_time_embed/1000:.1f}s")
    print(f"   Avg per document: {total_time_embed/max(total_processed,1):.1f}ms")
    print(f"   Embedding dimension: {embed_dim}")
    print(f"{'='*60}")



if __name__ == "__main__":
    # Load model on GPU
    llm = load_gguf_embedder(MODEL_PATH, N_GPU_LAYERS, N_CTX)

    # Setup collection (new collection for this model)
    collection = setup_collection(DB_PATH, COLLECTION_NAME)

    # Run ingestion
    ingest_moba_data(llm, collection, DATA_FOLDER, batch_size=BATCH_SIZE)

    # Free GPU memory after ingestion
    del llm
    free_vram()
    
    # Load model on GPU
    llm = load_gguf_embedder(MODEL_PATH_2, N_GPU_LAYERS, N_CTX)

    # Setup collection (new collection for this model)
    collection = setup_collection(DB_PATH_2, COLLECTION_NAME_2)

    # Run ingestion
    ingest_moba_data(llm, collection, DATA_FOLDER, batch_size=BATCH_SIZE)

    # Free GPU memory after ingestion
    del llm
    free_vram()
    print("\nGPU memory released. Ready for other layers.")
