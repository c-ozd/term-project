# IMPORTS

import gc
import time
import json
import re
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Protocol, runtime_checkable

import chromadb
from rank_bm25 import BM25Okapi

# DATA STRUCTURES

@dataclass
class RetrievedChunk:
    """A single retrieved document with metadata and provenance scores."""
    chunk_id:       str             # unique identifier (game_aspect_node_chunkidx)
    page_content:   str             # the actual text
    metadata:       dict            # full metadata dict from the chunk
    domain:         str             # "dota2" or "lol"  extracted from metadata
    dense_rank:     int | None      # rank in BGE results (None if not retrieved by dense)
    sparse_rank:    int | None      # rank in BM25 results (None if not retrieved by sparse)
    dense_score:    float | None    # cosine similarity from ChromaDB
    sparse_score:   float | None    # BM25 score
    rrf_score:      float           # fused RRF score
    token_count:    int             # token count from metadata


@dataclass
class RetrievalOutput:
    """Complete output from the hybrid retrieval layer."""
    candidates:     list[RetrievedChunk]   # top-N after RRF fusion, sorted by rrf_score desc
    query:          str
    domain_filter:  str | None             # which domain was filtered, or None for hybrid
    dense_count:    int                    # how many docs came from dense retrieval
    sparse_count:   int                    # how many docs came from sparse retrieval
    fused_count:    int                    # total unique docs after fusion
    latency_ms:     float
    retrieval_config: dict                 # the config from router, for logging


# DENSE RETRIEVER PROTOCOL

@runtime_checkable
class DenseRetrieverProtocol(Protocol):
    """
    Any dense retriever with these methods works in HybridRetriever.

    embed_query() is exposed so HybridRetriever can compute the query
    embedding ONCE before the thread pool fans out. This matters for
    GGUFDenseRetriever: llama_cpp.Llama is not thread-safe, so running
    two dense futures that each call create_embedding() on the same
    Llama instance concurrently can crash, hang, or return corrupted
    results. Embedding once up-front eliminates that race.

    query() accepts an optional pre-computed query_embedding. When
    provided, the retriever skips embedding and goes straight to
    ChromaDB lookup — this is the path HybridRetriever takes.
    """
    def embed_query(self, query_text: str) -> list[float]:
        ...

    def query(
        self,
        query_text:      str,
        top_k:           int = 20,
        domain_filter:   dict | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[tuple[dict, float, int]]:
        ...


# BM25 INDEX BUILDER

def simple_tokenize(text: str) -> list[str]:
    """
    Basic whitespace + punctuation tokenizer for BM25.
    """
    text = text.lower()
    text = re.sub(r"'s\b", "", text)  # strip possessives before tokenizing
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens



class BM25Index:
    """
    Domain-partitioned BM25 index built from JSONL chunk files.

    The index is partitioned by domain at construction time. There is
    one BM25Index per domain (bm25_dota2, bm25_lol). This makes domain
    filtering trivial: for SINGLE_DOMAIN, query only the relevant index.
    For HYBRID, query both and merge before RRF.
    """

    def __init__(self, domain: str, chunks: list[dict]):
        self.domain = domain
        self.chunks = chunks

        # Tokenize all documents for BM25
        self.tokenized_corpus = [
            simple_tokenize(c["page_content"]) for c in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"  BM25 index [{domain}]: {len(chunks)} documents indexed")

    def query(self, query_text: str, top_k: int = 20) -> list[tuple[dict, float, int]]:
        """
        Query the BM25 index and return top-k results.
        """
        tokenized_query = simple_tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices by score
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] > 0:   # skip zero-score (no term overlap)
                results.append((self.chunks[idx], float(scores[idx]), rank))

        return results


def build_bm25_indexes(
    dota_chunks: list[dict],
    lol_chunks:  list[dict],
) -> dict[str, BM25Index]:
    """
    Build domain-partitioned BM25 indexes from chunk lists.
    """
    print("Building BM25 indexes...")
    indexes = {
        "dota2": BM25Index("dota2", dota_chunks),
        "lol":   BM25Index("lol",   lol_chunks),
    }
    print(f"BM25 indexes ready: {len(dota_chunks) + len(lol_chunks)} total documents\n")
    return indexes


#  DENSE RETRIEVAL WRAPPER (sentence-transformers, CPU)

class DenseRetriever:
    """
    Dense retriever using sentence-transformers on CPU.
    """

    def __init__(
        self,
        chroma_path:      str = "./chroma_db",
        collection_name:  str = "term_project",
        embed_model_name: str = "BAAI/bge-base-en-v1.5",
    ):
        from sentence_transformers import SentenceTransformer

        print(f"Connecting to ChromaDB at {chroma_path}...")
        self.client = chromadb.PersistentClient(path=chroma_path)

        # Open collection WITHOUT an embedding function
        self.collection = self.client.get_collection(name=collection_name)
        print(f"  Collection '{collection_name}': {self.collection.count()} documents")

        # Load the same embedding model used during injection
        print(f"  Loading embedding model: {embed_model_name}...")
        self.embed_model = SentenceTransformer(embed_model_name, device="cpu")
        print(f"  Dense retriever ready.")

    def _embed_query(self, query_text: str) -> list[float]:
        """Embed a query using the same model and settings as injection."""
        # normalize_embeddings=True must match injection script
        embedding = self.embed_model.encode(
            query_text,
            normalize_embeddings=True, # Required for cosine-similarity calculations.
        )
        return embedding.tolist()

    def embed_query(self, query_text: str) -> list[float]:
        return self._embed_query(query_text)

    def query(
        self,
        query_text:      str,
        top_k:           int = 20,
        domain_filter:   dict | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[tuple[dict, float, int]]:
        """
        Query ChromaDB and return top-k results with cosine similarity.
        """
        if query_embedding is None:
            query_embedding = self._embed_query(query_text)

        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if domain_filter is not None:
            query_params["where"] = domain_filter

        results = self.collection.query(**query_params) # That `query` method is a ChromaDB method, our query above is a wrapper around it.

        # ChromaDB returns lists of lists (one per query). We have one query.
        documents  = results["documents"][0]    if results["documents"]  else []
        metadatas  = results["metadatas"][0]    if results["metadatas"]  else []
        distances  = results["distances"][0]    if results["distances"]  else []
        ids        = results["ids"][0]          if results["ids"]        else []

        output = []
        for rank, (doc, meta, dist, doc_id) in enumerate(
            zip(documents, metadatas, distances, ids), 1
        ):
            # Collection uses hnsw:space = "cosine"
            # ChromaDB cosine distance = 1 - cosine_similarity
            similarity = 1.0 - dist

            chunk_dict = {
                "page_content": doc,
                "metadata": meta,
                "id": doc_id,
            }
            output.append((chunk_dict, similarity, rank))

        return output


# GGUF DENSE RETRIEVAL WRAPPER

class GGUFDenseRetriever:
    """
    Dense retriever using a GGUF embedding model via llama-cpp-python.
    """

    def __init__(
        self,
        chroma_path:      str,
        collection_name:  str,
        gguf_model_path:  str,
        n_gpu_layers:     int  = -1,
        n_ctx:            int  = 8192,
        n_batch:          int  = 4096,
    ):
        # ChromaDB Connection
        print(f"Connecting to ChromaDB at {chroma_path}...")
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(name=collection_name)
        print(f"  Collection '{collection_name}': {self.collection.count()} documents")

        self._gguf_model_path = gguf_model_path
        self._n_gpu_layers    = n_gpu_layers
        self._n_ctx           = n_ctx
        self._n_batch         = n_batch

        # Lock guarding all create_embedding() calls. llama_cpp.Llama is not
        # thread-safe — concurrent calls can segfault, hang, or corrupt the
        # output buffer
        self._embed_lock = threading.Lock()

        self.llm = None
        self._embed_dim = None
        self._load_llama()

    def _load_llama(self) -> None:
        """
        Instantiate the GGUF Llama embedder using stashed params.
        """
        from llama_cpp import Llama

        print(f"  Loading GGUF embedder: {self._gguf_model_path}...")
        t0 = time.perf_counter()
        self.llm = Llama(
            model_path=self._gguf_model_path,
            embedding=True,
            n_gpu_layers=self._n_gpu_layers,
            n_ctx=self._n_ctx,
            n_batch=self._n_batch,
            verbose=False,
        )
        load_ms = (time.perf_counter() - t0) * 1000

        # Detect embedding dimension (and sanity-check it matches a prior load)
        test_result = self.llm.create_embedding("dim probe")
        new_dim = len(test_result["data"][0]["embedding"])
        if self._embed_dim is not None and new_dim != self._embed_dim:
            raise RuntimeError(
                f"Embedding dim changed on reload: {self._embed_dim} → {new_dim}. "
                f"This should never happen unless the GGUF file changed on disk."
            )
        self._embed_dim = new_dim
        print(f"  GGUF embedder ready. dim={self._embed_dim}, loaded in {load_ms:.0f}ms")

    def _embed_query(self, query_text: str) -> list[float]:
        """Embed a query using the GGUF model, L2-normalized. Thread-safe."""
        if self.llm is None:
            raise RuntimeError(
                "GGUFDenseRetriever embedder is unloaded. "
                "Call .reload() before issuing another query."
            )
        with self._embed_lock:
            result = self.llm.create_embedding(query_text)
        embedding = np.array(result["data"][0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()

    def embed_query(self, query_text: str) -> list[float]:
        return self._embed_query(query_text)

    def query(
        self,
        query_text:      str,
        top_k:           int = 20,
        domain_filter:   dict | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[tuple[dict, float, int]]:
        """
        Query ChromaDB.
        """
        if query_embedding is None:
            query_embedding = self._embed_query(query_text)

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if domain_filter:
            query_kwargs["where"] = domain_filter

        raw = self.collection.query(**query_kwargs)

        results = []
        if raw["ids"] and raw["ids"][0]:
            for rank_idx, (doc_id, document, metadata, distance) in enumerate(
                zip(raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]),
                start=1,
            ):
                cosine_similarity = 1.0 - distance
                chunk_dict = {"page_content": document, "metadata": metadata, "id": doc_id}
                results.append((chunk_dict, cosine_similarity, rank_idx))

        return results

    def unload(self):
        """Free GPU memory by deleting the GGUF model."""
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
            self.llm = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            print("  GGUF embedder unloaded, GPU memory released.")

    def reload(self):
        """
        Re-instantiate the GGUF Llama embedder after a prior unload().

        Used by VRAM-constrained flows (e.g. RTX 3060 6 GB laptop) that
        must free the embedder before Layer 5 generation loads the Llama
        3.1 8B GGUF, then bring it back for the next query's retrieval.
        Idempotent: reload()-ing an already-loaded embedder is a no-op.
        """
        if self.llm is not None:
            return  # Already loaded — nothing to do
        self._load_llama()


# RECIPROCAL RANK FUSION

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float, int]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion

    RRF score for document d = sum over lists L of: 1 / (k + rank_L(d))
    """
    rrf_scores = defaultdict(float)

    for ranked_list in ranked_lists:
        for doc_id, _score, rank in ranked_list:
            rrf_scores[doc_id] += 1.0 / (k + rank)

    # Sort by RRF score descending
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

# HYBRID RETRIEVER

class HybridRetriever:
    """
    Orchestrates parallel dense (ChromaDB) and sparse (BM25) retrieval,
    applies domain filtering before fusion, and returns RRF-fused candidates
    for the cross-encoder reranker.

    Accepts ANY dense retriever conforming to DenseRetrieverProtocol —
    both the CPU-based DenseRetriever (sentence-transformers) and the
    GPU-based GGUFDenseRetriever (llama-cpp-python) work interchangeably.

    Memory note:
        BM25 indexes live in RAM (~0.5 GB for both).
        ChromaDB uses memory-mapped files (~1-2 GB).
        DenseRetriever (sentence-transformers, CPU): ~0.5 GB RAM, no VRAM.
        GGUFDenseRetriever (llama-cpp, GPU): ~0.5-1 GB VRAM with n_gpu_layers=-1.
        Call unload_dense() after retrieval is done to free GGUF GPU memory
        before loading the cross-encoder reranker.
    """
    DOMAIN_TO_GAME_FIELD = {
        "dota2": "Dota 2",
        "lol":   "League of Legends",
    }

    def __init__(
        self,
        dense_retriever:  DenseRetrieverProtocol,
        bm25_indexes:     dict[str, BM25Index],
        rrf_k:            int = 60,
        top_k_per_source: int = 10,
        top_n_fused:      int = 10,
    ):
        """
        Args:
            dense_retriever:  Any object conforming to DenseRetrieverProtocol
                              (e.g. DenseRetriever or GGUFDenseRetriever).
            bm25_indexes:     {"dota2": BM25Index, "lol": BM25Index}.
            rrf_k:            RRF constant (default 60, per Cormack et al. 2009).
            top_k_per_source: How many results to fetch from each retriever per
                              domain.
            top_n_fused:      Total chunks passed to the reranker. Split across
                              domains according to the router's dynamic allocation.
        """
        self.dense = dense_retriever
        self.bm25  = bm25_indexes
        self.rrf_k = rrf_k
        self.top_k = top_k_per_source
        self.top_n = top_n_fused

    def unload_dense(self) -> None:
        """
        Release dense retriever resources if supported.
        """
        if hasattr(self.dense, "unload") and callable(self.dense.unload):
            self.dense.unload()

    def reload_dense(self) -> None:
        """
        Re-instantiate the dense retriever after a prior unload_dense().
        """
        if hasattr(self.dense, "reload") and callable(self.dense.reload):
            self.dense.reload()

    def _make_chunk_id(self, chunk: dict) -> str:
        """
        Generate a stable ID for deduplication across dense and sparse results.

        Must match the ID format used during ChromaDB injection:
            {game}_{entity}_{full_path}_{chunk_idx}

        For BM25-only results (not in ChromaDB), we reconstruct the same
        format from metadata fields.
        """
        if "id" in chunk:
            return chunk["id"]

        meta      = chunk.get("metadata", {})
        game      = meta.get("game", "Unknown")
        full_path = meta.get("full_path", "unknown_path")
        chunk_idx = meta.get("chunk_index", 0)

        entity = (
            meta.get("hero")
            or meta.get("champion")
            or meta.get("item")
            or meta.get("mechanics")
            or "general"
        )

        return f"{game}_{entity}_{full_path}_{chunk_idx}"

    def _retrieve_dense(
        self,
        query: str,
        domain_filter: dict | None,
        top_k: int | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[tuple[str, dict, float, int]]:
        """
        Dense retrieval via ChromaDB. Returns (chunk_id, chunk_dict, score, rank).
        """
        k = top_k if top_k is not None else self.top_k

        results = self.dense.query(
            query_text=query,
            top_k=k,
            domain_filter=domain_filter,
            query_embedding=query_embedding,
        )
        return [
            (self._make_chunk_id(chunk), chunk, score, rank)
            for chunk, score, rank in results
        ]

    def _retrieve_sparse(
        self,
        query: str,
        domain: str,
        top_k: int | None = None,
    ) -> list[tuple[str, dict, float, int]]:
        """
        Sparse retrieval via BM25 for a SINGLE domain.
        """
        k = top_k if top_k is not None else self.top_k

        if domain not in self.bm25:
            return []

        results = self.bm25[domain].query(query, top_k=k)
        return [
            (self._make_chunk_id(chunk), chunk, score, rank)
            for chunk, score, rank in results
        ]

    def _fuse_single_domain(
        self,
        dense_results:  list[tuple[str, dict, float, int]],
        sparse_results: list[tuple[str, dict, float, int]],
        top_n: int,
    ) -> tuple[
        list[tuple[str, float]],
        dict[str, dict],
        dict[str, tuple[float, int]],
        dict[str, tuple[float, int]],
    ]:
        """
        Run RRF fusion for one domain's dense + sparse results.
        """
        chunk_store   = {}
        dense_lookup  = {}
        sparse_lookup = {}

        for cid, chunk, score, rank in dense_results:
            dense_lookup[cid] = (score, rank)
            chunk_store[cid]  = chunk

        for cid, chunk, score, rank in sparse_results:
            sparse_lookup[cid] = (score, rank)
            if cid not in chunk_store:
                chunk_store[cid] = chunk

        dense_for_rrf  = [(cid, score, rank) for cid, _, score, rank in dense_results]
        sparse_for_rrf = [(cid, score, rank) for cid, _, score, rank in sparse_results]

        fused = reciprocal_rank_fusion(
            [dense_for_rrf, sparse_for_rrf],
            k=self.rrf_k,
        )

        return fused[:top_n], chunk_store, dense_lookup, sparse_lookup

    def _retrieve_hybrid(
        self,
        query: str,
        collections: list[str],
        allocation: dict[str, int],
    ) -> tuple[list, dict, dict, dict, int, int]:
        """
        All retrievals (dense + sparse for both domains) run in parallel.
        RRF is computed per-domain so BM25 scores never cross domain
        boundaries — the first cross-domain comparison happens at the
        reranker, where cross-encoder scores are directly comparable.

        Args:
            query:       The user query.
            collections: List of domain keys to retrieve from (always both).
            allocation:  {"dota2": N, "lol": M} where N + M == total budget.

        Returns:
            fused_all:      Combined fused results from both domains.
            chunk_store:    Merged chunk_id -> chunk_dict mapping.
            dense_lookup:   Merged chunk_id -> (score, rank) for dense hits.
            sparse_lookup:  Merged chunk_id -> (score, rank) for sparse hits.
            dense_count:    Total dense results across both domains.
            sparse_count:   Total sparse results across both domains.
        """
        query_embedding = self.dense.embed_query(query)

        # Parallel retrieval of dense and sparse results for both domains
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for domain in collections:
                chroma_game   = self.DOMAIN_TO_GAME_FIELD.get(domain, domain)
                chroma_filter = {"game": chroma_game}

                futures[(domain, "dense")] = executor.submit(
                    self._retrieve_dense, query, chroma_filter, self.top_k,
                    query_embedding,
                )
                futures[(domain, "sparse")] = executor.submit(
                    self._retrieve_sparse, query, domain, self.top_k
                )

            results = {key: future.result() for key, future in futures.items()}

        # Per-domain RRF fusion
        fused_all      = []
        chunk_store    = {}
        dense_lookup   = {}
        sparse_lookup  = {}
        dense_count    = 0
        sparse_count   = 0

        for domain in collections:
            dense_results  = results.get((domain, "dense"),  [])
            sparse_results = results.get((domain, "sparse"), [])

            dense_count  += len(dense_results)
            sparse_count += len(sparse_results)

            # Each domain's RRF output is capped by its allocation slot count
            per_domain_n = allocation.get(domain, self.top_n // 2)

            fused, cs, dl, sl = self._fuse_single_domain(
                dense_results, sparse_results, top_n=per_domain_n
            )

            fused_all.extend(fused)
            chunk_store.update(cs)
            dense_lookup.update(dl)
            sparse_lookup.update(sl)

        return fused_all, chunk_store, dense_lookup, sparse_lookup, dense_count, sparse_count

    def retrieve(self, query: str, retrieval_config: dict) -> RetrievalOutput:
        """
        Execute hybrid retrieval based on a retrieval config dict.
        """
        t0 = time.perf_counter()

        # Early exit: CLARIFICATION → no retrieval
        if not retrieval_config.get("should_retrieve", True):
            return RetrievalOutput(
                candidates=[],
                query=query,
                domain_filter=None,
                dense_count=0,
                sparse_count=0,
                fused_count=0,
                latency_ms=0.0,
                retrieval_config=retrieval_config,
            )

        collections = retrieval_config.get("collections", ["dota2", "lol"])
        allocation  = retrieval_config.get("allocation", {
            d: self.top_n // len(collections) for d in collections
        })

        # Always dual-domain retrieval with dynamic allocation
        fused, chunk_store, dense_lookup, sparse_lookup, \
            dense_count, sparse_count = self._retrieve_hybrid(
                query, collections, allocation
            )

        # Determine majority domain for logging
        majority_domain = max(allocation, key=allocation.get) if allocation else None

        # Build final candidate list
        candidates = []
        for cid, rrf_score in fused:
            chunk = chunk_store[cid]
            meta  = chunk.get("metadata", {})

            d_score, d_rank = dense_lookup.get(cid, (None, None))
            s_score, s_rank = sparse_lookup.get(cid, (None, None))

            # Extract domain 
            raw_game = str(meta.get("game", "")).strip()
            domain = "unknown"
            for router_key, chroma_val in self.DOMAIN_TO_GAME_FIELD.items():
                if raw_game.lower() == chroma_val.lower():
                    domain = router_key
                    break

            candidates.append(RetrievedChunk(
                chunk_id=cid,
                page_content=chunk["page_content"],
                metadata=meta,
                domain=domain,
                dense_rank=d_rank,
                sparse_rank=s_rank,
                dense_score=round(d_score, 4)  if d_score  is not None else None,
                sparse_score=round(s_score, 4) if s_score is not None else None,
                rrf_score=round(rrf_score, 6),
                token_count=meta.get("token_count") or chunk.get("token_count") or 0,
            ))

        latency_ms = (time.perf_counter() - t0) * 1000

        return RetrievalOutput(
            candidates=candidates,
            query=query,
            domain_filter=majority_domain,
            dense_count=dense_count,
            sparse_count=sparse_count,
            fused_count=len(candidates),
            latency_ms=round(latency_ms, 1),
            retrieval_config=retrieval_config,
        )


# DEFAULT RETRIEVAL CONFIG

DEFAULT_RETRIEVAL_CONFIG = {
    "should_retrieve": True,
    "collections":     ["dota2", "lol"],
    "allocation":      {"dota2": 5, "lol": 5},
    "reason":          "fixed config — no router, global RRF top 10",
}


# GLOBAL RRF HYBRID RETRIEVER (BASELINE)

class GlobalRRFHybridRetriever(HybridRetriever):
    """
    The project's baseline retriever: single global RRF fusion across
    both domains, taking the top 10 globally.

    The constructor and external interface match HybridRetriever exactly,
    so existing notebook code can swap in this class with a single import
    change. Use the module-level DEFAULT_RETRIEVAL_CONFIG to drive
    retrieve() in unrouted pipelines.
    """

    # Hardcoded global budget
    GLOBAL_TOP_N = 10

    def _retrieve_hybrid(
        self,
        query: str,
        collections: list[str],
        allocation: dict[str, int],   # accepted but IGNORED
    ) -> tuple[list, dict, dict, dict, int, int]:
        """
        Override of HybridRetriever._retrieve_hybrid.

        Identical to the parent up to the parallel retrieval step, then
        diverges: instead of two per-domain RRF calls, we collect all
        four ranked lists into a single global RRF and take the top 10.

        Args:
            query:       The user query.
            collections: Domain keys (always ["dota2", "lol"]).
            allocation:  Accepted for parent-class interface compatibility
                         but ignored — Global RRF uses a hardcoded
                         GLOBAL_TOP_N budget.

        Returns:
            Same 6-tuple shape as the parent's _retrieve_hybrid:
            (fused_all, chunk_store, dense_lookup, sparse_lookup,
             dense_count, sparse_count).
        """
        query_embedding = self.dense.embed_query(query)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for domain in collections:
                chroma_game   = self.DOMAIN_TO_GAME_FIELD.get(domain, domain)
                chroma_filter = {"game": chroma_game}

                futures[(domain, "dense")] = executor.submit(
                    self._retrieve_dense, query, chroma_filter, self.top_k,
                    query_embedding,
                )
                futures[(domain, "sparse")] = executor.submit(
                    self._retrieve_sparse, query, domain, self.top_k
                )

            results = {key: future.result() for key, future in futures.items()}

        chunk_store      = {}
        dense_lookup     = {}
        sparse_lookup    = {}
        all_ranked_lists = []
        dense_count      = 0
        sparse_count     = 0

        for domain in collections:
            dense_results  = results.get((domain, "dense"),  [])
            sparse_results = results.get((domain, "sparse"), [])

            dense_count  += len(dense_results)
            sparse_count += len(sparse_results)

            for cid, chunk, score, rank in dense_results:
                dense_lookup[cid] = (score, rank)
                chunk_store[cid]  = chunk

            for cid, chunk, score, rank in sparse_results:
                sparse_lookup[cid] = (score, rank)
                if cid not in chunk_store:
                    chunk_store[cid] = chunk

            all_ranked_lists.append(
                [(cid, score, rank) for cid, _, score, rank in dense_results]
            )
            all_ranked_lists.append(
                [(cid, score, rank) for cid, _, score, rank in sparse_results]
            )

        fused_global = reciprocal_rank_fusion(all_ranked_lists, k=self.rrf_k)
        fused_all    = fused_global[:self.GLOBAL_TOP_N]

        return (
            fused_all,
            chunk_store,
            dense_lookup,
            sparse_lookup,
            dense_count,
            sparse_count,
        )


# DATA LOADING HELPERS

def load_jsonl(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


if __name__ == "__main__":
    dota_chunks = load_jsonl(
        r"your_path"
    )
    lol_chunks = load_jsonl(
        r"your_path"
    )

    bm25_indexes = build_bm25_indexes(dota_chunks, lol_chunks)
