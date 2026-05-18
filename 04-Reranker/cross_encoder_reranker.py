# IMPORTS
import math
import re
import time
import pandas as pd
import torch
import numpy as np
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Layer 3
from hybrid_retriever import RetrievalOutput, RetrievedChunk

# Shared GPU helpers
from gpu_utils import free_vram, gpu_available

# DATA STRUCTURES
 
@dataclass
class RerankResult:
    """A single document after cross-encoder scoring."""
    chunk:              RetrievedChunk    # original chunk
    reranker_score:     float             # normalized cross-encoder score [0, 1] (sigmoid)
    raw_reranker_score: float             # raw logit before sigmoid normalization (for temperature tuning)
    passed_gate:        bool              # True if score >= threshold
    context_position:   int | None        # position in final context (after placement), None if gated out
 
 
@dataclass
class RerankerOutput:
    """Complete output from the reranking + gating + placement pipeline."""
    # Documents that passed threshold gating, in context-assembly order
    # (reverse-sorted placement: worst-first, best-last for recency bias)
    context_chunks:     list[RerankResult]
 
    # All documents scored, including those gated out, for analysis
    all_scored:         list[RerankResult]
 
    # Aggregate signals for Layer 6
    max_reranker_score:    float    # highest normalized reranker score
    min_reranker_score:    float    # lowest admitted score
    mean_reranker_score:   float    # mean of admitted chunks
    median_reranker_score: float    # median of admitted_results
    n_admitted:            int      # how many passed the gate
    n_scored:              int      # how many were scored total
    option_c_triggered:    bool     # True if zero docs passed gate, best-of-one fallback
    query_too_long:        bool     # True if query exceeded _MAX_QUERY_TOKENS, no scoring done
    total_context_tokens:  int      # token count of assembled context
    latency_ms:            float
 
    query:              str
    domain_filter:      str | None  # passed through from Layer 3 for logging


# CROSS-ENCODER RERANKER
 
class CrossEncoderReranker:
    """
    Cross-encoder reranker.
    """
 
    # Safe ceiling for model sequence length.
    _SAFE_MAX_LENGTH = 512

    # Max query tokens before we skip reranking entirely.
    # A 150-token query leaves at least 308 tokens for the doc (with
    # 4 special tokens). Anything longer isn't a genuine wiki question
    # it's either a prompt injection or a paste error. The pipeline
    # flags query_too_long and passes nothing to the next layer.
    _MAX_QUERY_TOKENS = 150

    # Models with verified larger context windows override the safe ceiling.
    _MODEL_MAX_LENGTHS = {
        "jina-reranker-v2-base": 8192,
    }
 
    MODELS = {
        "ms-marco-MiniLM-L6":      "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-TinyBERT-L2":    "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "ms-marco-MiniLM-L12":     "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "mxbai-rerank-base-v1":    "mixedbread-ai/mxbai-rerank-base-v1",
        "jina-reranker-v2-base":   "jinaai/jina-reranker-v2-base-multilingual",
        "bge-reranker-base":       "BAAI/bge-reranker-base",
        "bge-reranker-v2-m3":      "BAAI/bge-reranker-v2-m3",
        "bge-reranker-large":      "BAAI/bge-reranker-large",
        "mxbai-rerank-large-v1":   "mixedbread-ai/mxbai-rerank-large-v1",
    }
 
    def __init__(
        self,
        model_key:      str = "bge-reranker-large",
        device:         str = "cpu",
        temperature:    float = 4.0,
        clean_prefix:   bool = True,
    ):

        model_name = self.MODELS[model_key]
        print(f"Loading cross-encoder reranker: {model_name} on {device}")
 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
 
        self.device = device
        self.model_key = model_key
        self.model_name = model_name
        self.temperature = temperature
        self.clean_prefix = clean_prefix
 
        # Detect max sequence length from model config
        reported = getattr(self.tokenizer, "model_max_length", self._SAFE_MAX_LENGTH)
        model_override = self._MODEL_MAX_LENGTHS.get(model_key)
        if model_override:
            self._max_length = model_override
        elif reported > self._SAFE_MAX_LENGTH:
            self._max_length = self._SAFE_MAX_LENGTH
        else:
            self._max_length = reported

        # Detect how many special tokens the tokenizer adds for pairs.

        _a = self.tokenizer("a", add_special_tokens=False)["input_ids"]
        _b = self.tokenizer("b", add_special_tokens=False)["input_ids"]
        _pair = self.tokenizer("a", "b")["input_ids"]
        self._n_special = len(_pair) - len(_a) - len(_b)

        print(f"Reranker ready. (max_length: {self._max_length}, "
              f"special_tokens: {self._n_special}, "
              f"temperature: {self.temperature}, clean_prefix: {self.clean_prefix})")
 
    def to_device(self, device: str) -> "CrossEncoderReranker":

        if device != self.device:
            self.model.to(device)
            self.device = device
        return self
 
    def _normalize_score(self, raw_score: float) -> float:
        return 1.0 / (1.0 + math.exp(-raw_score / self.temperature))
 
    @staticmethod
    def _clean_for_reranker(text: str) -> str:
        """
        Prepare chunk text for cross-encoder scoring.
 
        Two transformations:
 
        1. Convert the structured prefix ("Game: Dota 2 | Hero: Phantom
           Assassin | Section: Abilities | Data: ...") into a compact
           comma-separated preamble: "Dota 2, Phantom Assassin, Abilities."
 
        2. Replace pipe separators (" | ") with periods (". ") in the
           remaining content.
        """
        # Step 1: convert structured prefix to compact preamble
        marker = "| Data: "
        idx = text.find(marker)
        if idx != -1:
            prefix_part = text[:idx].strip()
            content_part = text[idx + len(marker):]
 
            preamble_parts = []
            for segment in prefix_part.split(" | "):
                segment = segment.strip()
                if ": " in segment:
                    preamble_parts.append(segment.split(": ", 1)[1].strip())
                elif segment:
                    preamble_parts.append(segment.strip())
 
            preamble = ", ".join(preamble_parts) + ". " if preamble_parts else ""
            text = preamble + content_part
        else:
            marker_alt = "Data: "
            if text.startswith(marker_alt):
                text = text[len(marker_alt):]
 
        # Step 2: replace remaining pipe separators with sentence boundaries
        text = re.sub(r"\s*\|\s*", ". ", text)
        text = text.replace(".. ", ". ").replace("..", ".")
 
        return text.strip()
 
    def _doc_to_windows(
        self,
        query:          str,
        doc_text:       str,
        overlap_tokens: int = 50,
    ) -> list[str]:
        """
        Split a document into overlapping windows that each fit within
        the model's max_length when paired with the query.

        Goal: never lose chunk content just because query + chunk is
        too fat for the reranker's context window. Every token in the
        chunk gets scored in at least one window.

        Budget:
            doc_budget = max_length - query_tokens - n_special - overlap_tokens

            n_special is auto-detected in __init__ (4 for XLM-RoBERTa,
            3 for BERT/DeBERTa). The overlap_tokens subtraction acts
            as a safety margin that absorbs decode→re-tokenize drift
            (subword boundaries shift when token IDs are decoded to
            text and re-tokenized as a pair in score_pairs).

        Window construction:
            Window 1: doc_tokens[0 : doc_budget]
            Window 2: doc_tokens[doc_budget - overlap : doc_budget*2 - overlap]
            ...each decoded back to text for score_pairs.

            score_pairs takes the MAX score across all windows for each
            document — a chunk is relevant if ANY portion scores high.
        """
        query_ids = self.tokenizer(query, add_special_tokens=False)["input_ids"]

        # Query Too Long Guard
        # If the query exceeds the threshold, there's not enough room
        # for meaningful chunk content. Return empty — score_pairs
        # assigns -inf for this document, and RerankerPipeline flags
        # query_too_long at the pipeline level.
        if len(query_ids) > self._MAX_QUERY_TOKENS:
            return []

        doc_budget = self._max_length - len(query_ids) - self._n_special - overlap_tokens

        # If budget is too small for meaningful scoring (shouldn't
        # happen with the query guard, but defensive)
        if doc_budget <= 0:
            return []

        doc_ids = self.tokenizer(
            doc_text, add_special_tokens=False, verbose=False
        )["input_ids"]

        # Chunk fits in one window — no splitting needed
        if len(doc_ids) <= doc_budget:
            return [doc_text]

        # Split into overlapping windows
        # Guard: ensure stride > 0 (if doc_budget <= overlap, reduce
        # overlap dynamically to half the budget)
        effective_overlap = min(overlap_tokens, doc_budget // 2)
        stride = doc_budget - effective_overlap

        windows = []
        start = 0

        while start < len(doc_ids):
            end = min(start + doc_budget, len(doc_ids))
            window_ids = doc_ids[start:end]
            window_text = self.tokenizer.decode(window_ids, skip_special_tokens=True)
            windows.append(window_text)

            if end >= len(doc_ids):
                break
            start += stride

        return windows
 
    def score_pairs(
        self,
        query:      str,
        documents:  list[str],
        batch_size: int = 16,
    ) -> list[tuple[float, float]]:
        """
        Score each (query, document) pair and return (raw, normalized) scores.
 
        Returns:
            List of (raw_score, normalized_score) tuples, one per document.
        """
        if self.clean_prefix:
            documents = [self._clean_for_reranker(d) for d in documents]
 
        all_window_texts = []
        window_to_doc    = []
 
        for doc_idx, doc_text in enumerate(documents):
            windows = self._doc_to_windows(query, doc_text)
            for window_text in windows:
                all_window_texts.append(window_text)
                window_to_doc.append(doc_idx)
 
        all_raw_scores = []
 
        for i in range(0, len(all_window_texts), batch_size):
            batch_windows = all_window_texts[i : i + batch_size]

            inputs = self.tokenizer(
                [query] * len(batch_windows),
                batch_windows,
                return_tensors="pt",
                truncation="only_second",
                max_length=self._max_length,
                padding=True,
                verbose=False,
            ).to(self.device)
 
            with torch.no_grad():
                logits = self.model(**inputs).logits
                scores = logits.squeeze(-1).tolist()
 
            if isinstance(scores, float):
                scores = [scores]
 
            all_raw_scores.extend(scores)
 
        doc_raw_scores = [float('-inf')] * len(documents)
        for window_idx, raw_score in enumerate(all_raw_scores):
            doc_idx = window_to_doc[window_idx]
            if raw_score > doc_raw_scores[doc_idx]:
                doc_raw_scores[doc_idx] = raw_score
 
        results = []
        for raw in doc_raw_scores:
            results.append((raw, self._normalize_score(raw)))
 
        return results

# THRESHOLD GATING
 
def apply_threshold_gate(
    scored_results: list[RerankResult],
    threshold:      float,
) -> tuple[list[RerankResult], bool]:
    """
    Apply similarity threshold gating to scored results.
 
    Only documents with reranker_score >= threshold are admitted to
    the LLM context. This replaces fixed top-k selection entirely —
    the number of context documents is dynamic and quality-driven.
 
    Option-C fallback:
        If zero documents clear the threshold, the best available
        document is passed regardless. Its low reranker score will
        propagate through confidence orchestration, naturally
        routing to the low-confidence output tier without special-case
        logic.
    """
    admitted = [r for r in scored_results if r.reranker_score >= threshold]
 
    if not admitted and scored_results:
        # Option-C: pass best available document despite below-threshold score
        best = scored_results[0]   # already sorted descending
        best.passed_gate = True    # mark as admitted (via fallback)
        admitted = [best]
        return admitted, True      # option_c_triggered = True
 
    # Mark admitted docs
    for r in admitted:
        r.passed_gate = True
 
    return admitted, False

# CONTEXT POSITIONING
 
def recency_biased_placement(
    documents: list[RerankResult],
) -> list[RerankResult]:
    """
    Order documents worst-first, best-last to exploit recency bias.
        Position 1 (start of context): lowest-scoring admitted chunk
        Position k (end of context):   highest-scoring admitted chunk
    """
    # Reverse: worst first, best last (ascending score order)
    placed = list(reversed(documents))
 
    for i, doc in enumerate(placed):
        doc.context_position = i + 1
 
    return placed

def primacy_recency_placement(
    documents: list[RerankResult],
) -> list[RerankResult]:
    """
    Place chunks alternately at the END and START of the context,
    walking through ranks in descending-score order. Each rank lands
    one slot inward from the previous insertion on its side.

    The walk:
        rank 1 → END                          (recency anchor)
        rank 2 → START                        (primacy anchor)
        rank 3 → END side, one step inward    (just before rank 1)
        rank 4 → START side, one step inward  (just after rank 2)
        rank 5 → END side, two steps inward
        rank 6 → START side, two steps inward
        and so on, alternating ends until all chunks are placed.

    Same chunks, same content, same token budget. Only the order
    differs vs recency_biased_placement. Zero latency overhead.
    """
    if not documents:
        return []

    n = len(documents)
    placed: list[RerankResult] = [None] * n

    end_ptr   = n - 1
    start_ptr = 0

    for rank_idx, doc in enumerate(documents):
        if rank_idx % 2 == 0:
            placed[end_ptr] = doc
            end_ptr -= 1
        else:
            placed[start_ptr] = doc
            start_ptr += 1

    for i, doc in enumerate(placed):
        doc.context_position = i + 1

    return placed

# TOKEN BUDGET ENFORCEMENT
 
def enforce_token_budget(
    documents:    list[RerankResult],
    max_tokens:   int = 3500,
) -> list[RerankResult]:
    """
    Trim the document list to fit within the token budget.
 
    Applied AFTER threshold gating but BEFORE placement, so placement
    operates on the final document set and doesn't waste positions on
    documents that will be trimmed.
    """
    cumulative = 0
    admitted = []
 
    for doc in documents:
        doc_tokens = doc.chunk.token_count or 0
 
        if cumulative + doc_tokens > max_tokens and admitted:
            break
        cumulative += doc_tokens
        admitted.append(doc)
 
    return admitted

# RERANKER PIPELINE
 
class RerankerPipeline:
 
    def __init__(
        self,
        model_key:          str = "bge-reranker-large",
        device:             str = "cpu",
        gate_threshold:     float = 0.45,
        temperature:        float = 4.0,
        clean_prefix:       bool = True,
        max_context_tokens: int = 3500,
        batch_size:         int = 16,
    ):
        """
        Args:
            model_key:          Key from CrossEncoderReranker.MODELS dict.
            device:             Initial device. Model is loaded here; use
                                to_device() to shuttle for inference.
            gate_threshold:     Minimum normalized reranker score [0, 1].
            temperature:        Sigmoid normalization temperature.
            clean_prefix:       If True, convert structured chunk prefixes
                                to compact preambles before scoring.
            max_context_tokens: Token budget for assembled context (3,500).
            batch_size:         Pairs per forward pass.
        """
        self.reranker = CrossEncoderReranker(
            model_key=model_key, device=device,
            temperature=temperature, clean_prefix=clean_prefix,
        )
        self.gate_threshold = gate_threshold
        self.max_context_tokens = max_context_tokens
        self.batch_size = batch_size
 
    def to_device(self, device: str) -> "RerankerPipeline":
        """
        Move the cross-encoder model to a different device.
        """
        self.reranker.to_device(device)
        return self
 
    def rerank(
        self,
        query:            str,
        retrieval_output: RetrievalOutput,
    ) -> RerankerOutput:

        t0 = time.perf_counter()
 
        candidates = retrieval_output.candidates
 
        if not candidates:
            return RerankerOutput(
                context_chunks=[],
                all_scored=[],
                max_reranker_score=0.0,
                min_reranker_score=0.0,
                mean_reranker_score=0.0,
                median_reranker_score=0.0,
                n_admitted=0,
                n_scored=0,
                option_c_triggered=False,
                query_too_long=False,
                total_context_tokens=0,
                latency_ms=0.0,
                query=query,
                domain_filter=retrieval_output.domain_filter,
            )

        # Guard: query too long
        # If the query exceeds the max token threshold, the reranker
        # can't produce meaningful scores (too little room for chunk
        # content). Return empty with the flag set so the pipeline
        # can handle it (e.g., skip to Layer 5 with a warning, or
        # return a LOW confidence tier in Layer 6).
        query_ids = self.reranker.tokenizer(
            query, add_special_tokens=False
        )["input_ids"]
        if len(query_ids) > self.reranker._MAX_QUERY_TOKENS:
            latency_ms = (time.perf_counter() - t0) * 1000
            print(f"Query too long ({len(query_ids)} tokens > "
                  f"{self.reranker._MAX_QUERY_TOKENS} max). Skipping reranking.")
            return RerankerOutput(
                context_chunks=[],
                all_scored=[],
                max_reranker_score=0.0,
                min_reranker_score=0.0,
                mean_reranker_score=0.0,
                median_reranker_score=0.0,
                n_admitted=0,
                n_scored=0,
                option_c_triggered=False,
                query_too_long=True,
                total_context_tokens=0,
                latency_ms=round(latency_ms, 1),
                query=query,
                domain_filter=retrieval_output.domain_filter,
            )
 
        # Step 1: Score all candidates
        doc_texts = [c.page_content for c in candidates]
        score_pairs = self.reranker.score_pairs(
            query, doc_texts, batch_size=self.batch_size
        )
 
        scored_results = []
        for chunk, (raw, norm) in zip(candidates, score_pairs):
            scored_results.append(RerankResult(
                chunk=chunk,
                reranker_score=round(norm, 4),
                raw_reranker_score=round(raw, 4),
                passed_gate=False,
                context_position=None,
            ))
 
        # Step 2: Sort by reranker score descending
        scored_results.sort(key=lambda r: r.reranker_score, reverse=True)
 
        # Step 3: Threshold gating
        admitted, option_c = apply_threshold_gate(
            scored_results, self.gate_threshold
        )
 
        # Step 4: Token budget enforcement
        budgeted = enforce_token_budget(admitted, self.max_context_tokens)
 
        # Step 5: Chunk placement
        final = recency_biased_placement(budgeted)
        #final = primacy_recency_placement(budgeted)
 
        latency_ms = (time.perf_counter() - t0) * 1000
 
        admitted_scores = [r.reranker_score for r in final]
        max_score    = max(admitted_scores)  if admitted_scores else 0.0
        min_score    = min(admitted_scores)  if admitted_scores else 0.0
        mean_score   = (sum(admitted_scores) / len(admitted_scores)
                        if admitted_scores else 0.0)
        median_score = float(np.median(admitted_scores)) if admitted_scores else 0.0
        total_tokens = sum(r.chunk.token_count or 0 for r in final)
 
        return RerankerOutput(
            context_chunks=final,
            all_scored=scored_results,
            max_reranker_score=round(max_score, 4),
            min_reranker_score=round(min_score, 4),
            mean_reranker_score=round(mean_score, 4),
            median_reranker_score=round(median_score, 4),
            n_admitted=len(final),
            n_scored=len(scored_results),
            option_c_triggered=option_c,
            query_too_long=False,
            total_context_tokens=total_tokens,
            latency_ms=round(latency_ms, 1),
            query=query,
            domain_filter=retrieval_output.domain_filter,
        )

# INSPECTION & DEBUGGING UTILITIES
 
def print_reranker_results(output: RerankerOutput, max_show: int = 10):
    print("=" * 85)
    print(f"RERANKER RESULTS — {output.query[:65]}")
    print("=" * 85)

    if output.query_too_long:
        print(f"  QUERY TOO LONG — reranking skipped")
        print(f"  Latency : {output.latency_ms:.1f}ms")
        print("=" * 85)
        return

    print(f"  Scored          : {output.n_scored} candidates")
    print(f"  Admitted (gated): {output.n_admitted} "
          f"{'(Option-C fallback!)' if output.option_c_triggered else ''}")
    print(f"  Max score       : {output.max_reranker_score:.4f}")
    print(f"  Min score       : {output.min_reranker_score:.4f}")
    print(f"  Mean score      : {output.mean_reranker_score:.4f}")
    print(f"  Median score    : {output.median_reranker_score:.4f}")
    print(f"  Context tokens  : {output.total_context_tokens} / 3500")
    print(f"  Latency         : {output.latency_ms:.1f}ms")
    print(f"  Domain filter   : {output.domain_filter or 'None (hybrid)'}")
    print()
    for r in output.context_chunks[:max_show]:
        domain = r.chunk.domain
        print(f"    pos={r.context_position:<3} norm={r.reranker_score:>8.4f}  "
              f"raw={r.raw_reranker_score:>8.4f}  "
              f"domain={domain:<6}  tokens={r.chunk.token_count}")
        print(f"         {r.chunk.page_content[:100]}...")
        print()
 
    # Show score distribution of ALL scored documents
    if output.all_scored:
        all_norm = [r.reranker_score for r in output.all_scored]
        all_raw  = [r.raw_reranker_score for r in output.all_scored]
        print(f"  SCORE DISTRIBUTION (all {output.n_scored} candidates):")
        print(f"    normalized — max={max(all_norm):.4f}  min={min(all_norm):.4f}  "
              f"median={sorted(all_norm)[len(all_norm)//2]:.4f}")
        print(f"    raw logits — max={max(all_raw):.4f}  min={min(all_raw):.4f}  "
              f"median={sorted(all_raw)[len(all_raw)//2]:.4f}")
 
        # Domain breakdown
        domain_counts = {}
        for r in output.all_scored:
            d = r.chunk.domain
            domain_counts[d] = domain_counts.get(d, 0) + 1
        print(f"    domains: {domain_counts}")
 
        # Gated-out documents
        gated_out = [r for r in output.all_scored if not r.passed_gate]
        if gated_out:
            print(f"\n  GATED OUT ({len(gated_out)} documents below threshold):")
            for r in gated_out[:5]:
                print(f"    norm={r.reranker_score:>8.4f}  raw={r.raw_reranker_score:>8.4f}  "
                      f"domain={r.chunk.domain:<6}  "
                      f"{r.chunk.page_content[:80]}...")
            if len(gated_out) > 5:
                print(f"    ... and {len(gated_out) - 5} more")
 
    print("=" * 85)
 
 
def get_context_for_llm(output: RerankerOutput) -> str:
    """
    Assemble the final context string for the LLM prompt.
 
    Returns the concatenated page_content of all admitted chunks
    in context order (reverse-sorted: worst-first, best-last),
    separated by double newlines.
    """
    parts = []
    for r in output.context_chunks:
        parts.append(r.chunk.page_content)
    return "\n\n".join(parts)
 
 
def get_context_metadata(output: RerankerOutput) -> list[dict]:
    """
    Extract metadata for each context chunk — used for source
    citations in the final answer and Layer 6 logging.
    """
    meta_list = []
    for r in output.context_chunks:
        meta_list.append({
            "chunk_id":           r.chunk.chunk_id,
            "domain":             r.chunk.domain,
            "reranker_score":     r.reranker_score,
            "raw_reranker_score": r.raw_reranker_score,
            "context_position":   r.context_position,
            "metadata":           r.chunk.metadata,
        })
    return meta_list