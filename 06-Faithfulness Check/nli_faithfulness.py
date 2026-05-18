
# IMPORTS

import gc
import re
import time
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from gpu_utils import free_vram, gpu_available

# DATA STRUCTURES


@dataclass
class FaithfulnessOutput:
    """
    Per-claim and aggregate faithfulness signals from NLI scoring.
    """
    mean_faithfulness:  float
    max_faithfulness:   float
    faithful_claims:    int
    total_claims:       int
    per_claim_scores:   list     # list of dicts per claim
    latency_ms:         float



# NLI FAITHFULNESS CHECKER

class NLIFaithfulnessChecker:
    """
    NLI-based faithfulness scorer 

    For each generated claim, scores it against every retrieved chunk
    as a premise-hypothesis pair (premise = chunk, hypothesis = claim).
    A claim is faithful if any chunk entails it above threshold.
    """

    MODELS = {
        "deberta-base":       "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        "deberta-large":      "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        "facebook-bart-mnli": "facebook/bart-large-mnli",
        "deberta-v2-xlarge":  "microsoft/deberta-v2-xlarge-mnli",
        "deberta-v2-xxlarge": "microsoft/deberta-v2-xxlarge-mnli",
    }

    # Safe max length ceiling
    _SAFE_MAX_LENGTH = 512

    def __init__(self, model_key: str = "deberta-large", device: str = "cpu"):
        model_name = self.MODELS[model_key]
        print(f"Loading NLI faithfulness checker: {model_name} on {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.model_key = model_key
        self.model_name = model_name

        # Detect max sequence length from tokenizer config, capped for safety
        reported = getattr(self.tokenizer, "model_max_length", self._SAFE_MAX_LENGTH)
        if reported > 100000:
            self._max_length = self._SAFE_MAX_LENGTH
        else:
            self._max_length = reported

        # Detect how many special tokens the tokenizer adds for pairs.
        #   DeBERTa:     [CLS] A [SEP] B [SEP]  → 3 special tokens
        #   XLM-RoBERTa: <s> A </s></s> B </s>   → 4 special tokens
        _a = self.tokenizer("a", add_special_tokens=False)["input_ids"]
        _b = self.tokenizer("b", add_special_tokens=False)["input_ids"]
        _pair = self.tokenizer("a", "b")["input_ids"]
        self._n_special = len(_pair) - len(_a) - len(_b)

        # Detect the label ordering from the model config.
        # Most NLI models use: {0: "contradiction", 1: "neutral", 2: "entailment"}
        # but some reverse it. We resolve at init so check_faithfulness is fast.
        id2label = self.model.config.id2label
        self._entailment_idx = None
        for idx, label in id2label.items():
            if "entail" in label.lower():
                self._entailment_idx = int(idx)
                break

        if self._entailment_idx is None:
            raise ValueError(
                f"Cannot find entailment class in model labels: {id2label}. "
                "Expected a label containing 'entail'."
            )

        print(f"Faithfulness checker ready. Entailment index: {self._entailment_idx} "
              f"(labels: {id2label}, max_length: {self._max_length}, "
              f"special_tokens: {self._n_special})")

    def to_device(self, device: str) -> "NLIFaithfulnessChecker":
        if device != self.device:
            self.model.to(device)
            self.device = device
        return self

    def _score_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Score a single (premise, hypothesis) pair and return the raw
        entailment probability after ONE softmax over NLI classes.

        Truncation strategy: "only_first" truncates the premise (chunk)
        while keeping the hypothesis (claim) fully intact. Chunks can
        be ~500 tokens and claims add another 20-50, exceeding the 512
        token limit for some pairs.
        """
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation="only_first",   # NEVER truncate hypothesis
            max_length=self._max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits          # shape: [1, 3]
            probs = torch.softmax(logits, dim=-1)         # single softmax over NLI classes
            entailment_prob = probs[0, self._entailment_idx].item()

        return entailment_prob

    def _score_entailment_batch(
        self,
        premises:   list[str],
        hypotheses: list[str],
        batch_size: int = 32,
    ) -> list[float]:
        """
        Score multiple (premise, hypothesis) pairs in batches.
        Returns list of raw entailment probabilities.

        Uses truncation="only_first" to protect hypotheses from truncation
        when premises (chunks) are long.
        """
        all_scores = []

        for i in range(0, len(premises), batch_size):
            batch_premises   = premises[i : i + batch_size]
            batch_hypotheses = hypotheses[i : i + batch_size]

            inputs = self.tokenizer(
                batch_premises, batch_hypotheses,
                return_tensors="pt",
                truncation="only_first",   # NEVER truncate hypotheses
                max_length=self._max_length,
                padding=True,
                verbose=False,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits          # shape: [batch, 3]
                probs = torch.softmax(logits, dim=-1)         # softmax per row
                entailment_probs = probs[:, self._entailment_idx].tolist()

            all_scores.extend(entailment_probs)

        return all_scores

    def _chunk_to_windows(
        self,
        chunk_text:     str,
        hypothesis:     str,
        overlap_tokens: int = 50,
    ) -> list[str]:
        """
        Split a long chunk into overlapping text windows that each fit
        within the model's max_length when paired with the hypothesis.

        Used by check_faithfulness for chunks that exceed the context
        window. Each window is scored independently against the claim;
        the max score across windows is taken.
        """
        hyp_tokens = self.tokenizer(hypothesis, add_special_tokens=False)["input_ids"]
        premise_budget = self._max_length - len(hyp_tokens) - self._n_special - overlap_tokens

        if premise_budget <= 0:
            # Hypothesis alone exceeds context
            return [chunk_text]

        chunk_tokens = self.tokenizer(
            chunk_text, add_special_tokens=False, verbose=False
        )["input_ids"]

        if len(chunk_tokens) <= premise_budget:
            return [chunk_text]

        # Guard: ensure stride > 0
        effective_overlap = min(overlap_tokens, premise_budget // 2)
        stride = premise_budget - effective_overlap

        windows = []
        start = 0

        while start < len(chunk_tokens):
            end = min(start + premise_budget, len(chunk_tokens))
            window_ids = chunk_tokens[start:end]
            window_text = self.tokenizer.decode(window_ids, skip_special_tokens=True)
            windows.append(window_text)

            if end >= len(chunk_tokens):
                break
            start += stride

        return windows

    # Faithfulness Cross-Checking

    def check_faithfulness(
        self,
        claims:               list[str],
        context_chunks:       list[str],
        entailment_threshold: float = 0.50,
        batch_size:           int = 32,
    ) -> FaithfulnessOutput:
        """
        Check if generated answer claims are entailed by retrieved context.

        For each claim, scores it against every retrieved chunk as a
        premise-hypothesis pair: premise = chunk, hypothesis = claim.
        A claim is faithful if ANY chunk entails it above threshold.

        Long chunk handling:
            Chunks exceeding the model's context window (512 tokens for
            DeBERTa) are split into overlapping windows. Each window is
            scored independently; the max score across windows represents
            that chunk's entailment.
        """
        if not claims or not context_chunks:
            return FaithfulnessOutput(
                mean_faithfulness=0.0,
                max_faithfulness=0.0,
                faithful_claims=0,
                total_claims=len(claims),
                per_claim_scores=[],
                latency_ms=0.0,
            )

        t0 = time.perf_counter()

        # Build all (premise, hypothesis) pairs
        # premise = chunk text (or chunk window),  hypothesis = generated claim
        all_premises   = []
        all_hypotheses = []
        pair_map       = []   # (claim_idx, chunk_idx) for each pair

        for ci, claim in enumerate(claims):
            for ki, chunk in enumerate(context_chunks):
                windows = self._chunk_to_windows(chunk, claim)
                for window in windows:
                    all_premises.append(window)
                    all_hypotheses.append(claim)
                    pair_map.append((ci, ki))

        # Batched scoring
        all_scores = self._score_entailment_batch(
            all_premises, all_hypotheses, batch_size=batch_size
        )

        # Reshape: find best chunk per claim
        # For each claim, the faithfulness score is the MAX entailment
        # across all chunks (and all windows within each chunk).
        claim_best = {}   # claim_idx -> (best_score, best_chunk_idx)
        for (ci, ki), score in zip(pair_map, all_scores):
            if ci not in claim_best or score > claim_best[ci][0]:
                claim_best[ci] = (score, ki)

        per_claim = []
        for ci, claim in enumerate(claims):
            best_score, best_chunk = claim_best.get(ci, (0.0, -1))
            per_claim.append({
                "claim":            claim,
                "best_entailment":  round(best_score, 4),
                "faithful":         best_score >= entailment_threshold,
                "supporting_chunk": best_chunk,
            })

        faithful_count = sum(1 for c in per_claim if c["faithful"])
        mean_faith     = sum(c["best_entailment"] for c in per_claim) / len(per_claim)
        max_faith      = max(c["best_entailment"] for c in per_claim)
        latency_ms     = (time.perf_counter() - t0) * 1000

        return FaithfulnessOutput(
            mean_faithfulness=round(mean_faith, 4),
            max_faithfulness=round(max_faith, 4),
            faithful_claims=faithful_count,
            total_claims=len(claims),
            per_claim_scores=per_claim,
            latency_ms=round(latency_ms, 1),
        )



# CLAIM SPLITTER (for faithfulness checking)

def split_into_claims(answer: str, max_claims: int = 8) -> list[str]:
    """
    Split a generated answer into atomic claims for faithfulness checking.

    Simple sentence-boundary splitting. Cap at max_claims to control
    faithfulness latency (~5ms per claim*chunk pair in batched mode).
    """
    raw = re.split(r"(?<=[.!?])\s+", answer.strip())
    claims = [s.strip() for s in raw if len(s.strip().split()) >= 3]
    return claims[:max_claims]