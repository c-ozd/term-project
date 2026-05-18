# IMPORTS
from dataclasses import dataclass
from enum import Enum
import numpy as np


# TIER DEFINITIONS

class ResponseTier(Enum):
    HIGH     = "high"       # >  0.70 — confident, grounded answer
    MODERATE = "moderate"   # >  0.50 — answer with uncertainty note
    LOW      = "low"        # >= 0.20 — answer with low-confidence warning
    DEAD     = "dead"       # <  0.20 — abstain, ask user to rephrase


# CONFIGURATION

@dataclass
class OrchestratorConfig:
    # Signal weights (must sum to 1.0)
    # Defaults per spec: 0.7 reranker, 0.3 faithfulness.
    weight_reranker:     float = 0.70
    weight_faithfulness: float = 0.30

    # Tier thresholds
    # HIGH and MODERATE are strict ( > ); LOW is inclusive ( >= ).
    threshold_high:      float = 0.70
    threshold_moderate:  float = 0.50
    threshold_low:       float = 0.20
    # Below threshold_low → DEAD tier

    def __post_init__(self):
        total = self.weight_reranker + self.weight_faithfulness
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Signal weights must sum to 1.0, got {total:.3f}. "
                f"Current: reranker={self.weight_reranker}, "
                f"faithfulness={self.weight_faithfulness}"
            )


# ORCHESTRATOR OUTPUT

@dataclass
class OrchestratorOutput:
    """
    This is the FINAL output of the pipeline.
    """
    # Tier assignment
    tier:                ResponseTier
    aggregate_score:     float         # weighted sum of 2 signals [0, 1]

    # Individual signal values (for diagnostics)
    reranker_signal:     float         # median reranker score [0, 1]
    faithfulness_signal: float         # max claim entailment score [0, 1]

    # Weighted contributions (signal * weight)
    reranker_contribution:     float
    faithfulness_contribution: float

    # Formatted response
    response:            str           # final user-facing response text
    answer:              str           # raw answer from generation
    context_metadata:    list[dict]    # source citation data
    query:               str           # original query


# SIGNAL EXTRACTION

def extract_signals(reranker_output, faithfulness_output) -> dict:
    """
    Extract the two confidence signals.

    Reranker signal:
        Median across the admitted (gate-passing) reranker scores.
        Robust to single-outlier chunks that dominate a mean.

    Faithfulness signal:
        Max claim entailment from the NLI faithfulness check.
        "Did at least one claim find strong support in context?"
    """
    # Reranker signal
    reranker_scores = [
        r.reranker_score
        for r in reranker_output.context_chunks
    ]
    reranker_signal = float(np.median(reranker_scores)) if reranker_scores else 0.0

    # Faithfulness signal
    faithfulness_max = faithfulness_output.max_faithfulness

    return {
        "reranker_median":  reranker_signal,
        "faithfulness_max": faithfulness_max,
    }


# SCORING & TIER ASSIGNMENT

def compute_aggregate_score(
    signals: dict,
    config:  OrchestratorConfig = None,
) -> tuple[float, dict]:
    """
    Compute weighted aggregate confidence score.

    """
    if config is None:
        config = OrchestratorConfig()

    reranker_contrib = signals["reranker_median"]  * config.weight_reranker
    faith_contrib    = signals["faithfulness_max"] * config.weight_faithfulness

    aggregate = reranker_contrib + faith_contrib

    contributions = {
        "reranker":     round(reranker_contrib, 4),
        "faithfulness": round(faith_contrib, 4),
    }

    return round(aggregate, 4), contributions


def assign_tier(
    aggregate_score: float,
    config:          OrchestratorConfig = None,
) -> ResponseTier:
    """
    Map aggregate score to response tier.

    Comparison rules per spec:
        > 0.65   -> HIGH
        > 0.50   -> MODERATE
        >= 0.20  -> LOW 
        otherwise -> DEAD
    """
    if config is None:
        config = OrchestratorConfig()

    if aggregate_score > config.threshold_high:
        return ResponseTier.HIGH
    elif aggregate_score > config.threshold_moderate:
        return ResponseTier.MODERATE
    elif aggregate_score >= config.threshold_low:
        return ResponseTier.LOW
    else:
        return ResponseTier.DEAD


# RESPONSE FORMATTING


def format_citations(context_metadata: list[dict]) -> str:
    """
    Format context metadata into citation.
    """
    if not context_metadata:
        return ""

    lines = ["\nSources:"]
    for m in context_metadata:
        chunk_id = m.get("chunk_id", "unknown")
        domain   = m.get("domain", "unknown")
        score    = m.get("reranker_score", 0.0)
        lines.append(f"  [{domain}] {chunk_id} (relevance: {score:.2f})")

    return "\n".join(lines)


def format_response(
    tier:             ResponseTier,
    answer:           str,
    context_metadata: list[dict],
    aggregate_score:  float,
) -> str:
    citations = format_citations(context_metadata)

    if tier == ResponseTier.DEAD:
        response = (
            "I wasn't able to find a confident answer to your question. "
            "This could mean the question is too vague, the relevant information "
            "isn't in my knowledge base, or the retrieved context didn't match well enough.\n\n"
            "Could you try rephrasing your question with more specific details? "
            "For example, mentioning the game name (Dota 2 / League of Legends), "
            "specific hero/champion names, or the mechanic you're asking about."
        )
    elif tier == ResponseTier.LOW:
        response = (
            f"{answer}\n"
            f"{citations}\n\n"
            f"Low Confidence (score: {aggregate_score:.2f}): "
            f"This answer may be incomplete or partially inaccurate. "
            f"The retrieved context had limited relevance to your question. "
            f"Consider verifying this information on the official wiki."
        )
    elif tier == ResponseTier.MODERATE:
        response = (
            f"{answer}\n"
            f"{citations}\n\n"
            f"Note: While this answer is based on retrieved context, "
            f"there is some uncertainty in the match quality. "
            f"If precision matters, double-check the cited sources."
        )
    else:  # HIGH
        response = (
            f"{answer}\n"
            f"{citations}"
        )

    return response


# ORCHESTRATOR PIPELINE

class OrchestratorPipeline:
    def __init__(self, config: OrchestratorConfig = None):
        self.config = config or OrchestratorConfig()

    def run(
        self,
        reranker_output,
        answer:                str,
        generation:            GenerationOutput,
        faithfulness:          FaithfulnessOutput,
        context_metadata:      list[dict],
        total_latency_ms:      float,
        query:                 str,
        injection_used:        bool = False,
        injected_chunk_score:  float | None = None,
        injected_chunk_id:     str | None = None,
    ) -> OrchestratorOutput:
        # Step 1: Extract signals
        signals = extract_signals(reranker_output, faithfulness)

        # Step 2: Compute aggregate score
        aggregate_score, contributions = compute_aggregate_score(
            signals, self.config
        )

        # Step 3: Assign tier
        tier = assign_tier(aggregate_score, self.config)

        # Step 4: Format response
        response = format_response(
            tier=tier,
            answer=answer,
            context_metadata=context_metadata,
            aggregate_score=aggregate_score,
        )

        # Step 5: Package output
        return OrchestratorOutput(
            tier=tier,
            aggregate_score=aggregate_score,
            reranker_signal=signals["reranker_median"],
            faithfulness_signal=signals["faithfulness_max"],
            reranker_contribution=contributions["reranker"],
            faithfulness_contribution=contributions["faithfulness"],
            response=response,
            answer=answer,
            context_metadata=context_metadata,
            query=query,
        )