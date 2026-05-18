# IMPORTS

import gc
import time
import math
import numpy as np
from dataclasses import dataclass, field


# DATA STRUCTURES


@dataclass
class TokenLogProb:
    """Single token with its log-probability from generation."""
    token:   str
    logprob: float        # raw log-probability (negative)
    prob:    float        # exp(logprob) — probability in [0, 1]


@dataclass
class GenerationOutput:
    """
    Raw output from LLM generation.

    Extended from original to include per-token logprobs for
    parametric override detection. token_logprobs is populated
    on every generation call (zero extra cost) but only consumed
    by Layer 6 when faithfulness is low.
    """
    answer:          str
    latency_ms:      float
    model:           str
    token_logprobs:  list[TokenLogProb] = field(default_factory=list)


@dataclass
class OverrideDetectionResult:
    """
    Result of parametric override detection (Layer 6 binary flag).

    This is NOT a scoring signal — it lives outside the weighted sum.
    It fires only when: factual_token_confidence > 0.75 AND faithfulness < 0.40.

    When triggered, the response gets a parametric override warning
    regardless of aggregate tier score.
    """
    is_override:              bool
    factual_token_confidence: float   # median of factual tokens (prob < cutoff)
    n_factual_tokens:         int     # how many tokens were below cutoff
    n_total_tokens:           int     # total generated tokens
    prob_cutoff:              float   # the cutoff used (default 0.80)


# LLM GENERATOR — llama-cpp-python backend

class LlamaCppGenerator:
    """
    LLM generation via llama-cpp-python with logprob extraction.
    """

    DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant for competitive gaming wikis covering "
    "two separate games: Dota 2 and League of Legends. Answer the user's "
    "question using ONLY the context provided below.\n\n"

    "CROSS-DOMAIN INTERACTION RULE (critical):\n"
    "- Each chunk in the context begins with a structured header that "
    "identifies its game, for example 'Game: Dota 2 | Hero: ... | Data: ...' "
    "or 'Game: League of Legends | Champion: ... | Data: ...'. "
    "- Dota 2 and League of Legends are independent games with no shared "
    "mechanics, items, abilities, or interactions. Entities from the two "
    "games CANNOT interact with each other under any circumstance.\n"
    "- NEVER describe an interaction, comparison, synergy, counter, "
    "stacking behavior, or mechanical relationship between an entity "
    "from Dota 2 and an entity from League of Legends. Such interactions "
    "do not exist regardless of how the question is phrased.\n"
    "- If the question asks about a relationship between entities from "
    "different games, do not construct one. Instead, describe each "
    "entity separately within its own game with explicit game "
    "attribution, or state that the entities belong to separate games "
    "and cannot interact.\n"
    "- NEVER construct a claim about one game by inverting or contrasting "
    "a claim about the other game.\n"
    "- This rule applies even when context chunks for both entities "
    "appear well-grounded and topically relevant. Per-entity grounding "
    "does not imply cross-entity interaction.\n"
    "- Hedged or conditional framing is NOT permitted for cross-game "
    "relationships. Do not write phrases like 'if they interacted...', "
    "'we can infer that...', or 'this implies that...'"
    " when the entities involved belong to different games.\n\n"

    "REASONING:\n"
    "Before giving your final answer, reason step by step through the "
    "context to extract the correct information. If the question asks "
    "about a specific level, rank, or tier, identify which positional "
    "value corresponds to the requested level. Do NOT show your reasoning "
    "process in the response. Only provide the final answer with the "
    "specific values and facts.\n\n"

    "CONTEXT FORMAT GUIDE:\n"
    "- The context comes from a structured JSON knowledge base. Each "
    "chunk begins with a header of the form 'Game: <game> | "
    "<entity_type>: <entity_name> | Section: <section> | Data: <content>'. "
    "The 'Game:' field identifies which game the chunk describes; the "
    "content after 'Data:' is the substantive information.\n"
    "- Inside the Data content, you may encounter JSON syntax such as "
    "curly braces {}, square brackets [], and key-value pairs.\n"
    "- Square brackets [] denote lists. For example, a 'used_in' key "
    "with value ['Blade of the Ruined King', 'Wit\\'s End'] means the "
    "item is used in building those two items.\n"
    "- Slash-separated values like '15 / 40 / 80 / 150' represent "
    "scaling values that increase or decrease with level. The leftmost "
    "value is the lowest level, and the rightmost value is the highest "
    "level (maximum rank of the ability).\n"
    "- A slash-separated sequence may end with a trailing unit word that "
    "applies to every value in the sequence, not just the last one. For "
    "example, '80 / 90 / 100 / 110 / 120 Mana' means each of the five "
    "values is a mana cost (80 mana at rank 1, 120 mana at rank 5 — NOT "
    "'110' and a separate entity '120 Mana'). Similarly, "
    "'9 / 8 / 7 / 6 / 5 seconds' means each value is a duration in "
    "seconds. When extracting a specific value, strip the trailing unit "
    "and attach it to your chosen value (e.g., rank 5 of "
    "'9 / 8 / 7 / 6 / 5 seconds' is '5 seconds').\n"
    "- When asked about maximum level, or max rank, always use the "
    "RIGHTMOST value in the slash-separated sequence.\n"
    "- In League of Legends, champions have a maximum level of 18. "
    "If a question asks about level 18, treat it as maximum level "
    "and use the RIGHTMOST value.\n"
    "- In Dota 2, heroes have a maximum level of 30. "
    "If a question asks about level 30, treat it as maximum level "
    "and use the RIGHTMOST value.\n"
    "- When asked about level 1 or the base value, use the LEFTMOST "
    "value in the sequence.\n\n"

    "RESPONSE GUIDELINES:\n"
    "- For each game you have chunks for, prefix your statements with "
    "the game name: 'In Dota 2, ...' or 'In League of Legends, ...'. "
    "- Be precise and cite specific values, names, and mechanics from "
    "the context.\n"
    "- When answering about a specific level, extract the correct "
    "positional value from slash-separated sequences.\n"
    "- Keep answers concise and factual — avoid speculation.\n"
    "- When referencing specific heroes, champions, items, or abilities "
    "that are NOT explicitly mentioned in the user's question, always "
    "frame them as examples using phrases like 'for example', 'such as', "
    "or 'e.g.'. Never present unasked-for specific entities as if they "
    "were the focus of the question.\n"
    "- Never use ellipsis (...) in your response. Always write complete "
    "sentences.\n"
    "Answer strictly from the provided context."
)

    def __init__(
        self,
        model_path:     str,
        system_prompt:  str = None,
        temperature:    float = 0.5,
        top_p:          float = 0.9,
        max_tokens:     int = 512,
        n_ctx:          int = 5000,
        n_gpu_layers:   int = -1,
        seed:           int = 42,
        verbose:        bool = False,
    ):
        """
        Args:
            model_path:     Path to the .gguf model file.
            system_prompt:  Custom system prompt. If None, uses DEFAULT_SYSTEM_PROMPT.
            temperature:    Controls randomness. 0.5 = moderate (natural, grounded).
            top_p:          Nucleus sampling cutoff. 0.9 = 90% probability mass.
            max_tokens:     Max tokens to generate. 512 for wiki-style answers.
            n_ctx:          Context window size. 5000 fits our ~3,500 token budget.
            n_gpu_layers:   GPU offloading. -1 = all layers on GPU.
                            Set to 0 for CPU-only (if GPU is occupied).
            seed:           Random seed for reproducible generation.
            verbose:        If True, llama.cpp prints loading/inference details.
        """
        self.model_path = model_path
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self.verbose = verbose
        self._model = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently in memory."""
        return self._model is not None

    def load(self):
        """
        Load GGUF model into memory (GPU if n_gpu_layers != 0).

        VRAM usage: ~4.9 GB for Llama 3.1 8B Q4_K_M with n_gpu_layers=-1.

        logits_all=True is required for llama-cpp-python v0.3.16 to
        support logprob extraction. It computes logits for all token
        positions (not just the last), using ~50-100 MB extra VRAM.
        """
        if self._model is not None:
            print("  [Generator] Model already loaded, skipping.")
            return

        from llama_cpp import Llama
        print(f"  [Generator] Loading GGUF: {self.model_path}")
        t0 = time.perf_counter()

        self._model = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            seed=self.seed,
            logits_all=True,   # required for logprob extraction on v0.3.16
            verbose=self.verbose,
        )

        load_ms = (time.perf_counter() - t0) * 1000
        print(f"  [Generator] Model loaded ({load_ms:.0f}ms)")

    def unload(self, wait: float = 2.0):
        """
        Unload model from memory and free GPU VRAM.

        llama-cpp-python's Llama destructor calls llama_free() which
        releases ggml's CUDA allocations. gc.collect() ensures Python's
        garbage collector runs the destructor promptly.
        """
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            print(f"  [Generator] Model unloaded from memory.")

        if wait > 0:
            time.sleep(wait)

    def generate(self, query: str, context_str: str) -> GenerationOutput:
        """
        Generate an answer grounded in the provided context.
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Call load() before generate(). "
            )

        t0 = time.perf_counter()

        user_message = (
            f"Question: {query}\n\n"
            f"Context:\n"
            f"---\n"
            f"{context_str}\n"
            f"---\n\n"
            f"Based on the context above, answer the question: {query}"
        )

        response = self._model.create_chat_completion(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            logprobs=True,
            top_logprobs=3
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract answer text
        answer = response["choices"][0]["message"]["content"].strip()

        # Extract per-token logprobs 
        token_logprobs = []
        logprobs_content = (
            response["choices"][0]
            .get("logprobs", {})
            .get("content", [])
        )

        for tok_data in logprobs_content:
            lp = tok_data.get("logprob", 0.0)
            token_logprobs.append(TokenLogProb(
                token=tok_data.get("token", ""),
                logprob=lp,
                prob=math.exp(lp) if lp > -50 else 0.0,  # guard against -inf
            ))

        return GenerationOutput(
            answer=answer,
            latency_ms=round(latency_ms, 1),
            model=self.model_path.split("/")[-1].split("\\")[-1],
            token_logprobs=token_logprobs,
        )


# FACTUAL TOKEN ISOLATION + OVERRIDE DETECTION (EXPERIMENTAL)

def isolate_factual_tokens(
    token_logprobs: list[TokenLogProb],
    prob_cutoff:    float = 0.80,
) -> list[TokenLogProb]:
    """
    Filter generated tokens to those below the probability cutoff.

    Tokens above the cutoff (~0.80) are predominantly syntactic glue
    ("the", "is", "in", "of") that score high regardless of factual
    content. Tokens below the cutoff are where the model was actively
    choosing between alternatives
    """
    return [t for t in token_logprobs if t.prob < prob_cutoff]


def compute_factual_confidence(
    token_logprobs: list[TokenLogProb],
    prob_cutoff:    float = 0.80,
) -> tuple[float, int]:
    """
    Compute the median probability of factual (low-confidence) tokens.

    Edge cases:
        - Empty token list: returns (0.0, 0).
        - Zero factual tokens (all above cutoff): returns the actual
          median of ALL tokens. The model was highly confident on
          everything
        - One factual token: returns its probability directly.
        - Two or more: returns median of factual tokens.
    """
    if not token_logprobs:
        return 0.0, 0

    factual = isolate_factual_tokens(token_logprobs, prob_cutoff)

    if len(factual) == 0:
        # No tokens below cutoff: model was confident on everything.
        # Return actual median of all tokens for honest signal.
        all_probs = [t.prob for t in token_logprobs]
        return float(np.median(all_probs)), 0
    elif len(factual) == 1:
        return factual[0].prob, 1
    else:
        probs = [t.prob for t in factual]
        return float(np.median(probs)), len(factual)


def detect_parametric_override(
    token_logprobs:  list[TokenLogProb],
    faithfulness_max: float,
    override_token_threshold: float = 0.75,
    override_faith_threshold: float = 0.40,
    prob_cutoff:              float = 0.80,
) -> OverrideDetectionResult:
    """
    Binary parametric override detector.

    Fires when the model was CONFIDENT in its generation (high factual
    token probability) but the answer is NOT GROUNDED in context (low
    faithfulness). This combination indicates the model generated from
    parametric memory, ignoring the retrieved context.

    The 2D detection space:
        High faith + High token prob = best case (confident + grounded)
        High faith + Low token prob  = uncertain but grounded (fine for RAG)
        Low faith  + High token prob = PARAMETRIC OVERRIDE (dangerous)
        Low faith  + Low token prob  = pure hallucination (dead tier)
    """
    factual_confidence, n_factual = compute_factual_confidence(
        token_logprobs, prob_cutoff
    )

    is_override = (
        factual_confidence > override_token_threshold
        and faithfulness_max < override_faith_threshold
    )

    return OverrideDetectionResult(
        is_override=is_override,
        factual_token_confidence=round(factual_confidence, 4),
        n_factual_tokens=n_factual,
        n_total_tokens=len(token_logprobs),
        prob_cutoff=prob_cutoff,
    )


# ENTITY MISMATCH DETECTION
# When all admitted chunks reference the same entity (hero/champion/item)
# that the user did NOT ask about, the LLM tends to write as if the user
# asked about that entity. This function detects the mismatch and returns
# a warning string to prepend to the context, steering the LLM to frame
# entity-specific info as examples rather than direct answers.


def detect_entity_mismatch(
    query:           str,
    reranker_output,
) -> str | None:
    """
    Detect if admitted chunks reference entities absent from the query.

    Scans metadata fields (hero, champion, item) of all admitted chunks.
    If every chunk references the same entity and that entity does NOT
    appear in the query, returns a warning string to prepend to context.
    """
    chunk_entities = set()
    for r in reranker_output.context_chunks:
        meta = r.chunk.metadata
        for key in ("hero", "champion", "item"):
            val = meta.get(key)
            if val:
                chunk_entities.add(val)

    if not chunk_entities:
        return None

    query_lower = query.lower()
    unmentioned = [e for e in chunk_entities if e.lower() not in query_lower]

    if not unmentioned:
        return None

    entity_list = ", ".join(unmentioned)
    first = unmentioned[0]
    plural = "these entities" if len(unmentioned) > 1 else "this entity"

    return (
        f"IMPORTANT: The context below is specifically about {entity_list}. "
        f"The user did NOT ask about {plural}. "
        f"If you use information from this context, frame it as an example: "
        f"'for example, with {first}, ...' - do NOT present the "
        f"answer as if the user asked about {first}.\n\n"
    )


# ADAPTIVE CHUNK INJECTION (RARE CASE)
# When the LLM abstains ("context does not contain enough information")
# despite gated-out chunks existing, we inject the highest-scoring
# gated-out chunk into the context and regenerate. Only the FINAL
# answer is passed to the faithfulness checker

_ABSTENTION_PHRASES = [
    "does not contain sufficient information",
    "does not contain enough information",
    "cannot answer this question based on",
    "not enough information to answer",
    "insufficient information to answer",
    "no relevant information",
    "is not mentioned in the provided context",
    "is not mentioned in the context",
    "not mentioned in the provided context",
    "no information about",
    "does not mention",
    "does not provide information",
    "there is no information",
]


def _is_abstention(answer: str) -> bool:
    """Check if the LLM refused to answer."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in _ABSTENTION_PHRASES)


@dataclass
class AdaptiveRetryResult:
    """
    Output from generate_with_adaptive_retry().

    If injection was NOT needed (first attempt succeeded), this wraps the
    original GenerationOutput unchanged, with injection_used=False.

    If injection WAS used, this contains the second-attempt GenerationOutput
    and the injected chunk's reranker score for confidence recalculation.
    """
    generation:              GenerationOutput   # final answer (attempt 1 or 2)
    injection_used:          bool               # True if gated-out chunk was injected
    injected_chunk_score:    float | None       # reranker_score of injected chunk 
    injected_chunk_id:       str | None         # chunk_id for logging
    original_generation:     GenerationOutput | None  # attempt 1 output (only if injection was tried)
    context_str:             str                # final context string passed to LLM


def generate_with_adaptive_retry(
    generator:        "LlamaCppGenerator",
    query:            str,
    context_str:      str,
    reranker_output,
    status_callback=None,
) -> AdaptiveRetryResult:
    """
    Generate with optional retry via adaptive chunk injection.

    Flow:
        1. Generate answer with gated-in context.
        2. If LLM abstains AND gated-out chunks exist:
           a. Show status message ("Hmm, this seems harder than it looks...")
           b. Inject the highest-scoring gated-out chunk into context.
           c. Regenerate with expanded context.
           d. If second attempt ALSO abstains -> fall back to original answer.
        3. Return final answer for faithfulness checking.
    """
    # Attempt 1: Generate with original context
    gen1 = generator.generate(query, context_str)

    if not _is_abstention(gen1.answer):
        # First attempt succeeded - no injection needed
        return AdaptiveRetryResult(
            generation=gen1,
            injection_used=False,
            injected_chunk_score=None,
            injected_chunk_id=None,
            original_generation=None,
            context_str=context_str,
        )

    # LLM abstained - look for gated-out chunks to inject
    gated_out = [
        r for r in reranker_output.all_scored
        if not r.passed_gate
    ]

    if not gated_out:
        # No gated-out chunks available - return original answer
        print("  [Adaptive Retry] LLM abstained but no gated-out chunks available.")
        return AdaptiveRetryResult(
            generation=gen1,
            injection_used=False,
            injected_chunk_score=None,
            injected_chunk_id=None,
            original_generation=None,
            context_str=context_str,
        )

    # Inject highest-scoring gated-out chunk
    best_gated = gated_out[0]

    if status_callback:
        status_callback("Hmm, this seems harder than it looks... Let me look deeper.")
    print(f"  [Adaptive Retry] Injecting gated-out chunk: "
          f"{best_gated.chunk.chunk_id[:60]} "
          f"(score: {best_gated.reranker_score:.4f})")

    # Build expanded context: original + injected chunk at the END
    # (recency-biased placement — injected chunk is the newest addition,
    # but it's the lowest-quality chunk, so we place it at the START
    # of the context where the model attends least)
    expanded_context = (
        best_gated.chunk.page_content + "\n\n" + context_str
    )

    # Attempt 2: Regenerate with expanded context
    gen2 = generator.generate(query, expanded_context)

    if _is_abstention(gen2.answer):
        # Second attempt also failed - fall back to original
        print("  [Adaptive Retry] Second attempt also abstained. "
              "Falling back to original answer.")
        return AdaptiveRetryResult(
            generation=gen1,
            injection_used=False,
            injected_chunk_score=None,
            injected_chunk_id=None,
            original_generation=None,
            context_str=context_str,
        )

    # Injection succeeded
    print(f"  [Adaptive Retry] Injection helped - using second-attempt answer.")
    return AdaptiveRetryResult(
        generation=gen2,
        injection_used=True,
        injected_chunk_score=best_gated.reranker_score,
        injected_chunk_id=best_gated.chunk.chunk_id,
        original_generation=gen1,
        context_str=expanded_context,
    )