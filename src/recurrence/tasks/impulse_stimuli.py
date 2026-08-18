"""Impulse Stimuli & Multi-Regime Filler Generators (Sprint S11).

Provides length-equated stimulus pairs and 4 distinct filler regimes:
1. Constant-Token Filler (repeating neutral token ID)
2. Diverse Random Tokens (uniform sampling from audited neutral pool with fixed seed)
3. Natural Prose Narrative (frozen deterministic corpus text)
4. Active Semantic Interference (distractor entity-attribute sentences)
"""

from dataclasses import dataclass
import hashlib
import random
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ImpulseStimulusPair:
    """Length-equated stimulus pair for matched trajectory impulse response."""
    pair_id: str
    prefix: str
    event_a: str
    event_b: str
    target_a: str
    target_b: str
    query: str


# Canonical length-equated stimulus bank
CANONICAL_STIMULI_PAIRS: List[ImpulseStimulusPair] = [
    ImpulseStimulusPair(
        pair_id="item_material_01",
        prefix="Notice: ",
        event_a="The marked object was amber. ",
        event_b="The marked object was cobalt. ",
        target_a="amber",
        target_b="cobalt",
        query="Question: What color was the marked object? Answer:",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_02",
        prefix="Log entry: ",
        event_a="The container held copper. ",
        event_b="The container held silver. ",
        target_a="copper",
        target_b="silver",
        query="Question: What metal was in the container? Answer:",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_03",
        prefix="Observation: ",
        event_a="The artifact was garnet. ",
        event_b="The artifact was zircon. ",
        target_a="garnet",
        target_b="zircon",
        query="Question: What mineral was the artifact? Answer:",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_04",
        prefix="Report: ",
        event_a="The signal detected was quartz. ",
        event_b="The signal detected was basalt. ",
        target_a="quartz",
        target_b="basalt",
        query="Question: What rock was detected in the signal? Answer:",
    ),
]

FROZEN_NATURAL_PROSE_TEXT = (
    "The atmospheric pressure remained steady throughout the afternoon as weather stations across the valley "
    "recorded temperature gradients. Instruments calibrated for seasonal monitoring observed standard humidity "
    "readings along the river basin. Field technicians confirmed that data transmission protocols operated within "
    "expected variance limits, allowing automated archival systems to process incoming sensor streams sequentially. "
    "Meanwhile, power consumption metrics in the central facility demonstrated consistent baseline operation. "
    "Routine inspections of communication infrastructure verified that optical cables maintained nominal bandwidth, "
    "preventing latency accumulation across regional distribution networks."
)

FROZEN_SEMANTIC_INTERFERENCE_TEXT = (
    "The second specimen was marble. The third specimen was bronze. The fourth specimen was obsidian. "
    "The auxiliary unit was brass. The backup unit was platinum. The primary unit was limestone. "
    "The external fixture was slate. The internal fixture was chromium. The final component was sandstone. "
    "The first sample was titanium. The second sample was granite. The third sample was nickel. "
    "The fourth sample was iron. The fifth sample was dolomite. The sixth sample was diamond. "
    "The remaining sample was sapphire. The upper casing was aluminum. The lower casing was silicon."
)


def audit_stimulus_token_equality(
    pair: ImpulseStimulusPair,
    tokenizer: Any,
) -> Tuple[bool, int, int]:
    """Verify that Event A and Event B tokenize to the exact same number of tokens."""
    tokens_a = tokenizer.encode(pair.event_a, add_special_tokens=False) if tokenizer else pair.event_a.split()
    tokens_b = tokenizer.encode(pair.event_b, add_special_tokens=False) if tokenizer else pair.event_b.split()
    return len(tokens_a) == len(tokens_b), len(tokens_a), len(tokens_b)


def generate_constant_filler(
    length: int,
    constant_token_id: int = 15,
) -> List[int]:
    """Generate constant-token filler by repeating a single neutral token ID."""
    return [constant_token_id] * length


def generate_random_filler(
    length: int,
    seed: int,
    vocab_size: int,
    exclude_tokens: Optional[Set[int]] = None,
) -> List[int]:
    """Generate diverse random tokens sampled uniformly from an audited vocabulary pool."""
    rng = random.Random(seed)
    excluded = exclude_tokens or set()
    # Sample from neutral vocabulary space (avoiding first 10 control tokens and exclusions)
    valid_token_pool = [t for t in range(10, min(vocab_size, 10000)) if t not in excluded]
    if not valid_token_pool:
        valid_token_pool = [10, 11, 12, 13, 14, 15]
    return [rng.choice(valid_token_pool) for _ in range(length)]


def generate_natural_filler(
    length: int,
    tokenizer: Optional[Any] = None,
) -> List[int]:
    """Generate natural prose narrative filler from a frozen local text segment."""
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(FROZEN_NATURAL_PROSE_TEXT, add_special_tokens=False)
    else:
        # Synthetic deterministic token pool
        tokens = [((i * 17 + 23) % 150) + 10 for i in range(100)]

    # Cycle if length exceeds available prose
    out: List[int] = []
    while len(out) < length:
        out.extend(tokens)
    return out[:length]


def generate_interfering_filler(
    length: int,
    tokenizer: Optional[Any] = None,
) -> List[int]:
    """Generate active semantic interference filler (distractor entity-property sentences)."""
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(FROZEN_SEMANTIC_INTERFERENCE_TEXT, add_special_tokens=False)
    else:
        tokens = [((i * 31 + 47) % 150) + 10 for i in range(100)]

    out: List[int] = []
    while len(out) < length:
        out.extend(tokens)
    return out[:length]


def get_filler_tokens_for_regime(
    regime: str,
    length: int,
    seed: int,
    vocab_size: int,
    tokenizer: Optional[Any] = None,
    exclude_tokens: Optional[Set[int]] = None,
) -> List[int]:
    """Dispatch filler token generation for the specified regime."""
    if regime == "neutral_repeated" or regime == "constant":
        return generate_constant_filler(length)
    elif regime == "random_tokens" or regime == "random":
        return generate_random_filler(length, seed=seed, vocab_size=vocab_size, exclude_tokens=exclude_tokens)
    elif regime == "natural_prose" or regime == "natural":
        return generate_natural_filler(length, tokenizer=tokenizer)
    elif regime == "semantic_interference" or regime == "interfering":
        return generate_interfering_filler(length, tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown filler regime '{regime}'")
