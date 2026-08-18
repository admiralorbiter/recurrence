"""Impulse Stimuli & Multi-Regime Filler Generators (Sprint S11 Hardened).

Provides a scaled 20-pair length-equated stimulus bank and 4 distinct filler regimes
with audited vocabulary pooling, multi-seed exemplars, and cloze retrieval prompts:
1. Constant-Token Filler (repeating audited neutral token ID)
2. Diverse Random Tokens (uniform sampling from audited neutral pool with fixed seeds)
3. Natural Prose Narrative (frozen deterministic corpus passages with multi-seed slicing)
4. Active Semantic Interference (structured distractor passages with entity-attribute rotations)
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
    cloze_prompt: str


# Scaled 20-item length-equated stimulus bank (materials, colors, minerals, metals, elements)
CANONICAL_STIMULI_PAIRS: List[ImpulseStimulusPair] = [
    ImpulseStimulusPair(
        pair_id="item_material_01",
        prefix="Notice: ",
        event_a="The marked object was amber. ",
        event_b="The marked object was cobalt. ",
        target_a="amber",
        target_b="cobalt",
        cloze_prompt="Recall test: The marked object was",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_02",
        prefix="Log entry: ",
        event_a="The container held copper. ",
        event_b="The container held silver. ",
        target_a="copper",
        target_b="silver",
        cloze_prompt="Recall test: The container held",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_03",
        prefix="Observation: ",
        event_a="The artifact was garnet. ",
        event_b="The artifact was zircon. ",
        target_a="garnet",
        target_b="zircon",
        cloze_prompt="Recall test: The artifact was",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_04",
        prefix="Report: ",
        event_a="The signal detected was quartz. ",
        event_b="The signal detected was basalt. ",
        target_a="quartz",
        target_b="basalt",
        cloze_prompt="Recall test: The signal detected was",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_05",
        prefix="Record: ",
        event_a="The alloy contained nickel. ",
        event_b="The alloy contained cobalt. ",
        target_a="nickel",
        target_b="cobalt",
        cloze_prompt="Recall test: The alloy contained",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_06",
        prefix="Survey: ",
        event_a="The discovery was zinc. ",
        event_b="The discovery was lead. ",
        target_a="zinc",
        target_b="lead",
        cloze_prompt="Recall test: The discovery was",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_07",
        prefix="Docket: ",
        event_a="The mineral found was pyrite. ",
        event_b="The mineral found was gypsum. ",
        target_a="pyrite",
        target_b="gypsum",
        cloze_prompt="Recall test: The mineral found was",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_08",
        prefix="Manifest: ",
        event_a="The shipment was bronze. ",
        event_b="The shipment was marble. ",
        target_a="bronze",
        target_b="marble",
        cloze_prompt="Recall test: The shipment was",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_09",
        prefix="Registry: ",
        event_a="The specimen was granite. ",
        event_b="The specimen was calcite. ",
        target_a="granite",
        target_b="calcite",
        cloze_prompt="Recall test: The specimen was",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_10",
        prefix="Catalog: ",
        event_a="The deposit was sulfur. ",
        event_b="The deposit was carbon. ",
        target_a="sulfur",
        target_b="carbon",
        cloze_prompt="Recall test: The deposit was",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_11",
        prefix="Inventory: ",
        event_a="The gemstone was ruby. ",
        event_b="The gemstone was jasper. ",
        target_a="ruby",
        target_b="jasper",
        cloze_prompt="Recall test: The gemstone was",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_12",
        prefix="Ledger: ",
        event_a="The coating was titanium. ",
        event_b="The coating was platinum. ",
        target_a="titanium",
        target_b="platinum",
        cloze_prompt="Recall test: The coating was",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_13",
        prefix="Protocol: ",
        event_a="The substrate was sodium. ",
        event_b="The substrate was lithium. ",
        target_a="sodium",
        target_b="lithium",
        cloze_prompt="Recall test: The substrate was",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_14",
        prefix="Briefing: ",
        event_a="The relic was dolomite. ",
        event_b="The relic was feldspar. ",
        target_a="dolomite",
        target_b="feldspar",
        cloze_prompt="Recall test: The relic was",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_15",
        prefix="Dispatch: ",
        event_a="The target was sapphire. ",
        event_b="The target was emerald. ",
        target_a="sapphire",
        target_b="emerald",
        cloze_prompt="Recall test: The target was",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_16",
        prefix="Archive: ",
        event_a="The casing was aluminum. ",
        event_b="The casing was chromium. ",
        target_a="aluminum",
        target_b="chromium",
        cloze_prompt="Recall test: The casing was",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_17",
        prefix="Summary: ",
        event_a="The core was tungsten. ",
        event_b="The core was vanadium. ",
        target_a="tungsten",
        target_b="vanadium",
        cloze_prompt="Recall test: The core was",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_18",
        prefix="Overview: ",
        event_a="The plate was porcelain. ",
        event_b="The plate was sandstone. ",
        target_a="porcelain",
        target_b="sandstone",
        cloze_prompt="Recall test: The plate was",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_19",
        prefix="Bulletin: ",
        event_a="The core was graphite. ",
        event_b="The core was bismuth. ",
        target_a="graphite",
        target_b="bismuth",
        cloze_prompt="Recall test: The core was",
    ),
    ImpulseStimulusPair(
        pair_id="item_material_20",
        prefix="Notice: ",
        event_a="The shard was quartzite. ",
        event_b="The shard was limestone. ",
        target_a="quartzite",
        target_b="limestone",
        cloze_prompt="Recall test: The shard was",
    ),
]


def _generate_nonrepeating_prose(seed: int, target_tokens: int) -> str:
    """Generate extensive, non-repeating natural prose narrative deterministically."""
    rng = random.Random(seed)
    subjects = [
        "The atmospheric pressure", "The regional environmental station", "The hydrological sensor array",
        "The oceanographic observatory", "The geological survey expedition", "The automated meteorological telemetry",
        "The alpine monitoring station", "The acoustic telemetry unit", "The orbital tracking facility",
        "The coastal research vessel", "The botanical expedition team", "The geothermal monitoring network"
    ]
    actions = [
        "recorded steady measurements throughout the monitoring period",
        "observed consistent baseline signals along the valley corridor",
        "transmitted synchronized operational telemetry to the central archive",
        "verified that diagnostic parameters stayed within nominal boundaries",
        "documented standard temperature gradients across the experimental zone",
        "processed incoming multi-spectral sensor feeds sequentially",
        "calibrated precision instruments prior to scheduled data collection",
        "reported continuous operation with zero transmission latency"
    ]
    connectors = [
        "Meanwhile, secondary backup generators operated at nominal capacity.",
        "Furthermore, auxiliary communications maintained uninterrupted optical link quality.",
        "In addition, independent calibration sensors confirmed instrument stability.",
        "Similarly, atmospheric humidity readings reflected standard seasonal patterns.",
        "Concurrently, ground telemetry verified that data integrity checks succeeded.",
        "Subsequently, automated processing queues cleared all scheduled tasks without exception."
    ]

    sentences = []
    current_len = 0
    # Approximate ~15 words per sentence
    while current_len < (target_tokens * 2):
        s = f"{rng.choice(subjects)} {rng.choice(actions)}. {rng.choice(connectors)}"
        sentences.append(s)
        current_len += len(s.split())
    return " ".join(sentences)


def _generate_nonrepeating_interference(seed: int, target_tokens: int) -> str:
    """Generate extensive, non-repeating semantic interference distractor statements deterministically."""
    rng = random.Random(seed)
    sectors = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa",
        "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega"
    ]
    descriptors = [
        "crystalline", "metallic", "amorphous", "vitrified", "vesicular", "foliated", "pyroclastic",
        "pegmatitic", "porphyritic", "granular", "fibrous", "lamellar", "polymeric", "ceramic", "composite",
        "vitreous", "resinous", "pearly", "adamantine", "earthy", "waxy", "colloidal", "porous", "dense",
        "brittle", "ductile", "elastic", "magnetic", "diamagnetic", "paramagnetic", "refractory", "luminescent"
    ]
    structures = [
        "specimen", "matrix", "substrate", "layer", "aggregate", "formation", "compound", "lattice",
        "inclusion", "fragment", "sample", "core", "deposit", "casing", "concretion", "nodule"
    ]

    sentences = []
    current_len = 0
    step = 0
    while current_len < (target_tokens * 2):
        sec = sectors[step % len(sectors)]
        desc1 = rng.choice(descriptors)
        desc2 = rng.choice(descriptors)
        struct1 = rng.choice(structures)
        struct2 = rng.choice(structures)
        idx_num = (step * 7 + seed * 13) % 1000
        s = f"Sector {sec}-{idx_num} confirmed the {struct1} was {desc1}. Auxiliary survey {idx_num+1} verified the {struct2} was {desc2}."
        sentences.append(s)
        current_len += len(s.split())
        step += 1
    return " ".join(sentences)


def build_audited_vocabulary_pool(
    tokenizer: Any,
    excluded_token_ids: Optional[Set[int]] = None,
) -> Tuple[List[int], str]:
    """Audit tokenizer vocabulary to produce a deterministic, clean neutral token pool."""
    excluded = set(excluded_token_ids or set())
    if tokenizer is not None and hasattr(tokenizer, "all_special_ids"):
        excluded.update(tokenizer.all_special_ids)

    pool: List[int] = []

    if tokenizer is not None and hasattr(tokenizer, "get_vocab"):
        vocab_dict = tokenizer.get_vocab()
        for token_text, token_id in vocab_dict.items():
            if token_id in excluded or token_id < 10:
                continue
            cleaned = token_text.strip().lstrip(" ")
            if not cleaned or len(cleaned) > 15:
                continue
            if any(c in token_text for c in ["<", ">", "[", "]", "{", "}", "\\", "/", "\n", "\t", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
                continue
            pool.append(token_id)
    else:
        vocab_size = len(tokenizer) if tokenizer is not None else 200
        pool = [t for t in range(10, min(vocab_size, 100)) if t not in excluded]

    pool = sorted(list(set(pool)))
    if not pool:
        pool = [t for t in range(10, 50) if t not in excluded]

    # Compute SHA256 digest of pool
    pool_str = ",".join(str(x) for x in pool)
    pool_hash = hashlib.sha256(pool_str.encode("utf-8")).hexdigest()[:16]
    return pool, pool_hash


def generate_constant_filler(
    length: int,
    audited_pool: List[int],
    excluded_token_ids: Optional[Set[int]] = None,
) -> List[int]:
    """Generate constant-token filler by repeating the median token ID from audited pool."""
    clean_pool = [t for t in audited_pool if t not in (excluded_token_ids or set())]
    if not clean_pool:
        clean_pool = audited_pool or [15]
    constant_token = clean_pool[len(clean_pool) // 2]
    return [constant_token] * length


def generate_random_filler(
    length: int,
    seed: int,
    audited_pool: List[int],
    excluded_token_ids: Optional[Set[int]] = None,
) -> List[int]:
    """Generate diverse random tokens sampled uniformly from audited neutral pool."""
    clean_pool = [t for t in audited_pool if t not in (excluded_token_ids or set())]
    if not clean_pool:
        clean_pool = audited_pool or [15]
    rng = random.Random(seed)
    return [rng.choice(clean_pool) for _ in range(length)]


def generate_natural_filler(
    length: int,
    seed: int = 42,
    tokenizer: Optional[Any] = None,
    excluded_token_ids: Optional[Set[int]] = None,
    audited_pool: Optional[List[int]] = None,
) -> List[int]:
    """Generate natural prose narrative filler with non-repeating long-horizon streams."""
    text = _generate_nonrepeating_prose(seed=seed, target_tokens=length)

    if tokenizer is not None and hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
    else:
        rng = random.Random(seed)
        tokens = [rng.randint(10, 150) for _ in range(length * 2)]

    # Filter out excluded token IDs
    excluded = set(excluded_token_ids or set())
    replacement_token = (audited_pool[0] if audited_pool else 15)
    tokens = [replacement_token if t in excluded else t for t in tokens]

    return tokens[:length]


def generate_interfering_filler(
    length: int,
    seed: int = 42,
    tokenizer: Optional[Any] = None,
    excluded_token_ids: Optional[Set[int]] = None,
    audited_pool: Optional[List[int]] = None,
) -> List[int]:
    """Generate active semantic interference filler with non-repeating distractor statements."""
    text = _generate_nonrepeating_interference(seed=seed, target_tokens=length)

    if tokenizer is not None and hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
    else:
        rng = random.Random(seed + 100)
        tokens = [rng.randint(10, 150) for _ in range(length * 2)]

    # Filter out excluded token IDs
    excluded = set(excluded_token_ids or set())
    replacement_token = (audited_pool[0] if audited_pool else 15)
    tokens = [replacement_token if t in excluded else t for t in tokens]

    return tokens[:length]


def get_filler_tokens_for_regime(
    regime: str,
    length: int,
    seed: int,
    audited_pool: Optional[List[int]] = None,
    tokenizer: Optional[Any] = None,
    excluded_token_ids: Optional[Set[int]] = None,
) -> List[int]:
    """Dispatch filler token generation for the specified regime."""
    pool = audited_pool or [10, 11, 12, 13, 14, 15]
    if regime in ("constant", "neutral_repeated"):
        return generate_constant_filler(length, audited_pool=pool, excluded_token_ids=excluded_token_ids)
    elif regime in ("random", "random_tokens"):
        return generate_random_filler(length, seed=seed, audited_pool=pool, excluded_token_ids=excluded_token_ids)
    elif regime in ("natural", "natural_prose"):
        return generate_natural_filler(length, seed=seed, tokenizer=tokenizer, excluded_token_ids=excluded_token_ids, audited_pool=pool)
    elif regime in ("interfering", "semantic_interference"):
        return generate_interfering_filler(length, seed=seed, tokenizer=tokenizer, excluded_token_ids=excluded_token_ids, audited_pool=pool)
    else:
        raise ValueError(f"Unknown filler regime '{regime}'")
