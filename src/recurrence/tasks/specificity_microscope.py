"""Sprint S12c: Specificity Microscope Stimulus Battery.

Disentangles value-specific historical memory from same-template / shared-event alignment
at 2W = 4096 tokens across 4 balanced syntactic template families.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import hashlib
import json


@dataclass(frozen=True)
class MicroscopeTemplateFamily:
    family_id: str
    family_name: str
    template_str: str  # e.g. "The marked object was {val}."
    query_prefix: str  # e.g. "The marked object was "
    values: Tuple[str, ...]  # 4 audited single-token values


MICROSCOPE_FAMILIES: Tuple[MicroscopeTemplateFamily, ...] = (
    MicroscopeTemplateFamily(
        family_id="marked_object",
        family_name="Marked Object",
        template_str="The marked object was {val}.",
        query_prefix="The marked object was ",
        values=("amber", "cobalt", "garnet", "zircon"),
    ),
    MicroscopeTemplateFamily(
        family_id="sealed_container",
        family_name="Sealed Container",
        template_str="The sealed container held {val}.",
        query_prefix="The sealed container held ",
        values=("copper", "silver", "bronze", "nickel"),
    ),
    MicroscopeTemplateFamily(
        family_id="monitored_signal",
        family_name="Monitored Signal",
        template_str="The monitored signal showed {val}.",
        query_prefix="The monitored signal showed ",
        values=("alpha", "delta", "gamma", "theta"),
    ),
    MicroscopeTemplateFamily(
        family_id="archived_artifact",
        family_name="Archived Artifact",
        template_str="The archived artifact contained {val}.",
        query_prefix="The archived artifact contained ",
        values=("marble", "quartz", "basalt", "granite"),
    ),
)


@dataclass(frozen=True)
class MicroscopePair:
    pair_id: str
    family_id: str
    val_a: str
    val_b: str
    val_c: str  # Same-template wrong-value donor 1
    val_d: str  # Same-template wrong-value donor 2
    cross_family_id: str  # Cross-template family ID
    cross_val: str  # Cross-template donor value
    prefix: str
    prompt_a: str
    prompt_b: str
    prompt_c: str
    prompt_d: str
    prompt_cross: str
    query: str
    target_a: str
    target_b: str


def build_microscope_pairs() -> List[MicroscopePair]:
    """Build all 24 canonical value pairs across 4 template families."""
    pairs: List[MicroscopePair] = []
    num_families = len(MICROSCOPE_FAMILIES)

    for f_idx, fam in enumerate(MICROSCOPE_FAMILIES):
        vals = fam.values
        cross_fam = MICROSCOPE_FAMILIES[(f_idx + 1) % num_families]
        cross_val = cross_fam.values[0]

        # Generate all 6 unordered pairs (i < j)
        pair_num = 1
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                val_a = vals[i]
                val_b = vals[j]
                # Remaining two values in the family serve as same-template wrong values
                remaining = [v for v in vals if v not in (val_a, val_b)]
                val_c, val_d = remaining[0], remaining[1]

                pair_id = f"{fam.family_id}_p{pair_num:02d}_{val_a}_{val_b}"
                pair_num += 1

                prefix = "Beginning formal trial log.\n"
                prompt_a = f"{prefix}{fam.template_str.format(val=val_a)}\n"
                prompt_b = f"{prefix}{fam.template_str.format(val=val_b)}\n"
                prompt_c = f"{prefix}{fam.template_str.format(val=val_c)}\n"
                prompt_d = f"{prefix}{fam.template_str.format(val=val_d)}\n"
                prompt_cross = f"{prefix}{cross_fam.template_str.format(val=cross_val)}\n"
                query = fam.query_prefix

                # Targets have leading space
                target_a = f" {val_a}"
                target_b = f" {val_b}"

                pairs.append(MicroscopePair(
                    pair_id=pair_id,
                    family_id=fam.family_id,
                    val_a=val_a,
                    val_b=val_b,
                    val_c=val_c,
                    val_d=val_d,
                    cross_family_id=cross_fam.family_id,
                    cross_val=cross_val,
                    prefix=prefix,
                    prompt_a=prompt_a,
                    prompt_b=prompt_b,
                    prompt_c=prompt_c,
                    prompt_d=prompt_d,
                    prompt_cross=prompt_cross,
                    query=query,
                    target_a=target_a,
                    target_b=target_b,
                ))

    return pairs


def audit_microscope_panel(tokenizer: Any) -> Tuple[bool, str, Dict[str, Any]]:
    """Verify single-token lengths and leading space properties for all microscope values."""
    all_values = set()
    for fam in MICROSCOPE_FAMILIES:
        for val in fam.values:
            all_values.add(val)

    token_info = {}
    is_valid = True
    err_msgs = []

    for val in sorted(list(all_values)):
        toks_bare = tokenizer.encode(val, add_special_tokens=False)
        toks_spaced = tokenizer.encode(f" {val}", add_special_tokens=False)
        token_info[val] = {
            "bare_tokens": toks_bare,
            "bare_len": len(toks_bare),
            "spaced_tokens": toks_spaced,
            "spaced_len": len(toks_spaced),
        }
        if len(toks_spaced) != 1:
            is_valid = False
            err_msgs.append(f"Value '{val}' tokenizes to {len(toks_spaced)} tokens with leading space: {toks_spaced}")

    panel_json = json.dumps(token_info, sort_keys=True)
    panel_hash = hashlib.sha256(panel_json.encode("utf-8")).hexdigest()

    status_msg = "PASSED: All 16 values tokenize to exactly 1 token with leading space." if is_valid else "; ".join(err_msgs)
    return is_valid, panel_hash, {"status": status_msg, "token_info": token_info}
