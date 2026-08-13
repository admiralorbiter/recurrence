"""Post-decision confidence separation and rank discrimination analytics."""

from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_auroc2(confidences: List[float], correct: List[bool]) -> Optional[float]:
    """Compute Type-2 ROC Area Under Curve (AUROC2) for confidence discriminating accuracy."""
    if not confidences or len(confidences) != len(correct):
        return None
    
    # Filter valid non-None entries
    pairs = [(c, y) for c, y in zip(confidences, correct) if c is not None]
    if not pairs:
        return None
    
    confs, labels = zip(*pairs)
    labels = np.array(labels, dtype=bool)
    confs = np.array(confs, dtype=float)

    n_pos = np.sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        # Cannot compute AUC if all trials are correct or all are incorrect
        return None

    # Rank-sum calculation (Mann-Whitney U statistic)
    order = np.lexsort((np.random.RandomState(42).rand(len(confs)), confs))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(confs) + 1)

    # Handle ties by averaging ranks
    unique_vals = np.unique(confs)
    for val in unique_vals:
        ties = confs == val
        if np.sum(ties) > 1:
            ranks[ties] = np.mean(ranks[ties])

    u_stat = np.sum(ranks[labels]) - (n_pos * (n_pos + 1)) / 2.0
    auroc2 = float(u_stat / (n_pos * n_neg))
    return float(np.clip(auroc2, 0.0, 1.0))


def compute_post_decision_discrimination_from_pairs(
    paired_observations: List[Tuple[Optional[int], bool]]
) -> Dict[str, Optional[float]]:
    """Compute post-decision confidence separation and AUROC2 from paired (confidence, correct) tuples.

    Avoids indexing desynchronization by operating strictly over paired records.
    """
    valid_pairs = [(c, y) for c, y in paired_observations if c is not None]
    if not valid_pairs:
        return {
            "valid_confidence_count": 0,
            "mean_confidence_correct": None,
            "mean_confidence_incorrect": None,
            "confidence_separation": None,
            "auroc2": None,
        }

    confs_raw, labels = zip(*valid_pairs)
    labels = np.array(labels, dtype=float)
    confs_raw = np.array(confs_raw, dtype=float)

    # 1. Mean confidence by correctness
    pos_mask = labels == 1.0
    neg_mask = labels == 0.0

    mean_conf_pos = float(np.mean(confs_raw[pos_mask])) if np.any(pos_mask) else None
    mean_conf_neg = float(np.mean(confs_raw[neg_mask])) if np.any(neg_mask) else None
    separation = (
        float(mean_conf_pos - mean_conf_neg)
        if (mean_conf_pos is not None and mean_conf_neg is not None)
        else None
    )

    # 2. AUROC2 rank discrimination
    auroc = compute_auroc2(list(confs_raw), list(labels.astype(bool)))

    return {
        "valid_confidence_count": len(valid_pairs),
        "mean_confidence_correct": mean_conf_pos,
        "mean_confidence_incorrect": mean_conf_neg,
        "confidence_separation": separation,
        "auroc2": auroc,
    }


def compute_calibration_metrics(
    confidences_1_to_5: List[Optional[int]],
    correct_flags: List[bool]
) -> Dict[str, Optional[float]]:
    """Convenience wrapper for lists of confidences and correct flags."""
    paired = list(zip(confidences_1_to_5, correct_flags))
    return compute_post_decision_discrimination_from_pairs(paired)
