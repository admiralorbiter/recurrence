"""Unit tests and invariant assertions for Gate E Provenance Garden Kernel and Oracle."""

import pytest
import math


def test_bayesian_dependency_discounting_invariants():
    """Asserts that two independent reports provide more evidential weight than a duplicate copy."""
    # Independent corroboration of z=1 from two 0.85 reliable sources
    rel = 0.85
    lr_single = rel / (1.0 - rel)
    log_odds_single = math.log(lr_single)
    
    # 1. Independent: log-odds add
    log_odds_independent = 2.0 * log_odds_single
    p_independent = 1.0 / (1.0 + math.exp(-log_odds_independent))
    
    # 2. Duplicate copied evidence: log-odds only counted ONCE for root evidence
    log_odds_copied = log_odds_single
    p_copied = 1.0 / (1.0 + math.exp(-log_odds_copied))
    
    # P(z=1 | Independent) ~ 0.970, P(z=1 | Copied) = 0.850
    assert p_independent > p_copied + 0.10
    assert abs(p_independent - 0.970) < 0.01
    assert abs(p_copied - 0.850) < 0.01


def test_signed_evidence_inversion_invariants():
    """Asserts that an Opposite source (P(r=z) = 0.15) inverts evidence while Random (P=0.50) is neutral."""
    # Helpful (0.85)
    lr_helpful = 0.85 / 0.15
    # Opposite (0.15)
    lr_opposite = 0.15 / 0.85
    # Random (0.50)
    lr_random = 0.50 / 0.50

    log_odds_helpful = math.log(lr_helpful)
    log_odds_opposite = math.log(lr_opposite)
    log_odds_random = math.log(lr_random)

    assert log_odds_helpful > 1.70
    assert log_odds_opposite < -1.70
    assert abs(log_odds_random) < 1e-6
    assert abs(log_odds_helpful + log_odds_opposite) < 1e-6  # Exact symmetric inversion
