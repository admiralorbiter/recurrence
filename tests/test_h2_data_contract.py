"""Automated evidence contract and synchronization tests for Horizon 2."""

import json
from pathlib import Path
import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_h2_manifest_status(repo_root: Path) -> None:
    manifest_path = repo_root / "h2" / "MANIFEST.json"
    assert manifest_path.exists(), "h2/MANIFEST.json must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Core sprints S10 through S14 must all be FROZEN
    for sprint in ["S10", "S11b", "S12b", "S12c", "S13", "S14"]:
        assert manifest["evidence_status"].get(sprint) == "FROZEN", f"{sprint} must be marked FROZEN"

    # Canonical sources must all exist
    for src in manifest["canonical_sources"]:
        src_path = repo_root / src
        assert src_path.exists(), f"Canonical source {src} referenced in MANIFEST.json must exist"


def test_h2_core_json_contract(repo_root: Path) -> None:
    core_path = repo_root / "h2" / "data" / "core.json"
    assert core_path.exists(), "h2/data/core.json must exist"

    with open(core_path, "r", encoding="utf-8") as f:
        core = json.load(f)

    # Check S10-S14 presence
    assert core["meta"]["frozen_core"] == ["S10", "S11b", "S12b", "S12c", "S13", "S14"]

    # Check S11 timeline values
    timeline = core["s11"]["timeline"]
    n_map = {item["n"]: item for item in timeline}
    assert n_map[0]["r"] == 1.0
    assert n_map[256]["r"] == 0.234
    assert n_map[1024]["r"] == 0.194
    assert n_map[2048]["r"] == 0.285
    assert n_map[4096]["r"] == 0.3384

    # Check S12b causal swaps vs S12c specificity
    s12b_match = next(c for c in core["s12b"]["conditions"] if c["id"] == "match")
    assert round(s12b_match["estimate"], 2) == 74.10
    assert round(core["s12c"]["delta_p_value_spec"], 2) == 38.49

    # Check S14 results
    assert core["s14"]["tost"]["matched"]["p"] < 0.005
    assert len(core["ladder"]) == 7, "The canonical ladder must have exactly 7 dissociations"


def test_h2_site_data_sync(repo_root: Path) -> None:
    data_js_path = repo_root / "h2" / "site" / "data.js"
    assert data_js_path.exists(), "h2/site/data.js must exist"

    with open(data_js_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    assert content.startswith("window.H2 ="), "site/data.js must start with 'window.H2 ='"
    json_str = content[len("window.H2 ="):].rstrip(";")
    site_data = json.loads(json_str)

    # Check that site_data has all required fields
    assert "timeline" in site_data
    assert "transplant" in site_data
    assert "specificity" in site_data
    assert "dynamics" in site_data
    assert "strictC" in site_data
    assert "tost" in site_data
    assert "tiers" in site_data
    assert "ladder" in site_data
    assert len(site_data["ladder"]) == 7
