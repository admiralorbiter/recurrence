"""Test repository state and documentation synchronization."""

import json
import re
from pathlib import Path
import pytest


def test_core_json_frozen_status():
    repo_root = Path(__file__).resolve().parent.parent
    core_json_path = repo_root / "h2" / "data" / "core.json"
    assert core_json_path.exists(), "h2/data/core.json must exist"

    with open(core_json_path, "r", encoding="utf-8") as f:
        core_data = json.load(f)

    meta = core_data.get("meta", {})
    assert "S14" in meta.get("frozen_core", []), "S14 must be in frozen_core"
    assert core_data.get("s14", {}).get("status") == "FROZEN", "s14 status must be FROZEN"
    assert meta.get("status") == "FROZEN", "Horizon 2 meta status must be FROZEN"


def test_site_data_js_matches_core_json():
    repo_root = Path(__file__).resolve().parent.parent
    core_json_path = repo_root / "h2" / "data" / "core.json"
    data_js_path = repo_root / "h2" / "site" / "data.js"

    assert data_js_path.exists(), "h2/site/data.js must exist"

    with open(core_json_path, "r", encoding="utf-8") as f:
        core_data = json.load(f)

    with open(data_js_path, "r", encoding="utf-8") as f:
        js_text = f.read()

    # Extract JSON from window.H2 = { ... };
    match = re.search(r"window\.H2\s*=\s*(\{.*\})\s*;\s*$", js_text, re.DOTALL)
    assert match is not None, "h2/site/data.js must assign window.H2 = { ... };"

    site_data = json.loads(match.group(1))
    assert site_data["timeline"] == core_data["s11"]["timeline"]
    assert site_data["transplant"] == core_data["s12b"]["conditions"]
    assert site_data["dynamics"] == core_data["s13"]["dynamics"]
    assert site_data["strictC"] == core_data["s14"]["strictC"]


def test_h2_readme_consistency():
    repo_root = Path(__file__).resolve().parent.parent
    h2_readme_path = repo_root / "h2" / "README.md"
    assert h2_readme_path.exists(), "h2/README.md must exist"

    with open(h2_readme_path, "r", encoding="utf-8") as f:
        readme_text = f.read()

    assert "| S12c |" in readme_text and "FROZEN" in readme_text, "S12c must be marked FROZEN in h2/README.md"
    assert "| S13 |" in readme_text and "FROZEN" in readme_text, "S13 must be marked FROZEN in h2/README.md"
    assert "| S14 |" in readme_text and "FROZEN" in readme_text, "S14 must be marked FROZEN in h2/README.md"


def test_manifest_pinned_model_revision():
    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = repo_root / "docs" / "h2_environment_manifest.json"
    freeze_manifest_path = repo_root / "docs" / "H2_Core_Freeze_Manifest.md"

    expected_rev = "3620f4ca9c5d16ee56c00180474a3201ec7f734a"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert manifest_data.get("model_substrate", {}).get("pinned_revision") == expected_rev

    with open(freeze_manifest_path, "r", encoding="utf-8") as f:
        freeze_text = f.read()
    assert expected_rev in freeze_text, f"Expected revision {expected_rev} must appear in H2_Core_Freeze_Manifest.md"
