#!/usr/bin/env python3
"""Sync h2/data/core.json into h2/site/data.js."""

import json
from pathlib import Path


def sync_h2_data() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    core_json_path = repo_root / "h2" / "data" / "core.json"
    data_js_path = repo_root / "h2" / "site" / "data.js"

    with open(core_json_path, "r", encoding="utf-8") as f:
        core = json.load(f)

    # Build the website schema expected by site/app.js
    site_data = {
        "timeline": core["s11"]["timeline"],
        "transplant": core["s12b"]["conditions"],
        "specificity": {
            "value": core["s12c"]["delta_p_value_spec"],
            "ci": core["s12c"]["delta_p_value_spec_ci"],
            "projection": core["s12c"]["delta_proj_value_spec"],
            "projectionCI": core["s12c"]["delta_proj_value_spec_ci"],
        },
        "dynamics": core["s13"]["dynamics"],
        "strictC": core["s14"]["strictC"],
        "tost": core["s14"]["tost"],
        "tiers": core["s14"]["tiers"],
        "ladder": core["ladder"],
    }

    js_content = f"window.H2 = {json.dumps(site_data, indent=2)};\n"

    with open(data_js_path, "w", encoding="utf-8") as f:
        f.write(js_content)

    print(f"[sync_h2_data] Successfully synchronized {core_json_path} -> {data_js_path}")


if __name__ == "__main__":
    sync_h2_data()
