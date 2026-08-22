import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent

assets = [
    "crates/continuity_garden_core/src/bin/confirmatory_q17e_contraction_algebra.rs",
    "crates/continuity_garden_core/Cargo.toml",
    "research/contracts/CONTRACT-E-Q17E.md",
]

manifest = {
    "contract_id": "CONTRACT-E-Q17E",
    "base_sha": "75afd691996cb4a77eeed6b5f4361852239e48ae",
    "execution_base_sha": "17fee7ec3a51f319cdb1d4bffbd19eaef322f155",
    "assets": {},
}

for rel in assets:
    p = root / rel
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest["assets"][rel] = digest

out = root / "research" / "contracts" / "SEALING_MANIFEST-E-Q17E.json"
out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Generated {out}")
print(json.dumps(manifest, indent=2))
