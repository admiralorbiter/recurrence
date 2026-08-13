"""Pytest configuration for path resolution."""

import sys
from pathlib import Path

# Add src and repository root to sys.path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
