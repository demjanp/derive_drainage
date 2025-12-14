"""
Metadata helpers for derive_drainage.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_metadata(output_dir: Path, metadata: dict) -> None:
    """
    Write run metadata as JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "run_metadata.json"
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

