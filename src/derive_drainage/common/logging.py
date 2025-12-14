"""
Logging configuration for derive_drainage.
"""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(output_dir: Path) -> None:
    """
    Configure console and file logging to processing.log in the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "processing.log"

    for handler in list(logging.root.handlers):
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

