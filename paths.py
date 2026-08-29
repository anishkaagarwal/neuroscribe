"""
Shared path resolution for NeuroscribeAI.

Keeps the codebase portable: no machine-specific absolute paths.
Override the recordings location with the NEUROSCRIBE_RECORDINGS_DIR env var.
"""

import os
from pathlib import Path

# Directory that contains this file (the project root).
PROJECT_ROOT = Path(__file__).resolve().parent


def recordings_dir() -> Path:
    """Return the recordings/transcripts directory, creating it if needed."""
    override = os.getenv("NEUROSCRIBE_RECORDINGS_DIR")
    path = Path(override) if override else PROJECT_ROOT / "recordings"
    path.mkdir(parents=True, exist_ok=True)
    return path
