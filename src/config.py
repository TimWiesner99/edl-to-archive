"""Application configuration and data directory management.

Handles persistent settings and template file management.
All data is stored within the project directory.
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

# Project root is the parent of the src/ directory where this file lives.
# This works regardless of the current working directory.
_PROJECT_ROOT = Path(__file__).parent.parent

# Template headers (comma-delimited CSV format)
EDL_TEMPLATE_HEADER = "ID,Reel,Name,File Name,Track,Timecode In,Timecode Out,Duration,Source Start,Source End,Audio Channels,Comment\n"
SOURCE_TEMPLATE_HEADER = "TC in,Duur,Bestandsnaam,Omschrijving,Link,Bron,kosten,rechten / contact,to do,Bron in beeld,Aftiteling\n"

DEFAULT_CONFIG = {
    "exclusion_rules": "",
    "fps": 25,
    "delimiter": "comma",
    "collapse": True,
    "frames": False,
    "output_path": "",
    "skipped_version": "",
}


def get_app_data_dir() -> Path:
    """Get the data directory for config storage (data/ inside the project).

    Returns:
        Path to the data directory (created if it doesn't exist).
    """
    data_dir = _PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_output_dir() -> Path:
    """Get the default output directory (output/ inside the project).

    Returns:
        Path to the output directory (created if it doesn't exist).
    """
    output_dir = _PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _config_path() -> Path:
    return get_app_data_dir() / "config.json"


def load_config() -> dict:
    """Load configuration from disk, falling back to defaults for missing keys."""
    config = dict(DEFAULT_CONFIG)
    path = _config_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                stored = json.load(f)
            config.update(stored)
        except (json.JSONDecodeError, OSError):
            pass
    return config


def save_config(config: dict) -> None:
    """Save configuration to disk."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_edl_path() -> Path:
    """Get the path for the EDL file (input/ inside the project)."""
    return _PROJECT_ROOT / "input" / "EDL.csv"


def get_source_path() -> Path:
    """Get the path for the Source file (input/ inside the project)."""
    return _PROJECT_ROOT / "input" / "SOURCE.csv"


def ensure_template_files() -> tuple[bool, bool]:
    """Create template EDL and Source CSV files if they don't exist.

    Returns:
        Tuple of (edl_created, source_created) booleans.
    """
    edl_created = _create_template_if_missing(get_edl_path(), EDL_TEMPLATE_HEADER)
    source_created = _create_template_if_missing(get_source_path(), SOURCE_TEMPLATE_HEADER)
    return edl_created, source_created


def reset_template_file(path: Path) -> None:
    """Overwrite a file with a fresh template header."""
    if path == get_edl_path():
        header = EDL_TEMPLATE_HEADER
    elif path == get_source_path():
        header = SOURCE_TEMPLATE_HEADER
    else:
        raise ValueError(f"Unknown template path: {path}")
    path.write_text(header, encoding="utf-8")


def open_file_in_default_app(filepath: Path) -> bool:
    """Open a file in the OS default application.

    Returns:
        True if the file was opened successfully, False otherwise.
    """
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", str(filepath)])
        elif system == "Windows":
            subprocess.Popen(["start", "", str(filepath)], shell=True)
        else:
            subprocess.Popen(["xdg-open", str(filepath)])
        return True
    except (OSError, FileNotFoundError):
        return False


def _create_template_if_missing(path: Path, header: str) -> bool:
    """Create a template file if it doesn't exist. Returns True if created."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header, encoding="utf-8")
    return True
