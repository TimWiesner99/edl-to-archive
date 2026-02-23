"""Version checking and git-based update mechanism.

Checks the version in pyproject.toml on the main branch and pulls updates
via git pull. Fails silently on network errors.
"""

from __future__ import annotations

import re
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

GITHUB_REPO = "TimWiesner99/edl-to-archive"
CHECK_TIMEOUT = 5  # seconds

# Project root is the parent of the src/ directory where this file lives
_PROJECT_ROOT = Path(__file__).parent.parent


def get_current_version() -> str:
    """Read the version from the local pyproject.toml."""
    try:
        content = (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        if match:
            return match.group(1)
    except OSError:
        pass
    return "0.1"


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse '1.2.3' or 'v1.2.3' into a comparable tuple of ints."""
    parts = []
    for segment in v.strip().lstrip("v").split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _get_remote_version() -> str | None:
    """Fetch the version from pyproject.toml on the main branch.

    Returns the version string, or None on any error.
    """
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/pyproject.toml"
    try:
        with urllib.request.urlopen(url, timeout=CHECK_TIMEOUT) as resp:
            content = resp.read().decode("utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        if match:
            return match.group(1)
    except (urllib.error.URLError, OSError, TimeoutError):
        pass
    return None


def check_for_update() -> bool:
    """Return True if the main branch has a higher version than the local copy."""
    remote = _get_remote_version()
    if remote is None:
        return False
    return _parse_version(remote) > _parse_version(get_current_version())


def check_for_update_async(callback) -> None:
    """Run the version check in a background thread.

    Args:
        callback: Called with True if an update is available, False otherwise.
    """
    def _worker():
        result = check_for_update()
        callback(result)

    threading.Thread(target=_worker, daemon=True).start()


def pull_latest() -> tuple[bool, str]:
    """Run 'git pull origin main' from the project root.

    Returns:
        Tuple of (success: bool, output: str).
    """
    try:
        result = subprocess.run(
            ["git", "pull", "origin", "main"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output
    except FileNotFoundError:
        return False, (
            "git was not found on PATH.\n"
            "Please ensure Git is installed and available in your terminal."
        )
    except subprocess.TimeoutExpired:
        return False, "git pull timed out after 60 seconds."
    except OSError as e:
        return False, f"Failed to run git: {e}"
