"""Auto-update checker using GitHub Releases API.

Checks for newer versions on startup and presents an update notification.
Fails silently on network errors so offline users are never blocked.
"""

from __future__ import annotations

import json
import threading
import urllib.request
import urllib.error
import webbrowser
from importlib.metadata import version, PackageNotFoundError

# Configure with your GitHub repository (owner/repo format).
GITHUB_REPO = "TimWiesner99/edl-to-archive"

# Timeout for the API request (seconds).
CHECK_TIMEOUT = 3


def get_current_version() -> str:
    """Get the current application version from package metadata."""
    try:
        return version("edl-to-archive")
    except PackageNotFoundError:
        return "0.1"


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a version string like '1.2.3' or 'v1.2.3' into a tuple of ints."""
    v = v.strip().lstrip("v")
    parts = []
    for segment in v.split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def check_for_update() -> dict | None:
    """Check GitHub Releases for a newer version.

    Returns:
        A dict with keys 'version', 'url', 'body' if a newer release exists,
        or None if up-to-date or on error.
    """
    current = get_current_version()
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

    try:
        req = urllib.request.Request(
            api_url,
            headers={"Accept": "application/vnd.github.v3+json"},
        )
        with urllib.request.urlopen(req, timeout=CHECK_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None

    tag = data.get("tag_name", "")
    if not tag:
        return None

    if _parse_version(tag) > _parse_version(current):
        return {
            "version": tag.lstrip("v"),
            "url": data.get("html_url", ""),
            "body": data.get("body", ""),
        }

    return None


def check_for_update_async(callback) -> None:
    """Run the update check in a background thread.

    Args:
        callback: Called on the main thread with the result dict (or None).
                  The caller is responsible for scheduling the callback on
                  the GUI thread if needed.
    """

    def _worker():
        result = check_for_update()
        callback(result)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
