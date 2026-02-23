#!/usr/bin/env bash
# ── EDL-to-Archive Launcher (macOS / Linux) ──────────────────────────
# This script checks for uv (Python package manager), installs it if
# needed, then launches the application. uv automatically handles
# downloading the correct Python version and installing dependencies.
# ─────────────────────────────────────────────────────────────────────

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    # Check common install location
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "uv is not installed. Installing uv..."
        echo ""

        if command -v curl &> /dev/null; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
        elif command -v wget &> /dev/null; then
            wget -qO- https://astral.sh/uv/install.sh | sh
        else
            echo "Error: curl or wget is required to install uv."
            echo "Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
            exit 1
        fi

        if [ $? -ne 0 ]; then
            echo ""
            echo "Failed to install uv automatically."
            echo "Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
            exit 1
        fi

        export PATH="$HOME/.local/bin:$PATH"
        echo ""
        echo "uv installed successfully."
        echo ""
    fi
fi

# Run the GUI application
echo "Starting EDL-to-Archive..."
echo "(first launch may take a moment while dependencies are installed)"
echo ""
exec uv run gui.py
