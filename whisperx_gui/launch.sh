#!/bin/bash
# Simple launcher script for WhisperX GUI
# Run this from anywhere to launch the GUI

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WHISPERX_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if exists
if [ -f "$WHISPERX_DIR/.venv/bin/activate" ]; then
    source "$WHISPERX_DIR/.venv/bin/activate"
fi

# Run the GUI
python "$SCRIPT_DIR/run.py" "$@"
