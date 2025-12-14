#!/bin/bash
# transcribe.sh

export TORCH_LOAD_WEIGHTS_ONLY=0

uvx whisperx "$1" \
  --model tiny \
  --device cpu \
  --compute_type int8 \
  --language en \
  --output_format all

# Usage: ./transcribe.sh audio.mp3
