#!/bin/bash
set -e
echo "=== BrainFlowHyper: Generative Flow Matching ==="

echo "[1/3] Training model..."
python -m src.model --step train

echo "[2/3] Generating samples..."
python -m src.model --step generate

echo "[3/3] Evaluating..."
python -m src.evaluate

echo "=== Done ==="
