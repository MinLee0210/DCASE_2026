#!/bin/bash

# Default values
CONFIG="config/config.yml"
MODEL_PATH=""
SPLIT="val"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG="$2"; shift ;;
        -m|--model_path) MODEL_PATH="$2"; shift ;;
        -s|--split) SPLIT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model_path (-m) is required."
    echo "Usage: ./scripts/evaluate.sh -c <config> -m <model_path> [-s val|test]"
    exit 1
fi

echo "Starting evaluation..."
echo "  Config:     ${CONFIG}"
echo "  Model:      ${MODEL_PATH}"
echo "  Split:      ${SPLIT}"

# Activate the virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated."
else
    echo "Warning: .venv not found. Proceeding with global python."
fi

# Run the evaluation pipeline
python -m src evaluate \
    --config "${CONFIG}" \
    --model_path "${MODEL_PATH}" \
    --split "${SPLIT}"
