#!/bin/bash

# Default config path
CONFIG="config/config.yml"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Starting training with config: ${CONFIG}"

# Activate the virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated."
else
    echo "Warning: .venv not found. Proceeding with global python."
fi

# Run the training pipeline
# Note: You can prepend CUDA_VISIBLE_DEVICES=0 if you have multiple GPUs
python -m src train --config "${CONFIG}"
