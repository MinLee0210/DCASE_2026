#!/bin/bash

# Default values
CONFIG="config/config.yml"
RESUME=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG="$2"; shift ;;
        -r|--resume) RESUME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Starting training..."
echo "  Config:     ${CONFIG}"
if [[ -n "$RESUME" ]]; then
    echo "  Resume:     ${RESUME}"
fi

# Activate the virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated."
else
    echo "Warning: .venv not found. Proceeding with global python."
fi

# Build the command, conditionally appending --resume
CMD="python -m src train --config ${CONFIG}"
if [[ -n "$RESUME" ]]; then
    CMD="${CMD} --resume ${RESUME}"
fi

# Run the training pipeline
eval $CMD
