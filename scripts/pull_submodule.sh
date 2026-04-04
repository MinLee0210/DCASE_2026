#!/bin/bash
# Script to initialize and pull the latest code for git submodules

# Exit on any error
set -e

# Navigate to the project root directory (assuming script is in the script/ directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Fetching latest changes from the repository..."
git pull

echo "Initializing and updating submodules to their registered commits..."
git submodule update --init --recursive

echo "Pulling the latest updates from the remote tracking branches of the submodules..."
git submodule update --remote --merge

echo "Submodule pull complete!"
