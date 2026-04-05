#!/bin/bash
# Removes all registered git submodules, then places their content
# into the vendors/ directory as plain tracked directories.
#
# Compatible with bash 3.2+ (macOS default shell).
#
# Usage: ./script/migrate_to_vendors.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -f .gitmodules ]; then
    echo "No .gitmodules file found. Nothing to do."
    exit 0
fi

VENDORS_DIR="vendors"
mkdir -p "$VENDORS_DIR"

# Capture all submodule paths and their URLs into two parallel arrays
# BEFORE any removal, since git rm modifies .gitmodules mid-loop.
SUBMODULE_PATHS=()
SUBMODULE_URLS=()

while IFS=' ' read -r _key path; do
    url=$(git config --file .gitmodules --get "submodule.${path}.url")
    SUBMODULE_PATHS+=("$path")
    SUBMODULE_URLS+=("$url")
done < <(git config --file .gitmodules --get-regexp 'submodule\..*\.path')

if [ ${#SUBMODULE_PATHS[@]} -eq 0 ]; then
    echo "No submodules found. Nothing to do."
    exit 0
fi

for i in "${!SUBMODULE_PATHS[@]}"; do
    SUBMODULE_PATH="${SUBMODULE_PATHS[$i]}"
    SUBMODULE_URL="${SUBMODULE_URLS[$i]}"
    BASENAME=$(basename "$SUBMODULE_PATH")
    VENDOR_TARGET="$VENDORS_DIR/$BASENAME"

    echo "--------------------------------------------------"
    echo "Migrating: $SUBMODULE_PATH -> $VENDOR_TARGET"
    echo "Source URL: $SUBMODULE_URL"

    # Step 1: Deinit — unregisters from .git/config and clears working tree binding
    git submodule deinit -f "$SUBMODULE_PATH"

    # Step 2: Remove from the index and strip the entry from .gitmodules
    git rm -f "$SUBMODULE_PATH"

    # Step 3: Remove cached submodule metadata
    rm -rf ".git/modules/$SUBMODULE_PATH"

    # Step 4: Clone fresh into vendors/ (git rm deletes the directory)
    echo "Cloning $SUBMODULE_URL ..."
    git clone --depth=1 "$SUBMODULE_URL" "$VENDOR_TARGET"

    # Step 5: Strip the nested .git so it becomes a plain directory in this repo
    rm -rf "$VENDOR_TARGET/.git"

    echo "Done: $BASENAME is now at $VENDOR_TARGET"
done

echo "--------------------------------------------------"
echo "Staging vendors/ ..."
git add "$VENDORS_DIR"

# Remove .gitmodules if it is now empty
if [ -f .gitmodules ] && [ ! -s .gitmodules ]; then
    rm -f .gitmodules
    git rm --cached .gitmodules 2>/dev/null || true
fi

echo ""
echo "Migration complete. Review staged changes with 'git status', then commit:"
echo "  git commit -m 'chore: replace submodules with plain directories under vendors/'"
