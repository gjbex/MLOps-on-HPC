#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <target_directory>" >&2
    exit 1
fi

check_git_repo() {
    local dir="$1"
    while [[ "$dir" != "/" && -n "$dir" ]]; do
        if [[ -d "$dir/.git" ]]; then
            return 0
        fi
        dir=$(dirname "$dir")
    done
    return 1
}

target="$1"
if realpath_out=$(realpath "$target" 2>/dev/null); then
    target="$realpath_out"
fi

if check_git_repo "$target"; then
    echo "Error: the directory '$target' is already part of a Git repository." >&2
    exit 1
fi

mkdir -p "$target/dvc_data" "$target/ml_project"

cp requirements.txt "$target/ml_project/requirements.txt"
cp params.yaml "$target/ml_project/params.yaml"
cp -R src "$target/ml_project/src"
cp -R data "$target/ml_project/data"

echo "Setup complete. Created:"
echo "  - DVC data directory: $target/dvc_data"
echo "  - ML project directory: $target/ml_project"