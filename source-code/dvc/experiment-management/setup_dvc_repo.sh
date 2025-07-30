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

echo "Setup complete. Created:"
echo "  - DVC data directory: $target/dvc_data"
echo "  - ML project directory: $target/ml_project"#!/usr/bin/env bash

cd "$target/ml_project"

# intialize the repository and do the first commit
git init
echo "__pycache__/" >> .gitignore
git add .gitignore
git commit -m 'Add .gitignore file'
git add src params.yaml requirements.txt
git commit -m 'Add initial project files'

# create the Python virtual environment and install the required packages
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# make sure that the virtual environment directory is not tracked by git
echo "env/" >> .gitignore
git add .gitignore
git commit -m 'Add Python venv to .gitignore file'

# initialize DVC and add remote storage
dvc init
git commit -m 'Initialize DVC'
dvc remote add -d local_storage ../dvc_data
git add .dvc/config
git commit -m 'Add remote storage for DVC'
dvc config core.autostage true
git add .dvc/config
git commit -m 'Turn DVC autostage on'
