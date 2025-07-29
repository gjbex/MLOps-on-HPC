#!/usr/bin/env bash

set -euo pipefail

# intialize the repository and do the first commit
git init
git add src params.yaml requirements.txt
git commit -m 'Initial commit'

# create the Python virtual environment and install the required packages
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# make sure that the virtual environment directory is not tracked by git
echo "env/" >> .gitignore
git add .gitignore
git commit -m 'Add .gitignore file'

# initialize DVC and add remote storage
dvc init
git commit -m 'Initialize DVC'
dvc remote add -d local_storage ../dvc_data
git add .dvc/config
git commit -m 'Add remote storage for DVC'
