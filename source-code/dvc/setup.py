#!/usr/bin/env python3

# Python script to set up
#  - a directory for storing DVC data (to act as a remote)
#  - a directory for a local git repository
# This script takes the path to the directory where these
# directories should be created as an argument.  Before
# creating the directories, it will check whether the
# given directory is not in use by another git repository.
# It will than take the following steps:
#  - create a directory for DVC data
#  - create a directory for the local git repository
#  - copy the `requirements.txt` file into the git repository
#  - copy the `params.yaml` file into the git repository
#  - copy the `src` directory into the git repository
#  - copy the `datat` directory into the git repository

import argparse
import pathlib
import shutil
import sys

def check_git_repo(path):
    """Check if the given path or any directory above it is a git repository."""
    git_dir = path / '.git'
    if git_dir.exists():
        return True
    parent = path.parent
    while parent != parent.parent:  # Stop at the root directory
        git_dir = parent / '.git'
        if git_dir.exists():
            return True
        parent = parent.parent
    return False

def main():
    parser = argparse.ArgumentParser(description="Set up directories for DVC and local git repository.")
    parser.add_argument("path", type=pathlib.Path,
                        help="Path to the directory where the setup should be done.")
    args = parser.parse_args()

    path = args.path.resolve()
    
    if check_git_repo(path):
        print(f"Error: The directory {path} is already part of a git repository.", file=sys.stderr)
        return 1

    dvc_dir = path / 'dvc_data'
    git_dir = path / 'ml_project'

    dvc_dir.mkdir(parents=True, exist_ok=True)
    git_dir.mkdir(parents=True, exist_ok=True)

    # Copy necessary files into the git repository
    shutil.copy('requirements.txt', git_dir / 'requirements.txt')
    shutil.copy('params.yaml', git_dir / 'params.yaml')
    shutil.copytree('src', git_dir / 'src', dirs_exist_ok=True)
    shutil.copytree('data', git_dir / 'data', dirs_exist_ok=True)

if __name__ == "__main__":
    sys.exit(main())
