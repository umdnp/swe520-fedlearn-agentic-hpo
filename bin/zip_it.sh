#!/bin/bash

set -e

PROJECT_DIR=~/git
OUTPUT_NAME="agentic_hpo.zip"

echo "Creating zip archive using 7z ..."

# remove existing zip if it exists
if [ -f "$OUTPUT_NAME" ]; then
  rm "$OUTPUT_NAME"
fi

# create zip excluding data directory and binary files
7z a -tzip "$OUTPUT_NAME" "$PROJECT_DIR" \
  -xr!__pycache__ \
  -xr!.git \
  -xr!.idea \
  -xr!data \
  -xr!*.pkl \
  -xr!*.zip

echo "Done."
