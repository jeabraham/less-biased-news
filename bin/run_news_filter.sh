#!/usr/bin/env bash

# Change to your project dir
cd /Users/jabraham/Development/less-biased-news || exit 1

# Activate your venv
source ./venv/bin/activate

# Build an output filename with today's date
OUT="news_$(date +%Y-%m-%d).html"

# Run the script
python news_filter.py --format html --output "output/$OUT" --new-today --log INFO

# Open in default browser
open "output/$OUT"
