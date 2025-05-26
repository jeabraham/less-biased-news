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

OUTEMAIL="news_$(date +%Y-%m-%d)_email.html"

# Build the email file
python news_formatter.py --format email --output "output/$OUTEMAIL" > "output/$OUTEMAIL"

# Send the email
python send_email.py --to "john@theabrahams.ca" --subject "News for $(date +%Y-%m-%d)" --body "output/$OUTEMAIL" --log INFO
