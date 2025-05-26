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
python send_mail.py --subject "Less biased news for $(date +%Y-%m-%d)" --recipient john@theabrahams.ca --sender deborah_abraham@icloud.com --html-file "output/$OUTEMAIL" --password zsyk-qxdb-uimw-hiei --smtp-server smtp.mail.me.com --smtp-port 587
