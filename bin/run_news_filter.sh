#!/usr/bin/env bash

# Change to your project dir
cd /Users/jabraham/Development/less-biased-news || exit 1

# Activate your venv
source ./venv/bin/activate

# Build an output filename with today's date
OUT="news_$(date +%Y-%m-%d).html"

# Run the script
# python news_filter.py --format html --output "output/$OUT" --new-today --log INFO

# Open in default browser
open "output/$OUT"

# OUTEMAIL="news_$(date +%Y-%m-%d)_email.html"

# Build the email file
# python news_formatter.py --format email  > "output/$OUTEMAIL"

rsync -a --delete "output/" jea@fw-1.office.hbaspecto.com:/var/www/other-web/www-deliriumspb-com/html/johns_playground/

# Create a text file with cat:
cat <<EOF > simple_email.txt
You can view your news for today here:

https://www.deliriumspb.com/johns_playground/$(basename "$OUT")

EOF

osascript send_email/send_email.applescript john@theabrahams.ca "News Today $OUT" simple_email.txt john@theabrahams.ca True
