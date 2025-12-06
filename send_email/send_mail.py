import smtplib
import argparse
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import base64
from email import policy
from email.parser import BytesParser
from io import BytesIO  # Correct type for binary input

import traceback


def preview_email_content(raw_message):
    # Convert raw message to bytes since BytesParser expects a bytes-like object
    raw_message_bytes = raw_message.encode("utf-8")

    # Parse the raw email content to inspect parts
    message = BytesParser(policy=policy.default).parse(BytesIO(raw_message_bytes))
    for part in message.iter_parts():  # Iterate over MIME parts
        if part.get_content_type() == "text/html":
            # Decode the Base64-encoded body if needed
            html_content = part.get_payload(decode=True).decode("utf-8")
            print("Preview of Decoded HTML Content:")
            print(html_content[:2000])  # Print the first 1000 characters of actual HTML
            break
    else:
        print("No text/html part found in the email.")


# Function to send an HTML email
def send_html_email(sender, recipient, subject, html_file, smtp_server, smtp_port, login, password):
    try:
        # Read the HTML content from the specified file
        if not os.path.exists(html_file):
            raise FileNotFoundError(f"HTML file '{html_file}' does not exist.")

        with open(html_file, "r", encoding="utf-8") as file:
            html_content = file.read()
            if not html_content.strip():
                raise ValueError("The HTML file is empty or contains no content.")

        print("HTML Content Preview:")
        print(html_content[:1000])  # Print the first 500 characters of the HTML

        # Create the email message
        message = MIMEMultipart("alternative")
        message["From"] = sender
        message["To"] = recipient
        message["Subject"] = subject
        message["Bcc"] = sender  # Add sender to BCC for tracking
        message["Return-Receipt-To"] = sender

        # Attach the HTML content
        html_part = MIMEText(html_content, "html")
        message.attach(html_part)

        raw_message = message.as_string()
        print("Preview of Raw Email Content (First 1000 Characters):")
        print(raw_message[:1000])  # Print the first 500 characters for debugging
        preview_email_content(raw_message)

        # Connect to the SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Upgrade connection to secure
            server.login(login, password)  # Login
            response = server.sendmail(sender, recipient, message.as_string())  # Send email
            print(f"SMTP Server Response: {response}")

        print(f"Email successfully sent to {recipient} with subject: '{subject}'.")

    except Exception as e:
        print(f"Error sending email: {e}")
        traceback.print_exc()


# Test function to simulate the process without sending email
def test_email_reading(html_file):
    try:
        # Ensure the HTML file can be opened and read
        if not os.path.exists(html_file):
            print(f"Test Failed: HTML file '{html_file}' does not exist.")
            return

        with open(html_file, "r", encoding="utf-8") as file:
            html_content = file.read()

        print("Test Passed: HTML file has been successfully loaded.")
        print(f"Sample Content Preview:\n{html_content[:500]}")  # Display the first 500 characters

    except Exception as e:
        print(f"Test Failed: Error opening the HTML file: {e}")


# Main entry point for the argument parser
def main():
    parser = argparse.ArgumentParser(description="Send an HTML email using Python.")
    parser.add_argument("--subject", required=True, help="The subject of the email.")
    parser.add_argument("--recipient", required=True, help="The recipient's email address.")
    parser.add_argument("--sender", required=True, help="The sender's email address.")
    parser.add_argument("--html-file", required=True, help="Path to the HTML file to be sent.")
    parser.add_argument("--smtp-server", required=True, help="SMTP server address.")
    parser.add_argument("--smtp-port", type=int, default=587, help="SMTP server port (default: 587).")
    parser.add_argument("--password", required=True, help="The sender's email account password.")
    parser.add_argument("--test", action="store_true", help="Test mode: Verify HTML content without sending email.")

    args = parser.parse_args()

    if args.test:
        # Test the HTML file loading without sending the email
        test_email_reading(args.html_file)
    else:
        # Send the email as per the input arguments
        send_html_email(
            sender=args.sender,
            recipient=args.recipient,
            subject=args.subject,
            html_file=args.html_file,
            smtp_server=args.smtp_server,
            smtp_port=args.smtp_port,
            login=args.sender,
            password=args.password
        )


if __name__ == "__main__":
    main()
