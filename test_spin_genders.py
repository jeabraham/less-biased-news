"""
Test the spin_genders function by reading input from a file or stdin and configuration from config.yaml.
"""
import argparse
import sys
from pathlib import Path
import yaml
from ai_queries import spin_genders
from ai_utils import AIUtils


def main():
    # Define argument parser
    parser = argparse.ArgumentParser(description="Test the spin_genders function with input from a text file or stdin.")
    parser.add_argument(
        "filename", type=str, nargs="?",
        help="Path to the input text file to be processed. If not provided, stdin will be used."
    )

    # Parse arguments
    args = parser.parse_args()

    # Process input
    if args.filename:
        # Use filename if provided
        test_file = Path(args.filename)

        # Check if the file exists
        if not test_file.is_file():
            raise FileNotFoundError(f"The file '{args.filename}' does not exist.")

        # Load content from the test file
        with test_file.open("r", encoding="utf-8") as file:
            content = file.read()
    else:
        # Read from stdin if no filename is provided
        if sys.stdin.isatty():
            print("No filename provided and stdin is not piped. Exiting...", file=sys.stderr)
            sys.exit(1)
        content = sys.stdin.read()
        print("Reading from stdin...")

    # Load configuration from YAML file
    config_file = Path("config.yaml")
    if not config_file.exists():
        raise FileNotFoundError("The configuration file 'config.yaml' does not exist.")

    with config_file.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    # Initialize AIUtils instance (customize to match your implementation)
    ai_util = AIUtils(cfg)

    # Call spin_genders function with content and config
    result = spin_genders(content, cfg, ai_util)

    # Output the result
    print(result)


if __name__ == "__main__":
    main()
