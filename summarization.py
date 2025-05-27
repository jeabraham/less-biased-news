import sys
import argparse
from transformers import pipeline


def chunk_text(text, max_chunk_size):
    """
    Split text into smaller chunks, each within the model's maximum input length.
    """
    sentences = text.split('. ')  # Split text into sentences for natural boundaries
    chunks = []
    current_chunk = []

    for sentence in sentences:
        if len(' '.join(current_chunk + [sentence])) <= max_chunk_size:
            current_chunk.append(sentence)  # Add sentence to current chunk
        else:
            chunks.append(' '.join(current_chunk))  # Save current chunk
            current_chunk = [sentence]  # Start a new chunk

    if current_chunk:
        chunks.append(' '.join(current_chunk))  # Add any remaining sentences as the last chunk

    return chunks


def summarize_text_using_local_model(input_text, model="facebook/bart-large-cnn", max_length=130, min_length=30,
                                     strict=False):
    """
    Summarize the input text using the specified model and summarization settings.
    Intelligently adjusts summary length based on input length.

    Args:
        input_text (str): The full text to summarize.
        model (str): The Hugging Face model name for summarization (e.g., "t5-small" or "facebook/bart-large-cnn").
        max_length (int): The maximum length of the generated summary (in tokens).
        min_length (int): The minimum length of the generated summary (in tokens).
        strict (bool): If True, will recursively summarize until output is within max_length.

    Returns:
        str: The summarized text.
    """
    try:
        # Load the summarization pipeline
        summarizer = pipeline("summarization", model=model)

        # Define maximum input chunk size based on model constraints
        max_chunk_size = 512 if model.startswith("t5") else 1024  # T5 supports 512 tokens, BART supports 1024 tokens

        # Split input text into smaller chunks
        chunks = chunk_text(input_text, max_chunk_size)

        # Estimate input token count (rough approximation: ~4 chars per token)
        input_token_count = len(input_text) // 4

        # Dynamically adjust max_length based on input length
        # For very short inputs, we want an even shorter summary
        adjusted_max_length = max_length
        if input_token_count < max_length:
            # If input is shorter than the default max_length, make summary even shorter
            # Using a ratio of approximately 1/2 for summarization
            adjusted_max_length = max(min_length, input_token_count // 2)
            logger.debug(f"Input length ({input_token_count} tokens) is less than max_length ({max_length}). "
                         f"Adjusted max_length to {adjusted_max_length} tokens.")

        # Ensure min_length is always less than or equal to adjusted_max_length
        adjusted_min_length = min(min_length, adjusted_max_length - 10)

        # Summarize each chunk and combine the results
        summarized_chunks = []
        for chunk in chunks:
            summary = summarizer(
                chunk,
                max_length=adjusted_max_length,  # Using adjusted maximum output length
                min_length=adjusted_min_length,  # Using adjusted minimum output length
                do_sample=False  # Consistent, deterministic output
            )
            summarized_chunks.append(summary[0]["summary_text"])

        # Combine summarized chunks into a single output
        final_summary = " ".join(summarized_chunks)

        # If strict mode is enabled and the final summary is still too long, recurse
        if len(final_summary) > max_length * 4 and strict:  # Using char count approximation
            logger.debug(f"Summary still too long ({len(final_summary) // 4} tokens). Recursively summarizing.")
            final_summary = summarize_text_using_local_model(
                final_summary,
                model=model,
                max_length=max_length,
                min_length=min_length,
                strict=strict
            )

        return final_summary
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        # Return a truncated version of the original text as fallback
        return input_text[:max_length * 4] + "..." if len(input_text) > max_length * 4 else input_text



def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Summarize text using T5 or BART.")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/bart-large-cnn",
        choices=["t5-small", "facebook/bart-large-cnn"],
        help="Specify the model to use for summarization. Default is 'facebook/bart-large-cnn'."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=130,  # Default maximum output length
        help="Specify the maximum number of tokens for the summary."
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=30,  # Default minimum output length
        help="Specify the minimum number of tokens for the summary."
    )
    args = parser.parse_args()

    # Read input text from stdin
    input_text = sys.stdin.read().strip()
    if not input_text:
        print("Error: No input provided.", file=sys.stderr)
        sys.exit(1)

    try:
        # Generate the summary
        final_summary = summarize_text_using_local_model(
            input_text,
            model=args.model,
            max_length=args.max_length,
            min_length=args.min_length
        )
        print(final_summary)  # Output the summary to stdout

    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
