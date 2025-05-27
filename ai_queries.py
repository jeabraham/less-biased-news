# openai_utils.py
import logging
import string
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
import time
import random
import re
import logging

logger = logging.getLogger(__name__)
import tiktoken

from summarization import summarize_text_using_local_model

logger = logging.getLogger(__name__)

def truncate_for_openai(
    prompt: str,
    text: str,
    max_tokens: int,
    tokenizer
) -> str:
    """
    Truncate `text` so that token_count(prompt) + token_count(text) <= max_tokens.

    Args:
      prompt:     The part you always include (e.g. system+instruction).
      text:       The user-supplied body you need to fit in.
      max_tokens: Total tokens you can spend on prompt+text.
      tokenizer:  A tiktoken tokenizer for exact token counting.

    Returns:
      A version of `text` whose token length, when added to prompt tokens, <= max_tokens.
    """
    # 1) Tokenize prompt once
    prompt_tokens = tokenizer.encode(prompt)
    prompt_len = len(prompt_tokens)

    # 2) Early exit if prompt itself is too long
    if prompt_len >= max_tokens:
        logger.warning(
            f"Prompt is {prompt_len} tokens ≥ max_tokens={max_tokens}; truncating prompt!"
        )
        # Drop tokens from prompt itself—rare, but avoids infinite loop
        return tokenizer.decode(prompt_tokens[:max_tokens])

    # 3) Tokenize the text
    text_tokens = tokenizer.encode(text)
    total_len = prompt_len + len(text_tokens)

    # 4) If it fits, return original text
    if total_len <= max_tokens:
        return text

    # 5) Otherwise, truncate text_tokens to fit
    allowed = max_tokens - prompt_len
    logger.warning(
        f"Input too long ({total_len} tokens); truncating text to {allowed} tokens"
    )
    truncated = text_tokens[:allowed]
    # Decode back into a string (may cut mid-word)
    return tokenizer.decode(truncated)



def open_ai_call(prompt: str, text: str, return_tokens: int, cfg: dict, client, tokenizer: PreTrainedTokenizer, complex: bool = False) -> str:
    """
    Call the OpenAI API for generating a response based on the given prompt and text.
    Includes retry logic with exponential backoff for rate limit errors.
    """
    model = cfg["openai"].get("model", "gpt-3.5-turbo")
    if complex:
        model = cfg["openai"].get("complex_model", model)
    else:
        model = cfg["openai"].get("simple_model", model)
    max_toks = cfg["openai"]["max_tokens"]
    temp = cfg["openai"].get("temperature", 0)
    max_retries = 3  # Maximum number of retry attempts
    base_delay = 1.0  # Base delay in seconds before retrying

    safe_text = truncate_for_openai(prompt, text, max_tokens=max_toks - return_tokens, tokenizer=tokenizer)
    prompt_plus_text = prompt + "\n\n" + safe_text
    logger.debug("Running OpenAI call")

    for attempt in range(max_retries + 1):
        try:
            resp = client.ChatCompletion.create(
                model=model,
                temperature=temp,
                max_tokens=return_tokens,
                messages=[{"role": "user", "content": prompt_plus_text}]
            )
            answer = resp.choices[0].message.content.strip()
            logger.debug(f"OpenAI result: {answer}")
            return answer
        except Exception as e:
            error_message = str(e)

            # Check if this is a rate limit error
            if "rate_limit" in error_message.lower() and attempt < max_retries:
                # Extract wait time if available in the error message
                wait_time = None
                time_match = re.search(r"Please try again in (\d+\.\d+)s", error_message)
                if time_match:
                    wait_time = float(time_match.group(1))
                    # Add a small buffer to ensure we're past the rate limit window
                    wait_time += 0.5
                else:
                    # Exponential backoff with jitter
                    wait_time = base_delay * (2 ** attempt) * (0.5 + 0.5 * random.random())

                logger.warning(
                    f"OpenAI API rate limit exceeded. Retrying in {wait_time:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                # For non-rate limit errors or if we've exhausted retries
                logger.warning(f"OpenAI API error: {e}")
                return ""  # Return empty string after all retries failed


def run_local_call(
    prompt: str,
    text: str,
    return_tokens: int,
    cfg: dict,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str
) -> str:
    """
    Run a “prompt + text” completion on a local Hugging Face model.

    Args:
      prompt:    The instruction or system prompt.
      text:      The user text you want to include (already truncated to fit).
      return_tokens:  How many tokens you want the model to generate.
      cfg:       Your localai config (for temperature, sampling settings, etc).
      model:     The pre-loaded local model.
      tokenizer: The matching tokenizer.
      device:    'cpu', 'cuda' or 'mps'.

    Returns:
      The generated string.
    """
    # 1) Truncate the 'text' the same way you did for OpenAI:
    max_length = cfg["localai"].get("max_input_tokens", 2048) - return_tokens
    #safe_text = truncate_for_openai(prompt, text, max_tokens=available)

    # 2) Build the combined prompt
    prompt_plus_text = prompt + "\n\n" + text

    # 3) Tokenize and move inputs to the right device
    encoding = tokenizer(prompt_plus_text, return_tensors="pt", truncation=True, max_length= max_length)
    inputs = {k: t.to(device) for k, t in encoding.items()}

    # 4) Generate using max_new_tokens
    gen_kwargs = {
        "max_new_tokens": return_tokens,
        "temperature": cfg["localai"].get("temperature", 0.7),
        "do_sample": cfg["localai"].get("do_sample", True),
        "num_beams": 4,
        # ensure pad_token_id so generation can stop properly
        "pad_token_id": tokenizer.eos_token_id
    }

    outputs = model.generate(**inputs, **gen_kwargs)

    # 5) Decode and return
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def classify_leadership(text: str, cfg: dict, ai_util) -> bool:
    """
    Classify text to determine if it mentions a leader, optionally using a local AI if available.
    """
    logger.debug("Running zero-shot classification for leadership")

    # Decide between local AI or OpenAI
    if ai_util.local_capable:
        logger.debug("Using local AI for classification")
        prompt = cfg["prompts"]["classification"]
        result = run_local_call(prompt, text, 10, cfg, ai_util.local_model, ai_util.local_tokenizer, ai_util.local_model_device)
    elif ai_util.openai_client is not None:
        logger.debug("Using OpenAI for classification")
        result = open_ai_call(
            cfg["prompts"]["classification"], text, 10, cfg,
            ai_util.openai_client,
            ai_util.openai_tokenizer,
            complex=False,
        )
    else:
        logger.debug("Failed classification classification")
        return False, None
    # Process the result
    if result.lower().startswith("yes"):
        is_leader = True
        parts = result.split(" ", 1)
        leader_name = (
            parts[1].strip(string.punctuation + "—").strip()
            if len(parts) > 1
            else ""
        )
    else:
        is_leader = False
        leader_name = ""
    return is_leader, leader_name

def short_summary(text: str, cfg: dict, ai_util) -> str:
    """
    Generate a short summary of the text, optionally using a local AI or Hugging Face's local summarization.
    """
    logger.debug("Requesting short summary")

    # Decide between local AI, Hugging Face summarizer, or OpenAI
    if ai_util.local_summarization:
        logger.debug("Using Hugging Face summarization for short summary")
        try:
            summary = summarize_text_using_local_model(input_text=text, model=ai_util.local_summarization_model, max_length=200, min_length=30)
        except Exception as e:
            logger.error(f"Error during Hugging Face summarization: {e}")
            return ""
    elif ai_util.local_capable:
        logger.debug("Using local AI for short summary")
        prompt = cfg["prompts"].get("short_summary", "")
        summary = run_local_call(prompt, text, 200, cfg, ai_util.local_model, ai_util.local_tokenizer,
                                 ai_util.local_model_device)

    elif ai_util.openai_client is not None:
        logger.debug("Using OpenAI for short summary")
        summary = open_ai_call(
            cfg["prompts"].get("short_summary", ""), text, 200, cfg, ai_util.openai_client, ai_util.openai_tokenizer
        )
        if summary == "":
            logger.debug("OpenAI returned an empty summary, trying fallback")
            try:
                summary = summarize_text_using_local_model(input_text=text, model=ai_util.local_summarization_model,
                                                           max_length=200, min_length=30)
            except Exception as e:
                logger.error(f"Error during Hugging Face summarization: {e}")
                return ""
    else:
        logger.debug("Failed to generate short summary: No available summarization method")
        return ""

    logger.info("Summary generated")
    return summary


def clean_summary(text: str, cfg: dict, ai_util, leader_name: str = None) -> str:
    """
    Generate a clean summary of the text, focusing on women leaders or a specific leader.
    Optionally, use a local AI model if available.

    Args:
        text (str): The text to be summarized.
        cfg (dict): Configuration dictionary containing OpenAI/local AI and prompt settings.
        client: OpenAI client for API calls.
        ai_util: Instance of AIUtils for local inference, if available.
        leader_name (str, optional): Name of the leader to emphasize, if identified.

    Returns:
        str: A cleaned summary of the text.
    """
    prompt = cfg["prompts"]["clean_summary"]

    # If leader_name is provided, replace placeholder
    if leader_name:
        prompt = prompt.replace("<leader_name>", leader_name)
    else:
        # Replace with a generic focus
        prompt = prompt.replace("<leader_name>", "women leaders generally")

    # Decide between local AI or OpenAI
    if ai_util.local_capable:
        logger.debug("Using local AI for clean summary")
        cleaned_summary = run_local_call(prompt, text, 500, cfg, ai_util.local_model, ai_util.local_tokenizer, ai_util.local_model_device)
    elif ai_util.openai_client is not None:
        logger.debug("Using OpenAI for clean summary")
        cleaned_summary = open_ai_call(
            prompt,
            text,
            4000,
            cfg,
            ai_util.openai_client,
            ai_util.openai_tokenizer,
            complex=True,
        )
    else:
        logger.debug("Failed clean summary")
        return text

    return cleaned_summary


def spin_genders(text: str, cfg: dict, ai_util) -> str:
    """
    Rewrite the text with genders spun, optionally using local AI if available.
    """
    logger.debug("Requesting spin_genders rewrite")

    # Decide between local AI or OpenAI
    if ai_util.local_capable:
        logger.debug("Using local AI for gender spinning")
        prompt = cfg["prompts"]["spin_genders"]
        spun_result = run_local_call(prompt, text, 500, cfg,  ai_util.local_model, ai_util.local_tokenizer, ai_util.local_model_device)
    elif ai_util.openai_client is not None:
        logger.debug("Using OpenAI for gender spinning")
        spun_result = open_ai_call(
            cfg["prompts"]["spin_genders"],
            text,
            4000,
            cfg,
            ai_util.openai_client,
            ai_util.openai_tokenizer,
            complex=True,
        )
    else:
        logger.debug("Failed spin_genders")
        return text

    logger.info("Received spin_genders rewrite")
    return spun_result
