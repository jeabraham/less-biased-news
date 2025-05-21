# openai_utils.py
import logging
import string

logger = logging.getLogger(__name__)

def truncate_for_openai(prompt: str, text: str, max_tokens: int, chars_per_token: int = 4) -> str:
    """
    Naïvely truncate `text` so that its length in characters
    does not exceed (completion_tokens * chars_per_token).
    """
    char_limit = max_tokens * chars_per_token - len(prompt)
    if len(text) > char_limit:
        logger.warning(
            f"Input body too long ({len(text)} chars); truncating to {char_limit}"
        )
        return text[:char_limit]
    return text


def open_ai_call(prompt: str, text: str, return_tokens: int, cfg: dict, client) -> str:
    model    = cfg["openai"]["model"]
    max_toks = cfg["openai"]["max_tokens"]
    temp     = cfg["openai"].get("temperature", 0)

    safe_text = truncate_for_openai(prompt, text, max_tokens=max_toks - return_tokens)
    prompt_plus_text = prompt + "\n\n" + safe_text
    logger.debug("Running OpenAI call")

    try:
        resp = client.ChatCompletion.create(
            model=model,
            temperature=temp,
            max_tokens=return_tokens,
            messages=[{"role": "user", "content": prompt_plus_text}]
        )
        answer = resp.choices[0].message.content.strip()
        logger.debug(f"OpenAPI result: {answer}")
        return answer
    except Exception as e:
        logger.warning(f"OpenAI classification error: {e}")
        return ""

def classify_leadership(text: str, cfg: dict, client) -> bool:
    logger.debug("Running zero-shot classification for leadership")
    answer = open_ai_call(cfg["prompts"]["classification"], text, 10, cfg, client)
    # Process the result
    if answer.lower().startswith("yes"):
        is_leader = True
        # Check if there's more text after "yes" (indicating a name)
        parts = answer.split(" ", 1)
        leader_name = parts[1].strip(string.punctuation + "—").strip() if len(parts) > 1 else ""  # Clean name
    else:
        is_leader = False
        leader_name = ""
    return is_leader, leader_name


def short_summary(text: str, cfg: dict, client) -> str:
    logger.debug("Requesting short summary from OpenAI")
    summary = open_ai_call(cfg["prompts"]["short_summary"], text, 200, cfg, client)
    logger.info("Received summary")
    return summary


def clean_summary(text: str, cfg: dict, client, leader_name: str = None) -> str:
    """
    Cleanly summarize the given text, focusing on women leaders and optionally
    emphasizing a specific leader if their name is provided.

    Args:
        text (str): The text to be summarized.
        cfg (dict): Configuration dictionary containing OpenAI and prompt settings.
        client: OpenAI client for API calls.
        leader_name (str, optional): The name of the leader to emphasize, if identified.

    Returns:
        str: A cleaned summary of the text.
    """
    prompt = cfg["prompts"]["clean_summary"]

    # If leader_name is provided, replace the placeholder `<leader_name>` in the prompt
    if leader_name:
        prompt = prompt.replace("<leader_name>", leader_name)
    else:
        # If no leader is identified, replace the placeholder with a generic focus
        prompt = prompt.replace("<leader_name>", "women leaders generally")

    # Make the OpenAI API call with the updated prompt
    cleaned = open_ai_call(prompt, text, 4000, cfg, client)

    return cleaned

def spin_genders(text: str, cfg: dict, client) -> str:
    spun = open_ai_call(cfg["prompts"]["spin_genders"], text, 4000, cfg, client)
    logger.info("Received spin-genders rewrite")
    return spun
