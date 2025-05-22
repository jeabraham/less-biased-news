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
    """
    Call the OpenAI API for generating a response based on the given prompt and text.
    """
    model = cfg["openai"]["model"]
    max_toks = cfg["openai"]["max_tokens"]
    temp = cfg["openai"].get("temperature", 0)

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
        logger.debug(f"OpenAI result: {answer}")
        return answer
    except Exception as e:
        logger.warning(f"OpenAI API error: {e}")
        return ""


def classify_leadership(text: str, cfg: dict, client=None, ai_util=None) -> bool:
    """
    Classify text to determine if it mentions a leader, optionally using a local AI if available.
    """
    logger.debug("Running zero-shot classification for leadership")

    # Decide between local AI or OpenAI
    if ai_util and ai_util.local_capable:
        logger.debug("Using local AI for classification")
        prompt = cfg["prompts"]["classification"]
        result = ai_util.generate_response(prompt + "\n" + text)
    elif client is not None:
        logger.debug("Using OpenAI for classification")
        result = open_ai_call(
            cfg["prompts"]["classification"], text, 10, cfg, client
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


def short_summary(text: str, cfg: dict, client=None, ai_util=None) -> str:
    """
    Generate a short summary of the text, optionally using a local AI if available.
    """
    logger.debug("Requesting short summary")

    # Decide between local AI or OpenAI
    if ai_util and ai_util.local_capable:
        logger.debug("Using local AI for short summary")
        prompt = cfg["prompts"]["short_summary"]
        summary = ai_util.generate_response(prompt + "\n" + text)
    elif client is not None:
        logger.debug("Using OpenAI for short summary")
        summary = open_ai_call(
            cfg["prompts"]["short_summary"], text, 200, cfg, client
        )
    else:
        logger.debug("Failed short summary")
        return ""

    logger.info("Summary generated")
    return summary


def clean_summary(text: str, cfg: dict, client=None, ai_util=None, leader_name: str = None) -> str:
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
    if ai_util and ai_util.local_capable:
        logger.debug("Using local AI for clean summary")
        cleaned_summary = ai_util.generate_response(prompt + "\n" + text)
    elif client is not None:
        logger.debug("Using OpenAI for clean summary")
        cleaned_summary = open_ai_call(prompt, text, 4000, cfg, client)
    else:
        logger.debug("Failed clean summary")
        return text

    return cleaned_summary


def spin_genders(text: str, cfg: dict, client=None, ai_util=None) -> str:
    """
    Rewrite the text with genders spun, optionally using local AI if available.
    """
    logger.debug("Requesting spin-genders rewrite")

    # Decide between local AI or OpenAI
    if ai_util and ai_util.local_capable:
        logger.debug("Using local AI for gender spinning")
        prompt = cfg["prompts"]["spin_genders"]
        spun_result = ai_util.generate_response(prompt + "\n" + text)
    elif client is not None:
        logger.debug("Using OpenAI for gender spinning")
        spun_result = open_ai_call(
            cfg["prompts"]["spin_genders"], text, 4000, cfg, client
        )
    else:
        logger.debug("Failed sping genders")
        return text

    logger.info("Received spin-genders rewrite")
    return spun_result
