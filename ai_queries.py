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
from timing_tracker import get_timing_tracker

logger = logging.getLogger(__name__)


def _get_model_name(cfg: dict, task: str = None) -> str:
    """
    Get the model name being used for a specific task.
    
    Args:
        cfg: Configuration dictionary
        task: Task name (classify_leadership, short_summary, clean_summary, spin_genders, etc.)
    
    Returns:
        String identifying the model (e.g., "gpt-4", "qwen-2", "local-model")
    """
    # Check Ollama first (highest priority)
    if cfg.get("ollama", {}).get("enabled", False):
        if task:
            model_key = f"{task}_model"
            return cfg.get("ollama", {}).get(model_key, cfg.get("ollama", {}).get("model", "ollama"))
        return cfg.get("ollama", {}).get("model", "ollama")
    
    # Check local AI
    if cfg.get("localai", {}).get("enabled", False):
        return cfg.get("localai", {}).get("model", "local-model")
    
    # Check OpenAI
    if cfg.get("openai", {}).get("api_key"):
        if task:
            model_key = f"{task}_model"
            return cfg.get("openai", {}).get(model_key, cfg.get("openai", {}).get("model", "gpt-3.5-turbo"))
        return cfg.get("openai", {}).get("model", "gpt-3.5-turbo")
    
    return "unknown"


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



def ollama_call(prompt: str, text: str, return_tokens: int, cfg: dict, ai_util, task: str = None):
    """
    Call the Ollama API for generating a response based on the given prompt and text.

    Returns:
        dict with:
        {
            "text": str,              # model response (or empty string)
            "refused": bool,          # True if model refused the request
            "raw": str                # unmodified raw text from the model
        }
    }
    """

    try:
        # ----------------------------------------------------------------------
        # SAFE INPUT LIMITING
        # ----------------------------------------------------------------------
        MAX_INPUT_CHARS = cfg.get("ollama", {}).get("max_input_chars", 32000)

        if len(text) > MAX_INPUT_CHARS:
            logger.warning(
                f"Input text too long ({len(text)} chars), truncating to {MAX_INPUT_CHARS} chars"
            )
            text = text[:MAX_INPUT_CHARS]

        # ----------------------------------------------------------------------
        # BUILD PROMPT
        # ----------------------------------------------------------------------
        prompt_plus_text = f"{prompt}\n\n{text}"

        logger.debug("Running Ollama call")

        # ----------------------------------------------------------------------
        # TEMPERATURE
        # ----------------------------------------------------------------------
        temp = cfg.get("ollama", {}).get("temperature", 0.1)

        # ----------------------------------------------------------------------
        # API CALL
        # ----------------------------------------------------------------------
        raw_result = ai_util._call_ollama_api(
            prompt_plus_text,
            max_tokens=return_tokens,
            temperature=temp,
            task=task,
        )

        logger.debug(f"Ollama result preview: {raw_result[:200]}...")

        # ----------------------------------------------------------------------
        # REFUSAL DETECTION (FIRST ~300 CHARACTERS)
        # ----------------------------------------------------------------------
        refusal_snippets = cfg.get("ollama", {}).get("refusal_markers", [
            "i can't",
            "i cannot",
            "i’m not able",
            "i am not able",
            "cannot assist",
            "can't assist",
            "can't help with",
            "cannot help with",
            "not allowed",
            "violates",
            "as an ai",
            "i'm unable",
            "i cannot create",
            "i cannot provide",
        ])

        # Normalize for checking
        head = raw_result[:300].lower()

        refused = any(marker in head for marker in refusal_snippets)

        should_return = {
            "text": "" if refused else raw_result,
            "refused": refused,
            "raw": raw_result,
        }
        return should_return["text"]

    except Exception as e:
        logger.error(f"Ollama API call failed: {e}")
        # return {
        #     "text": "",
        #     "refused": True,
        #     "raw": "",
        # }
        return ""


def open_ai_call(prompt: str, text: str, return_tokens: int, cfg: dict, client, tokenizer: PreTrainedTokenizer, task: str = None) -> str:
    """
    Call the OpenAI API for generating a response based on the given prompt and text.
    Includes retry logic with exponential backoff for rate limit errors.
    
    Args:
        prompt: The prompt to send to the model
        text: The text to process
        return_tokens: Maximum number of tokens to generate
        cfg: Configuration dictionary
        client: OpenAI client
        tokenizer: Tokenizer for the model
        task: Task name (classify_leadership, short_summary, clean_summary, spin_genders)
    """
    model = cfg["openai"].get("model", "gpt-3.5-turbo")
    if task:
        model_key = f"{task}_model"
        model = cfg["openai"].get(model_key, model)
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
    Classify text to determine if it mentions a leader, using Ollama, local AI, or OpenAI.
    """
    tracker = get_timing_tracker()
    model_name = _get_model_name(cfg, "classify_leadership")
    
    with tracker.time_task("classify_leadership", model_name):
        logger.debug("Running zero-shot classification for leadership")

        # Prioritize: Ollama > Local AI > OpenAI
        if ai_util.ollama_enabled:
            logger.debug("Using Ollama for classification")
            prompt = cfg["prompts"]["classification"]
            result = ollama_call(prompt, text, 10, cfg, ai_util, task="classify_leadership")
        elif ai_util.local_capable:
            logger.debug("Using local AI for classification")
            prompt = cfg["prompts"]["classification"]
            result = run_local_call(prompt, text, 10, cfg, ai_util.local_model, ai_util.local_tokenizer, ai_util.local_model_device)
        elif ai_util.openai_client is not None:
            logger.debug("Using OpenAI for classification")
            result = open_ai_call(
                cfg["prompts"]["classification"], text, 10, cfg,
                ai_util.openai_client,
                ai_util.openai_tokenizer,
                task="classify_leadership",
            )
        else:
            logger.debug("Failed classification: No LLM provider available")
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


def reject_classification(text: str, cfg: dict, ai_util, prompt_name: str) -> tuple[bool, str]:
    """
    Classify text to determine if it should be rejected, using the rejection_model.
    
    Args:
        text: The article text to classify
        cfg: Configuration dictionary
        ai_util: Instance of AIUtils for inference
        prompt_name: The name of the rejection prompt to use from config
    
    Returns:
        tuple: (is_rejected: bool, response: str) where response contains the full model output
    """
    tracker = get_timing_tracker()
    
    # Get the rejection model name
    rejection_model = cfg.get("ollama", {}).get("rejection_model", cfg.get("ollama", {}).get("model", "llama3.1:8b-instruct-q4_K_M"))
    
    with tracker.time_task("reject_classification", rejection_model):
        logger.debug(f"Running rejection classification with prompt: {prompt_name}")
        
        # Get the prompt from config
        prompt = cfg["prompts"].get(prompt_name)
        if not prompt:
            logger.error(f"Rejection prompt '{prompt_name}' not found in config")
            return False, ""
        
        # Prioritize: Ollama > Local AI > OpenAI
        if ai_util.ollama_enabled:
            logger.debug(f"Using Ollama rejection model: {rejection_model}")
            # Temporarily override the task to use rejection_model
            # We'll call the ollama API directly with the rejection model
            result = ai_util._call_ollama_api(
                f"{prompt}\n\n{text}",
                max_tokens=200,
                temperature=cfg.get("ollama", {}).get("temperature", 0.1),
                task="rejection"
            )
        elif ai_util.local_capable:
            logger.debug("Using local AI for rejection classification")
            result = run_local_call(prompt, text, 200, cfg, ai_util.local_model, ai_util.local_tokenizer, ai_util.local_model_device)
        elif ai_util.openai_client is not None:
            logger.debug("Using OpenAI for rejection classification")
            result = open_ai_call(
                prompt, text, 200, cfg,
                ai_util.openai_client,
                ai_util.openai_tokenizer,
                task="rejection",
            )
        else:
            logger.debug("Failed rejection classification: No LLM provider available")
            return False, ""
        
        # Check if the response starts with REJECT or ACCEPT
        result_upper = result.strip().upper()
        is_rejected = result_upper.startswith("REJECT")
        
        logger.debug(f"Rejection classification result: {result[:100]}...")
        
        return is_rejected, result


def short_summary(text: str, cfg: dict, ai_util) -> str:
    """
    Generate a short summary of the text, using Ollama, local AI, or Hugging Face's local summarization.
    """
    tracker = get_timing_tracker()
    model_name = _get_model_name(cfg, "short_summary")
    
    with tracker.time_task("short_summary", model_name):
        logger.debug("Requesting short summary")

        # Prioritize: Ollama > Local summarization > Local AI > OpenAI
        if ai_util.ollama_enabled:
            logger.debug("Using Ollama for short summary")
            prompt = cfg["prompts"].get("short_summary", "")
            summary = ollama_call(prompt, text, 200, cfg, ai_util, task="short_summary")
        elif ai_util.local_summarization:
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
                cfg["prompts"].get("short_summary", ""), text, 200, cfg, ai_util.openai_client, ai_util.openai_tokenizer, task="short_summary"
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
    Uses Ollama, local AI model, or OpenAI.

    Args:
        text (str): The text to be summarized.
        cfg (dict): Configuration dictionary containing Ollama/OpenAI/local AI and prompt settings.
        ai_util: Instance of AIUtils for inference.
        leader_name (str, optional): Name of the leader to emphasize, if identified.

    Returns:
        str: A cleaned summary of the text.
    """
    tracker = get_timing_tracker()
    model_name = _get_model_name(cfg, "clean_summary")
    
    with tracker.time_task("clean_summary", model_name):
        prompt = cfg["prompts"]["clean_summary"]

        # If leader_name is provided, replace placeholder
        if leader_name:
            prompt = prompt.replace("<leader_name>", leader_name)
        else:
            # Replace with a generic focus
            prompt = prompt.replace("<leader_name>", "women leaders generally")

        # Prioritize: Ollama > Local AI > OpenAI
        if ai_util.ollama_enabled:
            logger.debug("Using Ollama for clean summary")
            cleaned_summary = ollama_call(prompt, text, 4000, cfg, ai_util, task="clean_summary")
        elif ai_util.local_capable:
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
                task="clean_summary",
            )
        else:
            logger.debug("Failed clean summary: No LLM provider available")
            return text

        return cleaned_summary


def spin_genders(text: str, cfg: dict, ai_util) -> str:
    """
    Rewrite the text with genders spun, using Ollama, local AI, or OpenAI.
    """
    tracker = get_timing_tracker()
    model_name = _get_model_name(cfg, "spin_genders")
    
    with tracker.time_task("spin_genders", model_name):
        logger.debug("Requesting spin_genders rewrite")

        # Prioritize: Ollama > Local AI > OpenAI
        if ai_util.ollama_enabled:
            logger.debug("Using Ollama for gender spinning")
            prompt = cfg["prompts"]["spin_genders"]
            spun_result = ollama_call(prompt, text, 4000, cfg, ai_util, task="spin_genders")
        elif ai_util.local_capable:
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
                task="spin_genders",
            )
        else:
            logger.debug("Failed spin_genders: No LLM provider available")
            return text

        logger.info("Received spin_genders rewrite")
        return spun_result


def clean_article(text: str, cfg: dict, ai_util) -> str:
    """
    Clean an article by removing ads, subscription prompts, and other boilerplate, using Ollama, local AI, or OpenAI.
    """
    tracker = get_timing_tracker()
    model_name = _get_model_name(cfg, "clean_article")
    
    with tracker.time_task("clean_article", model_name):
        logger.debug("Requesting clean_article")

        # Prioritize: Ollama > Local AI > OpenAI
        if ai_util.ollama_enabled:
            logger.debug("Using Ollama for cleaning article")
            prompt = cfg["prompts"]["clean_article"]
            cleaned_result = ollama_call(prompt, text, 4000, cfg, ai_util, task="clean_article")
        elif ai_util.local_capable:
            logger.debug("Using local AI for cleaning article")
            prompt = cfg["prompts"]["clean_article"]
            cleaned_result = run_local_call(prompt, text, 500, cfg,  ai_util.local_model, ai_util.local_tokenizer, ai_util.local_model_device)
        elif ai_util.openai_client is not None:
            logger.debug("Using OpenAI for cleaning article")
            cleaned_result = open_ai_call(
                cfg["prompts"]["clean_article"],
                text,
                4000,
                cfg,
                ai_util.openai_client,
                ai_util.openai_tokenizer,
                task="clean_article",
            )
        else:
            logger.debug("Failed clean_article: No LLM provider available")
            return text

        logger.info("Received clean_article")
        return cleaned_result


def add_background_on_women(text: str, cfg: dict, ai_util) -> str:
    """
    Add background information about women mentioned in the text, using Ollama, local AI, or OpenAI.
    """
    tracker = get_timing_tracker()
    model_name = _get_model_name(cfg, "add_background_on_women")
    
    with tracker.time_task("add_background_on_women", model_name):
        logger.debug("Requesting add_background_on_women enhancement")

        # Prioritize: Ollama > Local AI > OpenAI
        if ai_util.ollama_enabled:
            logger.debug("Using Ollama for adding background on women")
            prompt = cfg["prompts"]["add_background_on_women"]
            enhanced_result = ollama_call(prompt, text, 4000, cfg, ai_util, task="add_background_on_women")
        elif ai_util.local_capable:
            logger.debug("Using local AI for adding background on women")
            prompt = cfg["prompts"]["add_background_on_women"]
            enhanced_result = run_local_call(prompt, text, 500, cfg,  ai_util.local_model, ai_util.local_tokenizer, ai_util.local_model_device)
        elif ai_util.openai_client is not None:
            logger.debug("Using OpenAI for adding background on women")
            enhanced_result = open_ai_call(
                cfg["prompts"]["add_background_on_women"],
                text,
                4000,
                cfg,
                ai_util.openai_client,
                ai_util.openai_tokenizer,
                task="add_background_on_women",
            )
        else:
            logger.debug("Failed add_background_on_women: No LLM provider available")
            return text

        logger.info("Received add_background_on_women enhancement")
        return enhanced_result
