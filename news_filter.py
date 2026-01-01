import yaml
import logging
import spacy
from newsapi import NewsApiClient
from genderize import Genderize, GenderizeException

import argparse
import requests
import urllib.parse


from ai_utils import AIUtils
from article_fetcher import fetch_full_text_and_images

from analyze_image_gender import analyze_image_gender, FEMALE_CONFIDENCE_THRESHOLD
from datetime import timedelta, datetime
import time

import math
import os
import json
from datetime import date


from ai_queries import (
    classify_leadership,
    short_summary,
    clean_summary,
    spin_genders,
    add_background_on_women,
    clean_article,
    reject_classification,
)
from news_formatter import generate_text, generate_html
from timing_tracker import get_timing_tracker

# Approximate a character→token ratio if tiktoken not available:
APPROX_CHARS_PER_TOKEN = 4

# ─── Logging Setup ────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

try:
    from gender_guesser.detector import Detector as LocalGenderDetector
    local_detector = LocalGenderDetector()
    LOCAL_GENDER_AVAILABLE = True
except ImportError:
    LOCAL_GENDER_AVAILABLE = False

def setup_logging(level: str = "INFO"):
    """
    Initialize logging:
     - Console at the user-specified level (INFO/DEBUG/etc.)
     - File ('news_filter.log') at DEBUG level
    """
    # 1) Grab and reset the root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)           # <— ensure ALL messages are processed
    for h in list(root.handlers):
        root.removeHandler(h)

    # 2) Determine the console log level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # 3) Common formatter
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)

    # 4) Console handler at user‐specified level
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # 5) File handler at DEBUG
    log_file = "news_filter.log"
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    logger.debug(f"Logging initialized at {level} level; full debug log in '{log_file}'")

# ─── Config Loading ────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    logger.debug(f"Loading config from {path}")
    if not os.path.exists(path):
        logger.error(f"Config file not found: {path}")
        raise FileNotFoundError(f"{path} does not exist")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Configuration loaded successfully")
    return cfg


provider_last_call = {}


def fetch_pause(provider_name: str, cfg: dict, delay_key: str = "delay"):
    """
    Ensures sufficient delay between calls to the same provider.

    :param provider_name: Name of the provider (e.g., 'mediastack')
    :param cfg: Configuration dictionary containing delay information
    :param delay_key: Key in the configuration to look up the delay (default: "delay")
    """
    global provider_last_call

    # Get the delay for the provider or default to 0
    delay = cfg.get(provider_name, {}).get(delay_key, 0)

    # Get the last call time for the provider
    last_call = provider_last_call.get(provider_name)

    # If the last call exists and delay time hasn't passed, pause
    if last_call:
        elapsed_time = datetime.now() - last_call
        if elapsed_time < timedelta(seconds=delay):
            sleep_time = (timedelta(seconds=delay) - elapsed_time).total_seconds()
            print(f"Pausing for {sleep_time:.2f} seconds before calling {provider_name}")
            time.sleep(sleep_time)

    # Update the last call time to now
    provider_last_call[provider_name] = datetime.now()


def create_curl_command(url, params):
    """
    Create a curl command from the URL and parameters that can be used for debugging.
    Masks the API key partially for security.
    """
    # Create a copy of params to avoid modifying the original
    safe_params = params.copy()

    # Mask the API key for security (show only first 8 and last 4 chars)
    if "access_key" in safe_params and len(safe_params["access_key"]) > 12:
        key = safe_params["access_key"]
        masked_key = key[:8] + "..." + key[-4:]
        safe_params["access_key"] = masked_key

    # URL encode parameters
    param_strings = []
    for key, value in safe_params.items():
        encoded_value = urllib.parse.quote(str(value))
        param_strings.append(f"{key}={encoded_value}")

    # Construct the full URL with parameters
    url_with_params = f"{url}?{'&'.join(param_strings)}"

    # Create the curl command
    curl_cmd = f"curl -X GET '{url_with_params}'"

    # Add note about the masked API key
    if "access_key" in safe_params and "..." in safe_params["access_key"]:
        curl_cmd += "\n# Note: API key has been partially masked for security. Replace with your full key."

    return curl_cmd


def fetch_mediastack(cfg: dict, qcfg: dict, use_cache: bool = False, cache_dir: str = "cache") -> list:
    """
    Fetch articles from Mediastack, or if use_cache=True and a cached file exists,
    load from disk instead of hitting the API. Always saves live results to cache.

    :param cfg: Configuration dictionary for accessing provider settings
    :param qcfg: Query-specific configuration dictionary
    :param use_cache: Whether to load results from cache if available
    :param cache_dir: Directory to save or read cache files
    :return: A list of fetched articles
    """

    # ensure cache directory
    os.makedirs(cache_dir, exist_ok=True)
    # derive a safe filename from the query name
    safe_name = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in qcfg["name"])
    cache_file = os.path.join(cache_dir, f"{safe_name}.json")

    # 1) Try loading from cache
    if use_cache and os.path.exists(cache_file):
        logger.info(f"Loading Mediastack results for '{qcfg['name']}' from cache")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    fetch_pause("mediastack", cfg)

    base_url = cfg["mediastack"].get("base_url")
    raw_kw = qcfg.get("q")
    keywords = ",".join(raw_kw) if isinstance(raw_kw, list) else (raw_kw or "")
    params = {"access_key": cfg["mediastack"]["access_key"], "limit": qcfg.get("page_size", 100)}

    # Determine history_days (per-query override or global default)
    history = qcfg.get("history_days",
                       cfg["mediastack"].get("default_history_days", None))
    if history:
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=history)).isoformat()
        params["date"] = f"{start},{end}"

    # Set additional parameters
    if keywords:
        params["keywords"] = keywords
    if qcfg.get("countries"):
        params["countries"] = qcfg.get("countries")
    if qcfg.get("categories"):
        params["categories"] = qcfg.get("categories")
    if qcfg.get("languages"):  # Support for languages parameter
        params["languages"] = qcfg.get("languages")  # Add languages to params

    logger.debug(f"Mediastack query params: {params}")
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        logger.debug(f"Mediastack request URL: {resp.url}")
        resp.raise_for_status()

        data = resp.json()
        articles = data.get("data", [])

        if not articles:
            # Create a curl command that users can run to debug
            curl_cmd = create_curl_command(base_url, params)

            # Log error with parameters and curl command
            logger.error(
                f"Mediastack returned zero articles with params: {params}\n"
                f"Try running this curl command to debug:\n{curl_cmd}"
            )

            # Log metadata if available
            if "pagination" in data:
                logger.info(f"Response metadata: pagination={data['pagination']}")

            # Additional info about possible reasons
            logger.info(
                "Possible reasons for zero results:\n"
                "1. API key might be invalid or expired\n"
                "2. Query might be too restrictive\n"
                "3. Rate limit might be exceeded\n"
                "4. API service might be experiencing issues"
            )
        else:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(articles, f, indent=2, ensure_ascii=False)
                logger.info(f"Cached {len(articles)} items to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to write cache file {cache_file}: {e}")
            logger.info(f"Fetched {len(articles)} articles from Mediastack")

        return articles
    except Exception as e:
        logger.error(f"Mediastack fetch failed: {e}")

        # Add curl command for debugging the failed request
        curl_cmd = create_curl_command(base_url, params)
        logger.error(f"Failed request details - try this curl command:\n{curl_cmd}")

        return []

def get_cache_file_path(query_name: str) -> str:
    """
    Retrieves the path for the cache file for a given query name.
    Now uses a single growing cache file instead of separate current/yesterday files.

    :param query_name: Name of the query to retrieve path for
    :return: Path to the cache file
    """
    base_dir = "cache"  # Base directory for cached data
    cache_file = os.path.join(base_dir, f"{query_name}_cache.json")
    return cache_file

def load_cache(file_path: str) -> dict:
    """
    Loads the cache from the specified file. Returns an empty dictionary if the file doesn't exist.
    
    The cache structure is:
    {
        "articles": {
            "<article_key>": {
                "first_seen": "2025-12-07T12:34:56",
                "article_data": {...}
            }
        }
    }

    :param file_path: Path to the cache file
    :return: Dictionary with the cache contents
    """
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {"articles": {}}

def get_article_key(article: dict) -> str:
    """
    Generate a unique key for an article based on URL and title.
    
    :param article: Article dictionary
    :return: Unique key string
    """
    url = article.get("url", "")
    title = article.get("title", "")
    # Use both URL and title for better uniqueness
    return f"{url}||{title}"

def expire_cache(file_path: str, expire_days: int):
    """
    Remove cache entries older than the specified number of days.
    
    :param file_path: Path to the cache file
    :param expire_days: Number of days after which to expire entries
    """
    if not os.path.exists(file_path):
        logger.debug(f"Cache file {file_path} does not exist, nothing to expire")
        return
    
    cache = load_cache(file_path)
    
    # Calculate cutoff date
    cutoff = datetime.now() - timedelta(days=expire_days)
    
    # Filter out expired entries
    original_count = len(cache["articles"])
    expired_keys = []
    
    for key, entry in cache["articles"].items():
        first_seen_str = entry.get("first_seen", "")
        if first_seen_str:
            try:
                first_seen = datetime.fromisoformat(first_seen_str)
                if first_seen < cutoff:
                    expired_keys.append(key)
            except ValueError:
                logger.warning(f"Invalid timestamp format for cache entry: {first_seen_str}")
    
    # Remove expired entries
    for key in expired_keys:
        del cache["articles"][key]
    
    if expired_keys:
        logger.info(f"Expired {len(expired_keys)} cache entries older than {expire_days} days")
        # Save updated cache
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(cache, file, indent=4)
    else:
        logger.debug(f"No cache entries to expire (checked {original_count} entries)")

def save_cache(file_path: str, data: dict):
    """
    Saves the cache data to a file. Merges new articles with existing cache.
    Each article is stored with a timestamp of when it was first processed.

    :param file_path: Path to the cache file
    :param data: Data to save - should have structure {"articles": [...]}
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists
    
    # Load existing cache
    existing_cache = load_cache(file_path)
    
    # Get current timestamp
    current_time = datetime.now().isoformat()
    
    # Merge new articles with existing ones
    new_articles = data.get("articles", [])
    for article in new_articles:
        article_key = get_article_key(article)

        # Ensure we persist female-leader fields
        if article.get("status") == "female_leader":
            article["is_female_leader"] = True
            # leader_name may be None if not identified
        else:
            # Keep explicit False unless already True
            article.setdefault("is_female_leader", False)

        # Only add if not already in cache (preserves first_seen timestamp)
        if article_key not in existing_cache["articles"]:
            existing_cache["articles"][article_key] = {
                "first_seen": current_time,
                "article_data": article
            }
        else:
            # Update article data but keep original first_seen timestamp
            # Merge to avoid dropping existing leader info if newly missing
            cached_art = existing_cache["articles"][article_key]["article_data"]
            merged = dict(cached_art)
            merged.update(article)
            # If cached had leader_name and new one is missing, keep cached
            if "leader_name" in cached_art and not merged.get("leader_name"):
                merged["leader_name"] = cached_art["leader_name"]
            # If either indicates female leader, keep True
            if cached_art.get("is_female_leader") or merged.get("status") == "female_leader":
                merged["is_female_leader"] = True
            existing_cache["articles"][article_key]["article_data"] = merged
    
    # Save merged cache
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(existing_cache, file, indent=4)

def replace_male_first_names_with_initials(text, gender_map):
    """
    Replace male first names in the provided text with their initials, based on a given gender map.

    Args:
        text (str): The input text where names will be replaced.
        gender_map (dict): A dictionary mapping full names to their genders
                           (e.g., {"Stephen Hunter": "male", "Alice Doe": "female"}).

    Returns:
        str: The text with male first names replaced with their initials.
    """
    import re

    if not isinstance(text, str):
        raise ValueError("The input argument 'text' must be a string.")
    if not isinstance(gender_map, dict):
        raise ValueError("The input argument 'gender_map' must be a dictionary.")

    # Reverse sort persons by length to handle cases like ("John Smith", "John") properly
    male_names = [name for name, gender in gender_map.items() if gender and gender.lower() == "male"]
    male_names.sort(key=len, reverse=True)

    def replace_name(match):
        full_name = match.group(0)
        if full_name in gender_map and gender_map[full_name].lower() == "male":
            first_name, *last_name = full_name.split(" ")
            if first_name and last_name:
                # Replace first name with its initial
                return f"{first_name[0]}. {' '.join(last_name)}"
        return full_name

    # Construct regex to find male names in text
    name_pattern = r'\b(?:' + '|'.join(map(re.escape, male_names)) + r')\b'
    return re.sub(name_pattern, replace_name, text)

import re

def replace_male_pronouns_with_neutral(text):
    """
    Replace masculine pronouns with gender-neutral ones,
    preserving capitalization automatically and distinguishing
    'his' determiner vs. 'his' possessive pronoun.
    
    The logic:
    - "his" followed by adverbs at end/before punctuation → "theirs" (possessive pronoun)
    - "his" followed by a word → "their" (determiner)
    - "his" at end or before punctuation → "theirs" (possessive pronoun)
    """

    if not isinstance(text, str):
        raise ValueError("The input argument 'text' must be a string.")

    # Order matters: we need to handle 'his' carefully to distinguish determiner vs. possessive pronoun
    replacements = [
        # himself
        (r"\bhimself\b", "themself"),

        # 'his' before common adverbs that typically come at END (followed by punctuation or end of string)
        # This handles cases like "his too." or "his alone," or "his also!"
        (r"\bhis\b(?=\s+(?:too|alone|also|as well|entirely|exclusively|only)(?:\s*[,;.!?]|\s*$))", "theirs"),

        # 'his' determiner (followed by space and then a word) → their
        # This pattern matches "his" when it's followed by whitespace and then a word character
        # This will catch most cases like "his book", "his lifelong dream", "his only wish", etc.
        (r"\bhis\b(?=\s+\w)", "their"),

        # standalone 'his' → theirs (possessive pronoun)
        # This catches everything else: end of string, before punctuation, etc.
        (r"\bhis\b", "theirs"),

        # him → them
        (r"\bhim\b", "them"),

        # he → they
        (r"\bhe\b", "they"),
    ]

    def match_case(repl, original):
        """Return repl but matching the capitalization pattern of original."""
        if original.isupper():
            return repl.upper()
        if original[0].isupper():
            return repl.capitalize()
        return repl

    def replacer(match, repl):
        word = match.group(0)
        return match_case(repl, word)

    for pattern, repl in replacements:
        text = re.sub(
            pattern,
            lambda m, r=repl: replacer(m, r),
            text,
            flags=re.IGNORECASE
        )

    return text


# ─── Fetch & Filter ────────────────────────────────────────────────────

def fetch_articles(qcfg: dict, cfg: dict, use_cache: bool = False) -> list:
    """
    Fetch articles from a single news source based on query configuration.
    
    Args:
        qcfg: Query-specific configuration dictionary containing provider, query parameters, etc.
        cfg: Global configuration dictionary
        use_cache: Whether to use cached results if available
        
    Returns:
        list: List of raw article dictionaries fetched from the news source
    """
    name = qcfg.get("name", "Unknown")
    logger.info(f"Fetching articles for query '{name}'")
    
    # Determine which provider to use
    provider = qcfg.get("provider", "newsapi").lower()
    
    if provider == "mediastack":
        raw_articles = fetch_mediastack(cfg, qcfg, use_cache=use_cache)
    else:
        # Initialize NewsAPI client for newsapi provider
        newsapi = NewsApiClient(api_key=cfg["newsapi"]["api_key"])
        raw_articles = fetch_newsapi(cfg, name, qcfg, newsapi)
    
    logger.info(f"Fetched {len(raw_articles)} articles from {provider} for '{name}'")
    return raw_articles


def fetch_and_filter(cfg: dict, use_cache: bool = False, new_today: bool = False, expire_days: int = None) -> dict:
    logger.info("Initializing NewsAPI client")
    newsapi = NewsApiClient(api_key=cfg["newsapi"]["api_key"])
    logger.info("Loading spaCy model")
    nlp = spacy.load("en_core_web_sm")
    gndr = Genderize()

    aiclient = AIUtils(cfg)

    max_images = cfg.get("max_images", 3)  # Default to 3 if not specified in the YAML file

    results = {}
    for qcfg in cfg["queries"]:
        name = qcfg["name"]
        logger.info(f"Processing query '{name}'")

        # ─── Load and Manage Cache for `new_today` Logic ────────────────
        cache_file = get_cache_file_path(name)  # Get single cache file path
        
        # Expire old cache entries if requested
        if expire_days is not None:
            expire_cache(cache_file, expire_days)
        
        cached_keys = set()
        cached_data_map = {}
        if new_today:
            logger.info("Loading cache for new_today detection...")
            cache = load_cache(cache_file)
            
            # Build a set of cached article keys for quick lookup
            for article_key, entry in cache.get("articles", {}).items():
                cached_keys.add(article_key)
                # Store the cached article data for reuse
                cached_data_map[article_key] = entry["article_data"]
            
            logger.debug(f"Number of articles in cache: {len(cached_keys)}")
        else:
            logger.info("new_today is False. Skipping cache operations.")

        # ─── Fetch Raw Articles From the Right Source ────────────────
        raw_articles = fetch_articles(qcfg, cfg, use_cache=use_cache)

        # ─── Extract Names, Bodies, and Images ────────────────
        bodies, images, all_persons = gather_bodies_images_and_persons(raw_articles, nlp, max_images=max_images)

        # Perform gender detection
        gender_map = guess_genders(all_persons, gndr)

        hits = []
        stats = {"total": len(raw_articles), "persons": 0, "female_names": 0,
                 "keyword_hits": 0, "classified_ok": 0}

        summarize_selected = qcfg.get(
            "summarize_selected",
            cfg.get("default_summarize_selected", False)
        )

        # ─── Process Each Article ────────────────
        tracker = get_timing_tracker()
        for art, body, image_list in zip(raw_articles, bodies, images):
            # Process spaCy PERSON entities
            art["body"] = body
            
            # Check if article is in cache
            article_key = get_article_key(art)
            
            # Tag new articles based on cache, if enabled
            if new_today:
                is_new = article_key not in cached_keys
                art["new_today"] = is_new
                if is_new:
                    categorize_article_and_generate_content(art, image_list, cfg, qcfg, aiclient, nlp, gender_map,
                                                            stats,
                                                            summarize_selected)
                    # Log timing for new articles only
                    tracker.log_article_timing(art.get("title", "Unknown"), name)
                else:
                    # Fetch status and content from cached data if article is not new
                    cached_art = cached_data_map.get(article_key)
                    if cached_art:
                        art["status"] = cached_art.get("status", qcfg["fallback"])
                        art["content"] = cached_art.get("content", art.get("description", "[No content available]"))
                        art["image_urls"] = cached_art.get("image_urls", [])
                        art["most_relevant_status"] = cached_art.get("most_relevant_status", "no_face")
                        art["most_relevant_image"] = cached_art.get("most_relevant_image", None)
                        art["image_analysis"] = cached_art.get("image_analysis", {})
                        # Preserve female leader classification and leader name from cache
                        art["leader_name"] = cached_art.get("leader_name", None)
                        art["is_female_leader"] = cached_art.get("is_female_leader", art.get("status") == "female_leader")
                logger.debug(f"{'New Title ->' if is_new else 'NOT New Title ->'} {art['title']}")
            else:
                categorize_article_and_generate_content(art, image_list, cfg, qcfg, aiclient, nlp, gender_map,
                                                        stats,
                                                        summarize_selected)
                # Log timing for all articles when not using cache
                tracker.log_article_timing(art.get("title", "Unknown"), name)

            hits.append(art)

        logger.info(
            f"Stats for '{name}': total={stats['total']}, persons={stats['persons']}, "
            f"female_names={stats['female_names']}, keyword_hits={stats['keyword_hits']}, "
            f"classified_ok={stats['classified_ok']}"
        )

        results[name] = hits

        # ─── Save Processed Articles to Cache ────────────────
        save_cache(cache_file, {"articles": hits})  # Save processed articles to cache
        
        # ─── Log Query Completion Timing ────────────────
        tracker.log_query_complete(name)

    return results

def check_boolean(cfg: dict, name: str, default: bool = False) -> bool:
    """
    Interpret a configuration value as boolean.

    - Accepts actual booleans from YAML (True/False).
    - Accepts common string forms: "true"/"yes"/"1" => True, "false"/"no"/"0" => False (case-insensitive).
    - If key not present, returns default.
    - If value type is unexpected, falls back to bool(value).

    name can be a key or a dotted path (e.g., "female_leadership_detection.use_local_checks").
    """
    def get_by_path(d, path):
        cur = d
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    raw = get_by_path(cfg, name)
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return raw != 0
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"true", "yes", "y", "on", "1"}:
            return True
        if s in {"false", "no", "n", "off", "0"}:
            return False
    return bool(raw)

def categorize_article_and_generate_content(art,  image_list, cfg, qcfg, aiclient, nlp, gender_map, stats,
                                   summarize_selected = True):
    # We'll clean the article when we're sure we need it cleaned
    #body = clean_article(art["body"], cfg, aiclient)
    body = art["body"]
    
    # Check if rejection prompt is specified for this query
    rejection_prompt = qcfg.get("rejection_prompt")
    if rejection_prompt:
        logger.info(f"Checking article '{art.get('title', 'Unknown')}' for rejection using prompt: {rejection_prompt}")
        is_rejected, rejection_response = reject_classification(body, cfg, aiclient, rejection_prompt)
        
        if is_rejected:
            logger.info(f"Article REJECTED: {art.get('title', 'Unknown')}")
            logger.info(f"Rejection response: {rejection_response}")
            # Mark article as excluded and store rejection response
            art["status"] = "exclude"
            art["rejection_response"] = rejection_response
            return  # Skip further processing (no content generation needed)
        else:
            logger.info(f"Article ACCEPTED: {art.get('title', 'Unknown')}")
    
    doc = nlp(body)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    stats["persons"] += bool(persons)  # Increment PERSON counter if entities are found
    if check_boolean(qcfg, "classification", True):
        is_female_leader, leader_name = identify_female_leadership(body, cfg, gender_map, persons, stats, aiclient)
    else:
        is_female_leader, leader_name = False, None
    # Classify articles based on gender leadership
    if is_female_leader:
        art["status"] = "female_leader"
        art["leader_name"] = leader_name
        stats["classified_ok"] += 1
    else:
        # Check if any detected person in 'persons' is mapped to "women" via gender_map
        for person in persons:
            if (gender_map.get(person) or "").lower() == "female":
                art["status"] = qcfg.get("fallback_women", qcfg["fallback"])  # Use fallback_women if it exists
                break
        else:
            art["status"] = qcfg["fallback"]

    # Process all images to classify them for HTML display (even if we don't use fallback_image_female)
    process_article_images(art, image_list)
    
    # Handle article processing based on classification
    if art["status"] == "female_leader":
        # If the article is classified as about a female leader, handle content summarization
        cleaned = clean_article(body, cfg, aiclient)
        if summarize_selected:
            logger.info(f"Clean-summarizing '{art['title']}'")
            art["content"] = clean_summary(cleaned, cfg, aiclient, leader_name=art.get("leader_name"))
        else:
            # Keep the full fetched body
            art["content"] = cleaned
    else:
        # Handle fallback logic for image statuses (check if first image contains a woman)
        img_stat = art.get("most_relevant_status", "")
        if img_stat in ("female", "female_majority", "female_prominent"):
            art["status"] = qcfg.get("fallback_image_female", qcfg["fallback"])
            logger.info(f"Applying image-based fallback '{art['status']}' for article '{art['title']}'")
        if art["status"] == "show_full":
            cleaned = clean_article(body, cfg, aiclient)
            art["content"] = clean_summary(cleaned, cfg, aiclient, leader_name="women in general") if summarize_selected else cleaned
        elif art["status"] == "short_summary":
            # Generate a short summary
            cleaned = clean_article(body, cfg, aiclient)
            art["content"] = short_summary(cleaned, cfg, aiclient)
        elif art["status"] == "spin_genders":
            # Apply gender-spin logic
            cleaned = clean_article(body, cfg, aiclient)
            art["content"] = spin_genders(cleaned, cfg, aiclient)
        else:
            # Default to the article's description, but this article will usually be excluded anyways.
            art["content"] = art.get("description", "")
    if qcfg.get("remove_male_pronouns", False):
        art["content"] = replace_male_pronouns_with_neutral(art["content"])
    if qcfg.get("male_initials", False):
        art["content"] = replace_male_first_names_with_initials(art["content"], gender_map)
    if art["status"] in ("show_full", "spin_genders"):
        art["content"] = add_background_on_women(art["content"], cfg, aiclient)

def identify_female_leadership(body, cfg, gender_map, persons, stats, aiclient):
    """
    Identify if article is about female leadership.
    
    Behavior can be controlled by cfg['female_leadership_detection']:
    - use_local_checks: If True (default), check for female names and keywords first
    - require_female_names: If True (default), only call LLM if female names detected
    - require_leadership_keywords: If True (default), only call LLM if leadership keywords detected
    
    If use_local_checks is False, always calls classify_leadership directly.
    """
    # Get configuration settings
    detection_cfg = cfg.get("female_leadership_detection", {})
    use_local_checks = detection_cfg.get("use_local_checks", True)
    require_female_names = detection_cfg.get("require_female_names", True)
    require_leadership_keywords = detection_cfg.get("require_leadership_keywords", True)
    
    is_female_leader = False
    leader_name = None
    
    if not use_local_checks:
        # Skip local checks, directly call LLM
        try:
            is_female_leader, leader_name = classify_leadership(body, cfg, aiclient)
        except Exception as e:
            logger.warning(f"LLM classification error: {e}")
        return is_female_leader, leader_name
    
    # Perform local checks
    female_names = [p for p in persons if gender_map.get(p) == "female"]
    has_kw = any(
        kw.lower() in body.lower()
        for kw in cfg.get("leadership_keywords", [])
    )
    
    if female_names:
        stats["female_names"] += female_names.__len__()
    if has_kw:
        stats["keyword_hits"] += 1
    
    # Determine if we should call the LLM based on configuration
    should_call_llm = True
    if require_female_names and not female_names:
        should_call_llm = False
    if require_leadership_keywords and not has_kw:
        should_call_llm = False
    
    if should_call_llm:
        try:
            is_female_leader, leader_name = classify_leadership(body, cfg, aiclient)
        except Exception as e:
            logger.warning(f"LLM classification error: {e}")
    
    return is_female_leader, leader_name


def process_article_images(art, image_list):
    """
    Process a list of images associated with an article and determine the overall image status.
    """
    art["image_urls"] = image_list
    art["most_relevant_status"] = "no_face"
    art["most_relevant_image"] = None
    art["image_analysis"] = {}

    valid_statuses = ["female", "female_majority", "female_prominent"]
    default_priority = len(valid_statuses)

    logger.info(f"[ImagePipeline] Processing {len(image_list)} images for article: {art.get('title','(untitled)')}")

    for image_url in image_list:
        logger.info(f"[ImagePipeline] → Analyzing: {image_url}")

        try:
            faces, dimensions = analyze_image_gender(image_url)
        except Exception as e:
            logger.error(f"[ImagePipeline] DeepFace error on {image_url}: {e}", exc_info=True)
            faces, dimensions = [], None

        logger.info(f"[ImagePipeline]   Faces detected: {faces}")

        if faces and len(faces) > 0:
            # LOG each face in detail
            for idx, f in enumerate(faces):
                logger.info(
                    f"[ImagePipeline]   Face {idx}: gender={f.get('gender')} "
                    f"raw_scores={f.get('raw_scores')} "
                    f"confidence={f.get('confidence')} "
                    f"prominence={f.get('prominence')}"
                )

            image_status, prominence_score = female_faces(faces)
            logger.info(f"[ImagePipeline]   → female_faces() returned: {image_status}, prominence={prominence_score}")

        else:
            logger.info("[ImagePipeline]   → No faces returned")
            image_status = "no_face"
            prominence_score = 0

        art["image_analysis"][image_url] = {
            "status": image_status,
            "prominence_score": prominence_score,
            "raw_faces": faces,
        }

        current_priority = (
            valid_statuses.index(image_status)
            if image_status in valid_statuses
            else default_priority
        )

        # First image auto-selected
        if not art["most_relevant_image"]:
            logger.info(f"[ImagePipeline]   Selecting as initial best image")
            art["most_relevant_status"] = image_status
            art["most_relevant_image"] = image_url
            highest_priority = current_priority
            highest_prominence = prominence_score
            continue

        # Compare priority
        if (
            current_priority < highest_priority or
            (current_priority == highest_priority and prominence_score > highest_prominence)
        ):
            logger.info(
                f"[ImagePipeline]   Replacing previous best: "
                f"{art['most_relevant_image']} (status={art['most_relevant_status']}) "
                f"→ {image_url} (status={image_status})"
            )
            art["most_relevant_status"] = image_status
            art["most_relevant_image"] = image_url
            highest_priority = current_priority
            highest_prominence = prominence_score

    if not art.get("most_relevant_image") and image_list:
        art["most_relevant_image"] = image_list[0]
        logger.info(f"[ImagePipeline] Fallback: selected first image: {image_list[0]}")

def fetch_newsapi(cfg, name, qcfg, newsapi):
    desired = qcfg.get("page_size", 100)
    per_page = min(desired, 100)
    pages = math.ceil(desired / per_page)
    raw_articles = []
    for page in range(1, pages + 1):
        logger.info(f"[{name}] NewsAPI fetch page {page}/{pages}")
        fetch_pause("newsapi", cfg)
        resp = newsapi.get_everything(
            q=qcfg["q"],
            language=cfg["newsapi"]["language"],
            page_size=per_page,
            page=page
        )
        batch = resp.get("articles", [])
        logger.debug(f"[{name}]  → got {len(batch)} articles")
        raw_articles.extend(batch)
    logger.info(f"[{name}] Total fetched: {len(raw_articles)} articles")
    return raw_articles



def female_faces(faces) -> tuple[str, float]:
    """
    Analyzes the gender distribution in a list of face data and returns the image status and prominence.

    Args:
        faces (list): A list of face dictionaries, each containing information like 'gender',
                      'confidence', and 'prominence'.

    Returns:
        tuple:
            - status (str): Status of the image (e.g., "female_majority", "female_prominent",
                            "no_face", "no_female_faces", etc.).
            - prominence (float): Average prominence of relevant faces or the prominence
                                            of the most relevant face if applicable; otherwise, 0.
    """
    # Ensure 'faces' is a list
    if not isinstance(faces, list):
        logger.error(f"Unexpected faces type: {type(faces)}")
        return "error_invalid_faces", 0

    # Ensure all elements in 'faces' are dictionaries
    faces = [f for f in faces if isinstance(f, dict)]

    # Filter out female faces with sufficient confidence
    female_faces = [
        f for f in faces
        if f.get("gender") == "Woman" and f.get("confidence", 0) >= FEMALE_CONFIDENCE_THRESHOLD
    ]

    if not female_faces:
        # No confident female faces found
        return "no_female_faces", 0

    total_faces = len(faces)
    num_female = len(female_faces)

    # Determine the status of the image
    if total_faces > 2 and num_female > total_faces / 2:
        # More than half of the faces are women
        average_prominence = sum(f["prominence"] for f in female_faces) / num_female
        return "female_majority", average_prominence

    elif faces:
        # Fallback: Check the most prominent face
        best_face = max(faces, key=lambda f: f.get("prominence", 0))
        if best_face["gender"] == "Woman" and best_face["confidence"] >= FEMALE_CONFIDENCE_THRESHOLD:
            return "female_prominent", best_face["prominence"]
        else:
            # Single most prominent face is not a confident woman
            return "dropped_male_or_low_confidence", 0

    else:
        # No faces detected
        return "no_face", 0

def guess_genders(all_persons, gndr):
    try:
        gender_responses = gndr.get(list(all_persons))
        gender_map = {g["name"]: g.get("gender") for g in gender_responses}
        logger.debug(f"Genderize API response: {gender_map}")
    except GenderizeException as e:
        logger.warning(f"Genderize API error (falling back): {e}")
        if LOCAL_GENDER_AVAILABLE:
            logger.info("Using local gender-guesser fallback")
            gender_map = {}
            for name in all_persons:
                first = name.split()[0]
                g = local_detector.get_gender(first)
                gender_map[name] = "female" if g in ("female", "mostly_female") else "male" if g in (
                "male", "mostly_male") else None
            logger.debug(f"Local gender-map: {gender_map}")
        else:
            logger.error(
                "Local gender-guesser not installed; all names will be treated as unknown"
            )
            gender_map = {}
    return gender_map


def gather_bodies_images_and_persons(
        raw_articles: list[dict],
        nlp,
        max_images: int = 3
) -> tuple[list[str], list[list[str]], set[str]]:
    """
    For each article in raw_articles:
      - fetch its full text and images (fetch_full_text_and_images)
      - fall back to title/description if no text
      - extract PERSON entities via spaCy

    Args:
        raw_articles: A list of dictionaries containing article metadata (e.g., URLs, titles, descriptions).
        nlp: The spaCy NLP pipeline to process extracted text for entities.
        max_images: The maximum number of image URLs to retrieve per article.

    Returns:
        A tuple containing:
          - bodies: A list of article text bodies.
          - images: A list of lists of image URLs (per article).
          - persons: A set of all PERSON entities detected across all articles.
    """
    # Explicit type hints for local variables
    bodies: list[str] = []  # List of article text bodies
    images: list[list[str]] = []  # List of lists of image URLs (per article)
    persons: set[str] = set()  # Set of PERSON entities

    for art in raw_articles:
        url = art.get("url", "")

        # 1) Fetch text + multiple images using the helper function
        body, image_urls = fetch_full_text_and_images(url, max_images=max_images)

        # 2) Fallback if text extraction failed
        if not body:
            body = " ".join(filter(None, [
                art.get("title", ""),
                art.get("description", "")
            ]))

        # Append the body and the list of image URLs
        bodies.append(body)
        images.append(image_urls if image_urls else [])  # Add an empty list if no images found

        # 3) Extract PERSON entities from the text using spaCy NLP
        doc = nlp(body)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                persons.add(ent.text)

    return bodies, images, persons

# ─── Formatting ─────────────────────────────────────────────────────────


# ─── Main Entrypoint ───────────────────────────────────────────────────

def main(config_path: str = "config.yaml", output: str = None, fmt: str = "text", log_level: str = "INFO",
         use_cache: bool = False, new_today: bool = False, expire_days: int = None):
    setup_logging(log_level)
    cfg = load_config(config_path)

    try:
        res = fetch_and_filter(cfg, use_cache, new_today, expire_days)  # Pass flags to fetch_and_filter

        # Extract metadata (query strings) for HTML generation
        metadata = {section["name"]: section["q"] for section in cfg["queries"] if "name" in section and "q" in section}
        if fmt == "html":
            out = generate_html(res, metadata, cfg=cfg)
        else:  # Use plain text formatter for other formats
            out = generate_text(res, metadata)

        # Determine output file format
        if output is None:
            output = "news.html" if fmt == "html" else "news.txt"

        if output.lower() == "console":
            print(out)
        else:
            with open(output, "w", encoding="utf-8") as f:
                f.write(out)
            logger.info(f"Saved results to {output}")
    finally:
        # Always finalize timing tracker to write final summary
        tracker = get_timing_tracker()
        tracker.finalize()
        logger.info("Timing summary written to timing.log")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fetch/filter news by female leadership")
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("-o", "--output", help="Console or filepath")
    p.add_argument("-f", "--format", choices=["text", "html"], default="text")
    p.add_argument("-l", "--log", default="INFO")
    p.add_argument(
        "--use-cache", "-C",
        action="store_true",
        help="If set, load Mediastack queries from cache/<query_name>.json instead of calling API"
    )
    p.add_argument(
        "--new-today", "-N",
        action="store_true",
        help="If set, only process articles not seen in the cache (compares to all previously cached articles)"
    )
    p.add_argument(
        "--expire-older-than", "-E",
        type=int,
        metavar="DAYS",
        help="Expire cache entries older than the specified number of days"
    )
    a = p.parse_args()
    main(
        config_path=a.config,
        output=a.output,
        fmt=a.format,
        log_level=a.log,
        use_cache=a.use_cache,
        new_today=a.new_today,
        expire_days=a.expire_older_than
    )
