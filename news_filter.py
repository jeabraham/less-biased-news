import os
import yaml
import logging
import spacy
from newsapi import NewsApiClient
from genderize import Genderize, GenderizeException
import openai as openai_pkg
import argparse
import requests
from article_fetcher import fetch_full_text_and_images
from typing import List, Tuple, Set
from analyze_image_gender import analyze_image_gender, FEMALE_CONFIDENCE_THRESHOLD
from datetime import date, timedelta, datetime
import time
import json
from typing import Optional, Dict


import tiktoken

from openai_utils import (
    truncate_for_openai,
    classify_leadership,
    short_summary,
    clean_summary,
    spin_genders,
)

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
    keywords = ",".join(raw_kw) if isinstance(raw_kw,list) else (raw_kw or "")
    params = {"access_key":cfg["mediastack"]["access_key"],"limit":qcfg.get("page_size",100)}

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
    if qcfg.get("languages"):  # Support for languages parameter
        params["languages"] = qcfg.get("languages")  # Add languages to params

    logger.debug(f"Mediastack query params: {params}")
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        logger.debug(f"Mediastack request URL: {resp.url}")
        resp.raise_for_status()
        articles = resp.json().get("data",[])
        if not articles:
            logger.error(f"Mediastack returned zero articles with params: {params}")
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
        return []

# ─── Fetch & Filter ────────────────────────────────────────────────────
import math

def fetch_and_filter(cfg: dict, use_cache: bool = False) -> dict:
    logger.info("Initializing NewsAPI client")
    newsapi = NewsApiClient(api_key=cfg["newsapi"]["api_key"])
    logger.info("Loading spaCy model")
    nlp = spacy.load("en_core_web_sm")
    gndr = Genderize()
    openai_pkg.api_key = cfg["openai"]["api_key"]

    max_images = cfg.get("max_images", 3)  # Default to 3 if not specified in the YAML file

    results = {}
    for qcfg in cfg["queries"]:
        name = qcfg["name"]
        logger.info(f"Processing query '{name}'")
        # ─── fetch raw articles via the right source ────────────────
        provider = qcfg.get("provider", "newsapi").lower()
        if provider == "mediastack":
          # returns list of { title, url, description, etc. }
          raw_articles = fetch_mediastack(cfg, qcfg, use_cache=use_cache)
          logger.info(f"Fetched {len(raw_articles)} articles from Mediastack")
        else:
            raw_articles = fetch_newsapi(cfg, name, qcfg, newsapi)

        # 1) Extract all person names across all articles
        bodies, images, all_persons = gather_bodies_images_and_persons(raw_articles, nlp, max_images=max_images)

        logger.debug(f"Batch Genderize request names ({len(all_persons)}): {all_persons}")
        # 2) One shot gender detection
        gender_map = guess_genders(all_persons, gndr)

        hits = []
        stats = {"total": len(raw_articles), "persons": 0, "female_names": 0,
                 "keyword_hits": 0, "classified_ok": 0}

        summarize_selected = qcfg.get(
            "summarize_selected",
            cfg.get("default_summarize_selected", False)
        )

        # 3) Process each article
        for art, body, image_list in zip(raw_articles, bodies, images):

            # 1) Attach the raw image URLs (list of images for each article)
            process_article_images(art, image_list)

            # 3) Continue processing with spaCy to extract PERSON entities
            doc = nlp(body)
            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            stats["persons"] += bool(persons)  # Increment PERSON counter if any entities are found

            is_female_leader = identify_female_leadership(art, body, cfg, gender_map, persons, stats)

            # 6) Update article status based on classification and image-based fallback logic
            if is_female_leader:
                art["status"] = "female_leader"
                stats["classified_ok"] += 1
            else:
                art["status"] = qcfg["fallback"]
                # Existing fallback logic (show-full, short_summary, spin-genders, exclude)
                # You can log rejection reasons here

            if art["status"] == "female_leader":
                # If the article is classified as about a female leader, handle content summarization
                if summarize_selected:
                    logger.info(f"Clean-summarizing '{art['title']}'")
                    art["content"] = clean_summary(body, cfg, openai_pkg, leader_name=art.get("leader_name"))
                else:
                    # Keep the full fetched body
                    art["content"] = body
            else:
                # Handle fallback logic for image statuses
                img_stat = art.get("most_relevant_status", "")

                art["status"] = qcfg.get("fallback_image_male", qcfg["fallback"])

                if img_stat in ("female", "female_majority", "female_prominent"):
                    art["status"] = qcfg.get("fallback_image_female", qcfg["fallback"])
                    logger.info(f"Applying image‐based fallback '{art['status']}' for article '{art['title']}'")
                if art["status"] == "show-full":
                    # Summarize if necessary, or keep the full content
                    if summarize_selected:
                        logger.info(f"Clean-summarizing '{art['title']}'")
                        art["content"] = clean_summary(body, cfg, openai_pkg)
                    else:
                        art["content"] = body
                elif art["status"] == "short_summary":
                    # Generate a short summary
                    art["content"] = short_summary(body, cfg, openai_pkg)
                elif art["status"] == "spin-genders":
                    # Apply gender-spin logic
                    art["content"] = spin_genders(body, cfg, openai_pkg)
                else:
                    # Default to the article's description, but this article will usually be excluded anyways.
                    art["content"] = art.get("description", "")

            # Add the processed article to the results hits
            hits.append(art)

        logger.info(
            f"Stats for '{name}': total={stats['total']}, "
            f"persons={stats['persons']}, female_names={stats['female_names']}, "
            f"keyword_hits={stats['keyword_hits']}, classified_ok={stats['classified_ok']}"
        )

        results[name] = hits

    return results


def identify_female_leadership(art, body, cfg, gender_map, persons, stats):
    # 4) Analyze detected PERSON names for female associations
    female_names = [p for p in persons if gender_map.get(p) == "female"]
    has_kw = any(
        kw.lower() in body.lower()
        for kw in cfg.get("leadership_keywords", [])
    )
    if female_names:
        stats["female_names"] += 1
    if has_kw:
        stats["keyword_hits"] += 1
    # 5) Check if the article pertains to a female leader
    is_female_leader = False
    if female_names and has_kw:
        try:
            is_female_leader, leader_name = classify_leadership(body, cfg, openai_pkg)
            art["leader_name"] = leader_name
        except Exception as e:
            logger.warning(f"OpenAI classification error on '{art.get('title')}': {e}")
    return is_female_leader


def process_article_images(art, image_list):
    """
    Process a list of images associated with an article and determine the overall image status.

    Args:
        art (dict): The article dictionary where the results will be stored.
        image_list (list): A list of image URLs to process.

    Returns:
        None
    """
    art["image_urls"] = image_list
    art["most_relevant_status"] = "no_face"
    art["most_relevant_image"] = None
    art["image_analysis"] = {}  # Store prominence information for all images

    # Define valid statuses and their implicit priority
    valid_statuses = ["female", "female_majority", "female_prominent"]
    default_priority = len(valid_statuses)  # Use a high index for unexpected statuses

    for image_url in image_list:
        faces, dimensions = analyze_image_gender(image_url)
        if (faces and len(faces) > 0):
            image_status, prominence_score = female_faces(faces)  # Fetch details about the image
        else:
            image_status = "no_face"
            prominence_score = 0

        # Store analysis details for the image
        art["image_analysis"][image_url] = {
            "status": image_status,
            "prominence_score": prominence_score,
        }

        # Determine the current image's priority
        current_priority = (
            valid_statuses.index(image_status) if image_status in valid_statuses else default_priority
        )

        # Update the article's "most relevant" image based on priority and prominence score
        if not art["most_relevant_image"]:
            # If no image selected yet, choose the current one
            art["most_relevant_status"] = image_status
            art["most_relevant_image"] = image_url
            highest_priority = current_priority
            highest_prominence = prominence_score
        else:
            # Check if the current image is more relevant
            if (
                    current_priority < highest_priority  # Higher-priority status
                    or (current_priority == highest_priority and prominence_score > highest_prominence)
                    # Tie-breaking by prominence
            ):
                art["most_relevant_status"] = image_status
                art["most_relevant_image"] = image_url
                highest_priority = current_priority
                highest_prominence = prominence_score

    # Fallback if no relevant image is found
    if not art.get("most_relevant_image") and image_list:
        art["most_relevant_image"] = image_list[0]

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
                gender_map[name] = "female" if g in ("female", "mostly_female") else None
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

def format_text(results: dict, metadata: dict = None) -> str:
    """
    Generate a text summary from the filtered news results.

    Enhancements:
    - Formats section headers more clearly and consistently.
    - Adds query strings (metadata) below each section header, if available.
    - Improves layout and readability with consistent indentation and markers.
    - Avoids printing empty or useless information.

    Parameters:
    ----------
    results : dict
        Dictionary of results, where keys are query names and values are lists of articles.
    metadata : dict, optional
        A dictionary containing query-specific metadata such as query strings.

    Returns:
    --------
    str
        A structured and readable text output.
    """
    # Initialize output lines
    lines = []
    metadata = metadata or {}

    # Iterate through the results grouped by query name
    for name, articles in results.items():
        # Header for section (query name)
        lines.append("=" * 60)  # Divider line
        lines.append(f"=== {name} ===")  # Section header

        # Include query string metadata if available
        query_string = metadata.get(name, "Unknown query")
        lines.append(f"Query: {query_string}")
        lines.append("-" * 60)  # Sub-divider for better readability

        # Iterate through the articles in each section
        for art in articles:
            # Skip excluded articles
            if art.get("status") == "exclude":
                continue

            # Extract article details
            title = art.get("title", "No title")
            url = art.get("url", "#")
            status = art.get("status", "")
            leader_name = art.get("leader_name", "")
            if leader_name:
                status += f" ({leader_name})"  # Append leader name to status if available
            content = art.get("content") or art.get("description") or "[No content available]"

            # Print article info
            lines.append(f"- {title}  [{status}]")
            lines.append(f"  Link: {url}")

            # Add content in bullet-pointed paragraphs
            for paragraph in content.split("\n\n"):
                paragraph = paragraph.strip()
                if paragraph:  # Avoid blank paragraphs
                    lines.append(f"    {paragraph}")

            # Add a blank line after each article for separation
            lines.append("")

        # Add a space between sections for readability
        lines.append("")

    # Join the lines into a single text output
    return "\n".join(lines)


def generate_html(results: dict, metadata: dict = None) -> str:
    """
    Generate an enhanced HTML page from the filtered news results.

    Enhances the HTML output as follows:
    - Adds a Table of Contents (TOC) linking to each query's results.
    - Includes the current date and time at the top of the HTML.
    - Displays the "Processed by less_biased_news" credit with a GitHub link.
    - Displays query strings below each header for context.

    Parameters:
    ----------
    results : dict
        Dictionary of results, where keys are query names and values are lists of articles.
    metadata : dict, optional
        A dictionary containing additional metadata such as `query_strings`.

    Returns:
    --------
    str
        A complete HTML string.
    """
    logger.debug("Generating HTML output")

    # Adding optional metadata for query strings
    metadata = metadata or {}

    # Get the current date and time
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Initialize the HTML output
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "  <head>",
        "    <meta charset='utf-8'>",
        "    <title>News</title>",
        "    <style>",
        "      body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0 15px; }",
        "      h1, h2 { color: #333; }",
        "      ul { list-style-type: disc; margin-left: 20px; }",
        "      hr { border: 0; height: 1px; background: #ccc; margin: 20px 0; }",
        "      .toc { margin-bottom: 20px; }",
        "    </style>",
        "  </head>",
        "  <body>"
    ]

    # Add page header with date and credit
    html.extend([
        "    <h1>News Summary</h1>",
        f"    <p><strong>Date:</strong> {current_date}</p>",
        "    <p><strong>Processed by:</strong> "
        "<a href='https://github.com/jeabraham/less-biased-news' target='_blank'>less_biased_news</a></p>",
        "    <hr>"
    ])

    # Generate the Table of Contents (TOC)
    html.append("    <div class='toc'>")
    html.append("      <h2>Table of Contents</h2>")
    html.append("      <ul>")
    for query_name in results:
        section_id = query_name.replace(" ", "_").lower()  # Generate HTML-safe IDs
        html.append(f"        <li><a href='#{section_id}'>{query_name}</a></li>")
    html.append("      </ul>")
    html.append("    </div>")

    # Add each section for query results
    for query_name, articles in results.items():
        section_id = query_name.replace(" ", "_").lower()  # ID for linking
        query_string = metadata.get(query_name, "Not provided")  # Get query string from metadata

        html.append(f"    <section id='{section_id}'>")
        html.append(f"      <h2>{query_name}</h2>")
        html.append(f"      <p><strong>Query:</strong> {query_string}</p>")
        html.append("      <ul>")

        for art in articles:
            if art.get("status") == "exclude":
                continue  # Skip excluded articles
            title = art.get("title", "No title")
            url = art.get("url", "#")
            status = art.get("status", "")
            if art.get("leader_name"):
                status += f" ({art['leader_name']})"  # Append leader name to status
            content = art.get("content") or art.get("description") or ""

            html.append("        <li>")
            html.append(f"          [{status}] <a href='{url}' target='_blank'>{title}</a>")

            # Include the most relevant image if available
            if art.get("most_relevant_image"):
                image_url = art["most_relevant_image"]
                image_status = art.get("most_relevant_status", "")

                # Determine image size based on its status
                if image_status in ("female", "female_majority", "female_prominent"):
                    base = 300
                elif image_status == "no_face":
                    base = 200
                else:  # male or less relevant images
                    base = 100

                # Double size if the article is about a female leader
                if art.get("status") == "female_leader":
                    final = base * 2
                else:
                    final = base

                size_style = f"max-width:{final}px;"
                html.append(
                    "<figure>"
                    f"<img src='{image_url}' alt='' "
                    f"style='{size_style} display:block; margin:0.5em 0;'>"
                    "</figure>"
                )

            # Process and format content into paragraphs
            paragraphs = []
            for block in content.split("\n\n"):
                for para in block.split("\n"):
                    text = para.strip()
                    if text:
                        paragraphs.append(text)

            for para in paragraphs:
                html.append(f"          <p>{para}</p>")

            html.append("        </li>")

        html.append("      </ul>")
        html.append("    </section>")
        html.append("    <hr>")

    # Close the HTML structure
    html.extend([
        "  </body>",
        "</html>"
    ])

    return "\n".join(html)

# ─── Main Entrypoint ───────────────────────────────────────────────────

def main(config_path:str="config.yaml",output:str=None,fmt:str="text",log_level:str="INFO",use_cache:bool=False):
    setup_logging(log_level)
    cfg=load_config(config_path)
    res = fetch_and_filter(cfg, use_cache)  # Fetch and filter results from API

    # Extract metadata (query strings) for HTML generation
    metadata = {section["name"]: section["q"] for section in cfg if "name" in section and "q" in section}
    if fmt == "html":
        out = generate_html(res, metadata)
    else:  # Use plain text formatter for other formats
        out = format_text(res, metadata)

    # Determine output file format
    if output is None:
        output = "news.html" if fmt == "html" else "news.txt"

    if output.lower() == "console":
        print(out)
    else:
        with open(output, "w", encoding="utf-8") as f:
            f.write(out)
        logger.info(f"Saved results to {output}")

if __name__=="__main__":
    p=argparse.ArgumentParser(description="Fetch/filter news by female leadership")
    p.add_argument("-c","--config",default="config.yaml")
    p.add_argument("-o","--output",help="Console or filepath")
    p.add_argument("-f","--format",choices=["text","html"],default="text")
    p.add_argument("-l","--log",default="INFO")
    p.add_argument(
        "--use-cache", "-C",
        action="store_true",
        help="If set, load Mediastack queries from cache/<query_name>.json instead of calling API"
    )
    a=p.parse_args();
    main(config_path=a.config,output=a.output,fmt=a.format,log_level=a.log,use_cache=a.use_cache)
