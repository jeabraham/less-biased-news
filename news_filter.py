import os
import yaml
import logging
import spacy
from newsapi import NewsApiClient
from genderize import Genderize, GenderizeException
import openai as openai_pkg
import argparse
import requests
from article_fetcher import fetch_full_text_and_image
from typing import List, Tuple, Set
from analyze_image_gender import analyze_image_gender, FEMALE_CONFIDENCE_THRESHOLD
from datetime import date, timedelta, datetime
import time
import json

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

    if keywords: params["keywords"] = keywords

    if qcfg.get("countries"): params["countries"] = qcfg.get("countries")
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
        bodies, images, all_persons = gather_bodies_images_and_persons(raw_articles, nlp)

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
        for art, body, image_url in zip(raw_articles, bodies, images):
            # 1) attach the raw image URL
            art["image_url"] = image_url

            #Set the default status for the image
            art["image_status"] = "not_a_female_story"

            female_faces(art, image_url)

            doc = nlp(body)
            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            stats["persons"] += bool(persons)

            female_names = [p for p in persons if gender_map.get(p) == "female"]
            has_kw = any(
                kw.lower() in body.lower()
                for kw in cfg.get("leadership_keywords", [])
            )
            if female_names:
                stats["female_names"] += 1
            if has_kw:
                stats["keyword_hits"] += 1

            is_female_leader = False
            if female_names and has_kw:
                try:
                    is_female_leader, leader_name = classify_leadership(body, cfg, openai_pkg)
                    art["leader_name"] = leader_name
                except Exception as e:
                    logger.warning(f"OpenAI classification error on '{art.get('title')}': {e}")

            if is_female_leader:
                status = "female_leader"
                stats["classified_ok"] += 1
            else:
                status = qcfg["fallback"]
                # existing fallback logic (show-full, short_summary, spin-genders, exclude)
                # and you can log rejection reasons here

            art["status"] = status
            if status == "female_leader":
                # if this is a female_leader and we want a clean summary…

                if summarize_selected:
                   logger.info(f"Clean-summarizing '{art['title']}'")
                   art["content"] = clean_summary(body, cfg, openai_pkg)
                else:
                   # keep the full fetched body
                   art["content"] = body
            else:
                img_stat = art.get("image_status", "")

                art["status"] = qcfg.get("fallback_image_male", qcfg["fallback"])

                if img_stat in ("female", "female_majority", "female_prominent"):
                    art["status"] = qcfg.get("fallback_image_female", qcfg["fallback"])
                    logger.info(f"Applying image‐based fallback '{art["status"]}' for article '{art['title']}'")
                if art["status"] == "show-full":
                    if summarize_selected:
                        logger.info(f"Clean-summarizing '{art['title']}'")
                        art["content"] = clean_summary(body, cfg, openai_pkg)
                    else:
                        art["content"] = body
                elif art["status"] == "short_summary":
                      art["content"] = short_summary(body, cfg, openai_pkg)
                elif art["status"]  == "spin-genders":
                      art["content"] = spin_genders(body, cfg, openai_pkg)
                else:
                      art["content"] = art.get("description", "")
            hits.append(art)

        logger.info(
            f"Stats for '{name}': total={stats['total']}, "
            f"persons={stats['persons']}, female_names={stats['female_names']}, "
            f"keyword_hits={stats['keyword_hits']}, classified_ok={stats['classified_ok']}"
        )

        results[name] = hits

    return results


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


def female_faces(art, image_url):
    if image_url is None or image_url == "":
        faces = []
    else:
        faces = analyze_image_gender(image_url)
    female_faces = [
        f for f in faces
        if f["gender"] == "Woman" and f["confidence"] >= FEMALE_CONFIDENCE_THRESHOLD
    ]
    total_faces = len(faces)
    num_female = len(female_faces)

    if total_faces > 0 and num_female > total_faces / 2:
        # More than half the faces are women → keep image
        art["image_status"] = "female_majority"
        # record average prominence of female faces if you like
        art["image_prominence"] = sum(f["prominence"] for f in female_faces) / num_female
    elif faces:
        # fallback to the most prominent‐face rule
        best = max(faces, key=lambda f: f["prominence"])
        if best["gender"] == "Woman" and best["confidence"] >= FEMALE_CONFIDENCE_THRESHOLD:
            art["image_status"] = "female_prominent"
            art["image_prominence"] = best["prominence"]
        else:
            # drop it if the single best face isn’t a confident woman
            art["image_status"] = "dropped_male_or_low_confidence"
    else:
        # no faces detected
        art["image_status"] = "no_face"


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
    raw_articles: List[dict],
    nlp,
) -> Tuple[List[str], List[str], Set[str]]:
    """
    For each article in raw_articles:
      - fetch its full text and lead image (fetch_full_text_and_image)
      - fall back to title/description if no text
      - extract PERSON entities via spaCy

    Returns:
      bodies:   [ body1, body2, … ]
      images:   [ image_url1, image_url2, … ]
      persons:  { all detected PERSON names across all bodies }
    """
    bodies: List[str] = []
    images: List[str] = []
    persons: Set[str] = set()

    for art in raw_articles:
        url = art.get("url", "")
        # 1) fetch text + image in one call
        body, image_url = fetch_full_text_and_image(url)

        # 2) fallback if extraction failed
        if not body:
            body = " ".join(filter(None, [
                art.get("title", ""),
                art.get("description", "")
            ]))

        bodies.append(body)
        images.append(image_url or "")

        # 3) collect PERSON entities
        doc = nlp(body)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                persons.add(ent.text)

    return bodies, images, persons


# ─── Formatting ─────────────────────────────────────────────────────────

def format_text(results: dict)->str:
    lines=[]
    for name,arts in results.items():
        lines.append(f"=== {name} ===")
        for art in arts:
            content = art.get("content", art.get("description", ""))
            lines.append(f"- {title}  [{art.get('status', '')}]")
            lines.append(f"  {url}")
            lines.append(f"  {content}\n")
            st=art.get("status","original")
    return "\n".join(lines)


def generate_html(results: dict) -> str:
    """
    Generate an HTML page from the filtered news results.

    Each query name becomes a <h2> header, and each article is an <li>
    including its status tag, link, and content.
    """
    logger.debug("Generating HTML output")

    html = [
        "<!DOCTYPE html>",
        "<html>",
        "  <head>",
        "    <meta charset='utf-8'>",
        "    <title>News</title>",
        "  </head>",
        "  <body>"
    ]

    for query_name, articles in results.items():
        html.append(f"    <h2>{query_name}</h2>")
        html.append("    <ul>")

        for art in articles:
            if art.get("status") == "exclude":
                continue  # skip excluded articles
            title = art.get("title", "No title")
            url = art.get("url", "#")
            status = art.get("status", "")
            if art.get("leader_name"): status.append(f" ({art['leader_name']})")
            content = art.get("content") or art.get("description") or ""

            html.append("      <li>")
            html.append(f"        [{status}] <a href='{url}' target='_blank'>{title}</a>")

            if art.get("image_url"):
                image_status = art.get("image_status", "")
                # 1) Determine base size by detected face
                if image_status in ("female", "female_majority", "female_prominent"):
                    base = 300
                elif image_status == "no_face":
                    base = 200
                else:  # male or dropped images
                    base = 100

                # 2) Double when this is a female_leader
                if art.get("status") == "female_leader":
                    final = base * 2
                else:
                    final = base

                size_style = f"max-width:{final}px;"
                html.append(
                    "<figure>"
                    f"<img src='{art['image_url']}' alt='' "
                    f"style='{size_style} display:block; margin:0.5em 0;'>"
                    "</figure>"
                )

            # split on double newlines first, then single newlines
            paragraphs = []
            for block in content.split("\n\n"):
                for para in block.split("\n"):
                    text = para.strip()
                    if text:
                        paragraphs.append(text)

            for para in paragraphs:
                html.append(f"        <p>{para}</p>")

            html.append("      </li>")

        html.append("    </ul>")

    html.extend([
        "  </body>",
        "</html>"
    ])

    return "\n".join(html)

# ─── Main Entrypoint ───────────────────────────────────────────────────

def main(config_path:str="config.yaml",output:str=None,fmt:str="text",log_level:str="INFO",use_cache:bool=False):
    setup_logging(log_level)
    cfg=load_config(config_path)
    res=fetch_and_filter(cfg, use_cache)
    out=generate_html(res) if fmt=="html" else format_text(res)
    if output is None: output="news.html" if fmt=="html" else "news.txt"
    if output.lower()=="console": print(out)
    else:
        with open(output,"w",encoding="utf-8") as f: f.write(out)
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
