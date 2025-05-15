import os
import yaml
import logging
import spacy
from newsapi import NewsApiClient
from genderize import Genderize
from openai import OpenAI
import openai as openai_pkg
import argparse
import requests
from article_fetcher import fetch_full_text

# ─── Logging Setup ────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO"):
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=numeric_level
    )
    logger.debug(f"Logging initialized at {level} level")

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

# ─── OpenAI Helpers ────────────────────────────────────────────────────

def classify_leadership(text: str, cfg: dict, client: OpenAI) -> bool:
    prompt = cfg["prompts"]["classification"] + "\n\n" + text
    logger.debug("Running zero-shot classification for leadership")
    try:
        resp = client.chat.completions.create(
            model=cfg["openai"]["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg["openai"].get("temperature", 0),
            max_tokens=1
        )
        answer = resp.choices[0].message.content.strip().lower()
        logger.debug(f"Classification result: {answer}")
        return answer == "yes"
    except openai_pkg.error.OpenAIError as e:
        logger.warning(f"OpenAI classification error: {e}")
        return False


def summarize(text: str, cfg: dict, client: OpenAI) -> str:
    prompt = cfg["prompts"]["summarize"].format(text=text)
    logger.debug("Requesting summary from OpenAI")
    try:
        resp = client.chat.completions.create(
            model=cfg["openai"]["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg["openai"].get("temperature", 0),
            max_tokens=60
        )
        summary = resp.choices[0].message.content.strip()
        logger.info("Received summary")
        return summary
    except openai_pkg.error.OpenAIError as e:
        logger.warning(f"OpenAI summarization error: {e}")
        return text


def spin_genders(text: str, cfg: dict, client: OpenAI) -> str:
    prompt = cfg["prompts"]["spin_genders"].format(text=text)
    logger.debug("Requesting spin-genders rewrite from OpenAI")
    try:
        resp = client.chat.completions.create(
            model=cfg["openai"]["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg["openai"].get("temperature", 0),
            max_tokens=200
        )
        rewrite = resp.choices[0].message.content.strip()
        logger.info("Received spin-genders rewrite")
        return rewrite
    except openai_pkg.error.OpenAIError as e:
        logger.warning(f"OpenAI spin-genders error: {e}")
        return text

# ─── Mediastack Fetcher ─────────────────────────────────────────────────

def fetch_mediastack(cfg: dict, qcfg: dict) -> list:
    base_url = cfg["mediastack"].get("base_url")
    # Normalize keywords
    raw_kw = qcfg.get("q")
    if isinstance(raw_kw, list):
        keywords = ",".join(raw_kw)
    else:
        keywords = raw_kw or ""

    # Build params dict
    params = {
        "access_key": cfg["mediastack"]["access_key"],
        "limit": qcfg.get("page_size", 100)
    }
    if keywords:
        params["keywords"] = keywords
    # Optional countries filter (ISO 2-letter codes)
    country_code = qcfg.get("countries")
    if country_code:
        params["countries"] = country_code

    # Debug: show request parameters
    logger.debug(f"Mediastack query params: {params}")
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        # Debug: log full URL
        logger.debug(f"Mediastack request URL: {resp.url}")
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("data", [])
        if not articles:
            logger.error(f"Mediastack returned zero articles with params: {params}")
        else:
            logger.info(f"Fetched {len(articles)} articles from Mediastack")
        return articles
    except Exception as e:
        logger.error(f"Mediastack fetch failed: {e}")
        return []

# ─── Fetch & Filter ────────────────────────────────────────────────────

def fetch_and_filter(cfg: dict) -> dict:
    setup_logging(cfg.get("log_level", "INFO"))
    newsapi = NewsApiClient(api_key=cfg.get("newsapi", {}).get("api_key"))
    client = OpenAI(api_key=cfg["openai"]["api_key"])
    nlp = spacy.load("en_core_web_sm")
    gndr = Genderize()

    results = {}
    for qcfg in cfg.get("queries", []):
        name = qcfg.get("name")
        logger.info(f"Processing query '{name}'")

        # Provider
        if qcfg.get("provider") == "mediastack":
            raw = fetch_mediastack(cfg, qcfg)
            # Local filter by country if provided (free plan fallback)
            if qcfg.get("countries"):
                before = len(raw)
                raw = [a for a in raw if a.get("country") == qcfg.get("countries")]
                logger.debug(f"Filtered {before - len(raw)} articles by country '{qcfg.get('countries')}'")
            articles = [
                {"title": a.get("title", ""), "description": a.get("description", ""), "url": a.get("url", "")}
                for a in raw
            ]
        else:
            try:
                resp = newsapi.get_everything(
                    q=qcfg.get("q"),
                    language=cfg["newsapi"].get("language", "en"),
                    page_size=qcfg.get("page_size", 100)
                )
                articles = resp.get("articles", [])
                logger.debug(f"Fetched {len(articles)} from NewsAPI")
            except Exception as e:
                logger.error(f"NewsAPI fetch failed: {e}")
                articles = []

        # Batch gender detection
        names = set()
        for art in articles:
            text = " ".join(filter(None, [art.get("title"), art.get("description")]))
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    names.add(ent.text)
        try:
            genders = gndr.get(list(names))
            gender_map = {g["name"]: g.get("gender") for g in genders}
        except Exception as e:
            logger.warning(f"Genderize batch error: {e}")
            gender_map = {}

        hits = []
        for art in articles:
            url = art.get("url", "")
            full_text = fetch_full_text(url) if qcfg.get("classification") else ""
            body = full_text or " ".join(filter(None, [art.get("title"), art.get("description")]))

            status = "original"
            if qcfg.get("classification"):
                doc = nlp(body)
                persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
                female_names = [p for p in persons if gender_map.get(p) == "female"]
                has_kw = any(kw.lower() in body.lower() for kw in cfg.get("leadership_keywords", []))
                if female_names and has_kw and classify_leadership(body, cfg, client):
                    status = "female_leader"
                else:
                    action = qcfg.get("fallback", "exclude")
                    if action == "show-full":
                        status = "show-full"
                    elif action == "summarize":
                        art["snippet"] = summarize(body, cfg, client)
                        status = "summarized"
                    elif action == "spin-genders":
                        art["snippet"] = spin_genders(body, cfg, client)
                        status = "spun"
                    else:
                        continue

            art["status"] = status
            art["full_text"] = full_text
            hits.append(art)

        logger.info(f"{len(hits)} articles retained for '{name}'")
        results[name] = hits

    return results

# ─── Formatting ─────────────────────────────────────────────────────────

def format_text(results: dict) -> str:
    lines = []
    for name, articles in results.items():
        lines.append(f"=== {name} ===")
        for art in articles:
            status = art.get("status", "original")
            title = art.get("title", "No title")
            url = art.get("url", "")
            snippet = art.get("snippet", art.get("description", ""))
            body = art.get("full_text", "")
            content = snippet or body
            lines.append(f"- [{status}] {title}\n  {url}\n  {content}\n")
    return "\n".join(lines)


def generate_html(results: dict) -> str:
    html = ["<html><head><meta charset='utf-8'><title>News</title></head><body>"]
    for name, articles in results.items():
        html.append(f"<h2>{name}</h2><ul>")
        for art in articles:
            status = art.get("status", "original")
            title = art.get("title", "No title")
            url = art.get("url", "#")
            snippet = art.get("snippet", art.get("description", ""))
            body = art.get("full_text", "")
            content = snippet or body
            html.append(
                f"<li><strong>[{status}]</strong> <a href='{url}'>{title}</a><p>{content}</p></li>"
            )
        html.append("</ul>")
    html.append("</body></html>")
    return "\n".join(html)

# ─── Main Entrypoint ───────────────────────────────────────────────────

def main(
    config_path: str = "config.yaml",
    output: str = None,
    fmt: str = "text",
    log_level: str = "INFO"
):
    setup_logging(log_level)
    cfg = load_config(config_path)
    results = fetch_and_filter(cfg)
    content = generate_html(results) if fmt == "html" else format_text(results)

    if output is None:
        output = "news.html" if fmt == "html" else "news.txt"
    if output.lower() == "console":
        print(content)
    else:
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved results to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch & filter news by female leadership")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
    parser.add_argument("-o", "--output", help="Console or output filepath")
    parser.add_argument("-f", "--format", choices=["text", "html"], default="text", help="Output format")
    parser.add_argument("-l", "--log", default="INFO", help="Logging level")
    args = parser.parse_args()
    main(config_path=args.config, output=args.output, fmt=args.format, log_level=args.log)
