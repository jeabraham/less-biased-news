import os
import yaml
import logging
import spacy
import requests
from newsapi import NewsApiClient
from genderize import Genderize
from openai import OpenAI

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
    logger.debug(f"Attempting to load config from '{path}'")
    if not os.path.exists(path):
        logger.error(f"Config file not found: {path}")
        raise FileNotFoundError(f"{path} does not exist")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("Configuration loaded successfully")
    return cfg


# ─── OpenAI Helpers ────────────────────────────────────────────────────
def classify_leadership(text: str, cfg: dict) -> bool:
    prompt = cfg["prompts"]["classification"] + "\n\n" + text
    logger.debug("Running zero-shot classification for leadership")
    resp = client.chat.completions.create(model=cfg["openai"]["model"],
    temperature=cfg["openai"]["temperature"],
    max_tokens=1,
    messages=[{"role": "user", "content": prompt}])
    answer = resp.choices[0].message.content.strip().lower()
    logger.debug(f"Classification result: '{answer}'")
    return answer == "yes"

def summarize(text: str, cfg: dict) -> str:
    p = cfg["prompts"]["summarize"].format(text=text)
    logger.debug("Requesting summary from OpenAI")
    resp = client.chat.completions.create(model=cfg["openai"]["model"],
    temperature=cfg["openai"]["temperature"],
    max_tokens=60,
    messages=[{"role": "user", "content": p}])
    summary = resp.choices[0].message.content.strip()
    logger.info("Received summary")
    return summary

def spin_genders(text: str, cfg: dict) -> str:
    p = cfg["prompts"]["spin_genders"].format(text=text)
    logger.debug("Requesting gender-spin rewrite from OpenAI")
    resp = client.chat.completions.create(model=cfg["openai"]["model"],
    temperature=cfg["openai"]["temperature"],
    max_tokens=200,
    messages=[{"role": "user", "content": p}])
    spun = resp.choices[0].message.content.strip()
    logger.info("Received spin-genders rewrite")
    return spun


# ─── Fetch & Filter ────────────────────────────────────────────────────
def fetch_and_filter(cfg: dict) -> dict:
    logger.info("Initializing NewsAPI client")
    newsapi = NewsApiClient(api_key=cfg["newsapi"]["api_key"])
    logger.info("Loading spaCy model")
    nlp = spacy.load("en_core_web_sm")
    gndr = Genderize()

    results = {}
    for qcfg in cfg["queries"]:
        name = qcfg["name"]
        logger.info(f"Running query '{name}'")
        try:
            resp = newsapi.get_everything(
                q=qcfg["q"],
                language=cfg["newsapi"]["language"],
                page_size=qcfg["page_size"]
            )
        except Exception as e:
            logger.error(f"NewsAPI query failed for '{name}': {e}")
            continue

        hits = []
        articles = resp.get("articles", [])
        logger.debug(f"Fetched {len(articles)} articles for '{name}'")

        for art in articles:
            text = " ".join(filter(None, [art.get("title"), art.get("description")]))
            if not qcfg["classification"]:
                hits.append(art)
                continue

            # Pre-filter
            doc = nlp(text)
            people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            genders = gndr.get(people) if people else []
            female_names = {g["name"] for g in genders if g.get("gender") == "female"}
            has_keyword = any(
                kw.lower() in text.lower()
                for kw in cfg["leadership_keywords"]
            )

            is_female_leader = False
            if female_names and has_keyword:
                is_female_leader = classify_leadership(text, cfg)

            if is_female_leader:
                logger.debug(f"Article kept (female leader): {art.get('title')}")
                hits.append(art)
            else:
                action = qcfg["fallback"]
                logger.debug(f"Applying fallback '{action}' for article: {art.get('title')}")
                if action == "show-full":
                    hits.append(art)
                elif action == "summarize":
                    art["snippet"] = summarize(text, cfg)
                    hits.append(art)
                elif action == "spin-genders":
                    art["snippet"] = spin_genders(text, cfg)
                    hits.append(art)
                # exclude == no action

        logger.info(f"{len(hits)} articles after filtering for '{name}'")
        results[name] = hits

    return results


# ─── Output Formatting ─────────────────────────────────────────────────
def format_text(results: dict) -> str:
    logger.debug("Formatting results as plain text")
    lines = []
    for name, articles in results.items():
        lines.append(f"=== {name} ===")
        for art in articles:
            title = art.get("title", "No title")
            url   = art.get("url", "")
            snippet = art.get("snippet", art.get("description", ""))
            lines.append(f"- {title}\n  {url}\n  {snippet}\n")
    return "\n".join(lines)

def generate_html(results: dict) -> str:
    logger.debug("Generating HTML output")
    html = ["<html><head><meta charset='utf-8'><title>News</title></head><body>"]
    for name, articles in results.items():
        html.append(f"<h2>{name}</h2><ul>")
        for art in articles:
            title = art.get("title", "No title")
            url   = art.get("url", "#")
            snippet = art.get("snippet", art.get("description", ""))
            html.append(f"<li><a href='{url}'>{title}</a><p>{snippet}</p></li>")
        html.append("</ul>")
    html.append("</body></html>")
    return "\n".join(html)


# ─── Main Entrypoint ───────────────────────────────────────────────────
def main(
    config_path: str = "config.yaml",
    output: str = "news.txt",
    fmt: str = "text",
    log_level: str = "INFO"
):
    setup_logging(log_level)
    logger.info("Starting news_filter module")
    cfg = load_config(config_path)
    results = fetch_and_filter(cfg)

    # Initialize the OpenAI client
    client = OpenAI(api_key=cfg["openai"]["api_key"])

    if fmt == "html":
        content = generate_html(results)
    else:
        content = format_text(results)

    if output == "console":
        print(content)
        logger.info("Printed results to console")
    else:
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved results to '{output}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and filter news by female leadership")
    parser.add_argument("--config",   "-c", default="config.yaml", help="Path to config file")
    parser.add_argument("--output",   "-o", default="news.txt",
                        help="'console' or filename to save results")
    parser.add_argument("--format",   "-f", choices=["text","html"], default="text",
                        help="Output format")
    parser.add_argument("--log",      "-l", default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    main(config_path=args.config, output=args.output, fmt=args.format, log_level=args.log)
