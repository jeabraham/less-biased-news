import os
import yaml
import logging
import spacy
from newsapi import NewsApiClient
from genderize import Genderize
from openai import OpenAI
import openai as openai_pkg
import argparse
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

# ─── Fetch & Filter ────────────────────────────────────────────────────
def fetch_and_filter(cfg: dict) -> dict:
    logger.info("Initializing NewsAPI and OpenAI clients")
    newsapi = NewsApiClient(api_key=cfg["newsapi"]["api_key"])
    client = OpenAI(api_key=cfg["openai"]["api_key"])

    logger.info("Loading spaCy model and initializing Genderize")
    nlp = spacy.load("en_core_web_sm")
    gndr = Genderize()

    results = {}
    for qcfg in cfg.get("queries", []):
        name = qcfg["name"]
        logger.info(f"Running query '{name}'")
        try:
            resp = newsapi.get_everything(
                q=qcfg["q"],
                language=cfg["newsapi"].get("language", "en"),
                page_size=qcfg.get("page_size", 100)
            )
        except Exception as e:
            logger.error(f"NewsAPI query failed for '{name}': {e}")
            continue

        hits = []
        for art in resp.get("articles", []):
            url = art.get("url", "")
            full_text = fetch_full_text(url) if qcfg.get("classification", False) else ""
            base_text = full_text or " ".join(filter(None, [art.get("title"), art.get("description")]))

            if not qcfg.get("classification", False):
                art["full_text"] = full_text
                hits.append(art)
                continue

            doc = nlp(base_text)
            people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            genders = gndr.get(people) if people else []
            female_names = {g["name"] for g in genders if g.get("gender") == "female"}
            has_keyword = any(
                kw.lower() in base_text.lower()
                for kw in cfg.get("leadership_keywords", [])
            )

            is_female_leader = False
            if female_names and has_keyword:
                is_female_leader = classify_leadership(base_text, cfg, client)

            if is_female_leader:
                logger.debug(f"Kept (female leader): {art.get('title')}")
                art["full_text"] = full_text
                hits.append(art)
            else:
                action = qcfg.get("fallback", "exclude")
                logger.debug(f"Fallback '{action}' for article: {art.get('title')}")
                if action == "show-full":
                    art["full_text"] = full_text
                    hits.append(art)
                elif action == "summarize":
                    art["snippet"] = summarize(base_text, cfg, client)
                    hits.append(art)
                elif action == "spin-genders":
                    art["snippet"] = spin_genders(base_text, cfg, client)
                    hits.append(art)

        logger.info(f"{len(hits)} articles after filtering for '{name}'")
        results[name] = hits

    return results

# ─── Formatting ─────────────────────────────────────────────────────────
def format_text(results: dict) -> str:
    logger.debug("Formatting text output")
    lines = []
    for name, articles in results.items():
        lines.append(f"=== {name} ===")
        for art in articles:
            title = art.get("title", "No title")
            url = art.get("url", "")
            snippet = art.get("snippet", art.get("description", ""))
            text = art.get("full_text", "")
            lines.append(f"- {title}\n  {url}\n  {snippet or text}\n")
    return "\n".join(lines)


def generate_html(results: dict) -> str:
    logger.debug("Formatting HTML output")
    html = ["<html><head><meta charset='utf-8'><title>News</title></head><body>"]
    for name, articles in results.items():
        html.append(f"<h2>{name}</h2><ul>")
        for art in articles:
            title = art.get("title", "No title")
            url = art.get("url", "#")
            snippet = art.get("snippet", art.get("description", ""))
            text = art.get("full_text", "")
            content = snippet or text
            html.append(f"<li><a href='{url}'}>{title}</a><p>{content}</p></li>")
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
    logger.info("Starting news filter")
    cfg = load_config(config_path)
    results = fetch_and_filter(cfg)

    content = generate_html(results) if fmt == "html" else format_text(results)

    if output.lower() == "console":
        print(content)
        logger.info("Printed to console")
    else:
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved results to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch & filter news by female leadership")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
    parser.add_argument("-o", "--output", default="news.txt", help="Console or output filepath")
    parser.add_argument("-f", "--format", choices=["text", "html"], default="text", help="Output format")
    parser.add_argument("-l", "--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()
    main(config_path=args.config, output=args.output, fmt=args.format, log_level=args.log)
