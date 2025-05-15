import os
import yaml
import logging
import spacy
from newsapi import NewsApiClient
from genderize import Genderize, GenderizeException
from openai import OpenAI
import openai as openai_pkg
import argparse
import requests
from article_fetcher import fetch_full_text

# ─── Logging Setup ────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

try:
    from gender_guesser.detector import Detector as LocalGenderDetector
    local_detector = LocalGenderDetector()
    LOCAL_GENDER_AVAILABLE = True
except ImportError:
    LOCAL_GENDER_AVAILABLE = False

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

# ─── OpenAI & Mediastack Helpers ───────────────────────────────────────

def classify_leadership(text: str, cfg: dict, client: OpenAI) -> bool:
    prompt = cfg["prompts"]["classification"] + "\n\n" + text
    logger.debug("Running zero-shot classification for leadership")
    try:
        resp = client.chat.completions.create(
            model=cfg["openai"]["model"],
            messages=[{"role":"user","content":prompt}],
            temperature=cfg["openai"].get("temperature",0),
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
            messages=[{"role":"user","content":prompt}],
            temperature=cfg["openai"].get("temperature",0),
            max_tokens=200
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
            messages=[{"role":"user","content":prompt}],
            temperature=cfg["openai"].get("temperature",0),
            max_tokens=4000
        )
        rewrite = resp.choices[0].message.content.strip()
        logger.info("Received spin-genders rewrite")
        return rewrite
    except openai_pkg.error.OpenAIError as e:
        logger.warning(f"OpenAI spin-genders error: {e}")
        return text

from datetime import date, timedelta

def fetch_mediastack(cfg: dict, qcfg: dict) -> list:
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
            logger.info(f"Fetched {len(articles)} articles from Mediastack")
        return articles
    except Exception as e:
        logger.error(f"Mediastack fetch failed: {e}")
        return []

# ─── Fetch & Filter ────────────────────────────────────────────────────

def fetch_and_filter(cfg: dict) -> dict:
    logger.info("Initializing NewsAPI client")
    newsapi = NewsApiClient(api_key=cfg["newsapi"]["api_key"])
    logger.info("Loading spaCy model")
    nlp = spacy.load("en_core_web_sm")
    gndr = Genderize()
    openai_client = OpenAI(api_key=cfg["openai"]["api_key"])

    results = {}
    for qcfg in cfg["queries"]:
        name = qcfg["name"]
        logger.info(f"Processing query '{name}'")
        # ─── fetch raw articles via the right source ────────────────
        provider = qcfg.get("provider", "newsapi").lower()
        if provider == "mediastack":
          # returns list of { title, url, description, etc. }
          raw_articles = fetch_mediastack(cfg, qcfg)
          logger.info(f"Fetched {len(raw_articles)} articles from Mediastack")
        else:
          resp = newsapi.get_everything(
              q = qcfg["q"],
              language = cfg["newsapi"]["language"],
              page_size = qcfg["page_size"]
              )
          raw_articles = resp.get("articles", [])
          logger.info(f"Fetched {len(raw_articles)} articles from NewsAPI")

        # 1) Extract all person names across all articles
        all_persons = set()
        bodies = []
        get_bodies_and_person_names(all_persons, bodies, nlp, raw_articles)

        logger.debug(f"Batch Genderize request names ({len(all_persons)}): {all_persons}")

        # 2) One shot gender detection
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
                    gender_map[name] = "female" if g in ("female","mostly_female") else None
                logger.debug(f"Local gender-map: {gender_map}")
            else:
                logger.error(
                    "Local gender-guesser not installed; all names will be treated as unknown"
                )
                gender_map = {}

        hits = []
        stats = {"total": len(raw_articles), "persons": 0, "female_names": 0,
                 "keyword_hits": 0, "classified_ok": 0}

        # 3) Process each article
        for art, body in zip(raw_articles, bodies):
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
                    is_female_leader = classify_leadership(body, cfg, openai_client)
                except Exception as e:
                    logger.warning(f"OpenAI classification error on '{art.get('title')}': {e}")

            if is_female_leader:
                status = "female_leader"
                stats["classified_ok"] += 1
            else:
                status = qcfg["fallback"]
                # existing fallback logic (show-full, summarize, spin-genders, exclude)
                # and you can log rejection reasons here

            art["status"] = status
            if status in ("female_leader", "show-full"):
                  # keep the full fetched body
                  art["content"] = body
            elif status == "summarize":
                  art["content"] = summarize(body, cfg, openai_client)
            elif status == "spin-genders":
                  art["content"] = spin_genders(body, cfg, openai_client)
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


def get_bodies_and_person_names(all_persons, bodies, nlp, raw_articles):
    for art in raw_articles:
        body = fetch_full_text(art["url"]) or " ".join(filter(None, [art.get("title"), art.get("content")]))
        bodies.append(body)
        doc = nlp(body)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                all_persons.add(ent.text)


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
            title = art.get("title", "No title")
            url = art.get("url", "#")
            status = art.get("status", "")
            content = art.get("content", art.get("description", ""))

            html.append("      <li>")
            html.append(f"        [{status}] <a href='{url}' target='_blank'>{title}</a>")

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

def main(config_path:str="config.yaml",output:str=None,fmt:str="text",log_level:str="INFO"):
    setup_logging(log_level)
    cfg=load_config(config_path)
    res=fetch_and_filter(cfg)
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
    a=p.parse_args(); main(a.config,a.output,a.format,a.log)
