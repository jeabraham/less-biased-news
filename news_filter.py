import yaml
import spacy
from newsapi import NewsApiClient
from genderize import Genderize
import openai
import argparse
import os


def load_config(path="config.yaml"):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def classify_leadership(text: str, prompts: dict, model: str, temperature: float) -> bool:
    """Zero-shot classify whether text describes a woman in leadership."""
    prompt = prompts["classification"] + "\n\n" + text
    resp = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=1,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip().lower() == "yes"


def summarize(text: str, prompts: dict, model: str, temperature: float) -> str:
    """Summarize text via OpenAI."""
    p = prompts["summarize"].format(text=text)
    resp = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=60,
        messages=[{"role": "user", "content": p}]
    )
    return resp.choices[0].message.content.strip()


def spin_genders(text: str, prompts: dict, model: str, temperature: float) -> str:
    """Rewrite text with female-centric focus via OpenAI."""
    p = prompts["spin_genders"].format(text=text)
    resp = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=200,
        messages=[{"role": "user", "content": p}]
    )
    return resp.choices[0].message.content.strip()


def fetch_and_filter(config: dict) -> dict:
    """
    Fetch news per each query in config and apply classification/fallback rules.
    Returns a dict mapping query names to lists of article dicts.
    """
    # Initialize clients
    newsapi = NewsApiClient(api_key=config["newsapi"]["api_key"])
    nlp = spacy.load("en_core_web_sm")
    gndr = Genderize()
    openai.api_key = config["openai"]["api_key"]

    results = {}
    for qcfg in config.get("queries", []):
        name = qcfg.get("name")
        resp = newsapi.get_everything(
            q=qcfg.get("q"),
            language=config["newsapi"].get("language", "en"),
            page_size=qcfg.get("page_size", 100)
        )

        hits = []
        for art in resp.get("articles", []):
            text = " ".join(filter(None, [art.get("title"), art.get("description")]))

            if not qcfg.get("classification", False):
                hits.append(art)
                continue

            # Pre-filter with NER + Genderize
            doc = nlp(text)
            people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            genders = gndr.get(people)
            female_names = {g["name"] for g in genders if g.get("gender") == "female"}
            has_keyword = any(
                kw.lower() in text.lower()
                for kw in config.get("leadership_keywords", [])
            )

            is_female_leader = False
            if female_names and has_keyword:
                is_female_leader = classify_leadership(
                    text,
                    config.get("prompts", {}),
                    config["openai"]["model"],
                    config["openai"].get("temperature", 0)
                )

            if is_female_leader:
                hits.append(art)
            else:
                action = qcfg.get("fallback", "exclude")
                if action == "show-full":
                    hits.append(art)
                elif action == "summarize":
                    art["snippet"] = summarize(
                        text,
                        config.get("prompts", {}),
                        config["openai"]["model"],
                        config["openai"].get("temperature", 0)
                    )
                    hits.append(art)
                elif action == "spin-genders":
                    art["snippet"] = spin_genders(
                        text,
                        config.get("prompts", {}),
                        config["openai"]["model"],
                        config["openai"].get("temperature", 0)
                    )
                    hits.append(art)
                # exclude: do nothing

        results[name] = hits
    return results


def generate_html(news_dict: dict) -> str:
    """Generate an HTML string from news dictionary."""
    html = ["<html>", "<body>"]
    for section, articles in news_dict.items():
        html.append(f"<h2>{section.replace('_', ' ').title()}</h2>")
        html.append("<ul>")
        for art in articles:
            title = art.get("title") or "No Title"
            url = art.get("url") or "#"
            snippet = art.get("snippet") or art.get("description") or ""
            html.append(
                f"<li><a href=\"{url}\" target=\"_blank\">{title}</a>"
                f"<p>{snippet}</p></li>"
            )
        html.append("</ul>")
    html.append("</body>")
    html.append("</html>")
    return "\n".join(html)


def format_text(news_dict: dict) -> str:
    """Format news dictionary as plain text."""
    lines = []
    for section, articles in news_dict.items():
        lines.append(f"== {section.replace('_', ' ').title()} ==")
        for art in articles:
            title = art.get("title") or "No Title"
            url = art.get("url") or ""
            snippet = art.get("snippet") or art.get("description") or ""
            lines.append(f"- {title}\n  {url}\n  {snippet}\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Filter and format news by female leadership.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--format", choices=["text", "html"], default="text",
        help="Output format: plain text or HTML."
    )
    parser.add_argument(
        "--print", action="store_true",
        help="Print output to console instead of saving to file."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file path (defaults to news.txt or news.html)."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    news = fetch_and_filter(config)

    if args.format == "html":
        content = generate_html(news)
        default_file = "news.html"
    else:
        content = format_text(news)
        default_file = "news.txt"

    out_path = args.output or default_file
    if args.print:
        print(content)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved news to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
