import yaml
import spacy
import requests
from newsapi import NewsApiClient
from genderize import Genderize
import openai

# 1) Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# 2) Init clients
newsapi = NewsApiClient(api_key=cfg["newsapi"]["api_key"])
nlp = spacy.load("en_core_web_sm")
gndr = Genderize()
openai.api_key = cfg["openai"]["api_key"]


def classify_leadership(text: str) -> bool:
    """Zero-shot: YES if female leadership, else NO."""
    prompt = cfg["prompts"]["classification"] + "\n\n" + text
    resp = openai.ChatCompletion.create(
        model=cfg["openai"]["model"],
        temperature=cfg["openai"]["temperature"],
        max_tokens=1,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip().lower() == "yes"


def summarize(text: str) -> str:
    p = cfg["prompts"]["summarize"].format(text=text)
    resp = openai.ChatCompletion.create(
        model=cfg["openai"]["model"],
        temperature=cfg["openai"]["temperature"],
        max_tokens=60,
        messages=[{"role": "user", "content": p}]
    )
    return resp.choices[0].message.content.strip()


def spin_genders(text: str) -> str:
    p = cfg["prompts"]["spin_genders"].format(text=text)
    resp = openai.ChatCompletion.create(
        model=cfg["openai"]["model"],
        temperature=cfg["openai"]["temperature"],
        max_tokens=200,
        messages=[{"role": "user", "content": p}]
    )
    return resp.choices[0].message.content.strip()


results = {}

for qcfg in cfg["queries"]:
    name = qcfg["name"]
    print(f"\n▶ Running query '{name}' …")
    resp = newsapi.get_everything(
        q=qcfg["q"],
        language=cfg["newsapi"]["language"],
        page_size=qcfg["page_size"]
    )

    hits = []
    for art in resp.get("articles", []):
        text = " ".join(filter(None, [art.get("title"), art.get("description")]))

        # If no classification needed, just include
        if not qcfg["classification"]:
            hits.append(art)
            continue

        # 1) Quick NER + gender pre-filter
        doc = nlp(text)
        people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        genders = gndr.get(people)
        female_names = {g["name"] for g in genders if g.get("gender") == "female"}
        has_keyword = any(
            kw.lower() in text.lower()
            for kw in cfg["leadership_keywords"]
        )

        # 2) Zero-shot classification
        is_female_leader = False
        if female_names and has_keyword:
            is_female_leader = classify_leadership(text)

        if is_female_leader:
            hits.append(art)
        else:
            action = qcfg["fallback"]
            if action == "show-full":
                hits.append(art)
            elif action == "summarize":
                art["snippet"] = summarize(text)
                hits.append(art)
            elif action == "spin-genders":
                art["snippet"] = spin_genders(text)
                hits.append(art)
            # if action == "exclude", do nothing

    # Store or print results
    results[name] = hits
    print(f" → {len(hits)} results for '{name}'")
    for a in hits:
        print(f"   • {a['title']} — {a['url']}")
