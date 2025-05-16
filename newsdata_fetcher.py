import os
import json
import logging
import requests
from datetime import date, timedelta

logger = logging.getLogger(__name__)

def fetch_newsdata(qcfg: dict, cfg: dict, use_cache: bool = False, cache_dir: str = "newsdata_cache") -> list:
    """
    Fetch articles from NewsData.io, with optional caching.

    Parameters:
    - qcfg: query configuration dict from config.yaml
    - cfg: global config dict containing api_key, base_url, default_history_days
    - use_cache: if True, load from cache if available
    - cache_dir: directory to store cache files

    Returns:
    - List of article dicts (raw JSON from NewsData.io)
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    # Safe filename based on query name
    safe_name = "\".join(c if c.isalnum() or c in (\" \", \"_\", \"-\") else \"_\" for c in qcfg[\"name\"])
    cache_file = os.path.join(cache_dir, f\"{safe_name}.json\")

    # 1) Load from cache if requested
    if use_cache and os.path.exists(cache_file):
        logger.info(f\"Loading NewsData.io results for '{qcfg['name']}' from cache\")
        with open(cache_file, \"r\", encoding=\"utf-8\") as f:
            return json.load(f)

    # 2) Build API request parameters
    api_key = cfg.get(\"newsdata\", {}).get(\"api_key\")
    base_url = cfg.get(\"newsdata\", {}).get(\"base_url\", \"https://newsdata.io/api/1/news\")
    params = {\"apikey\": api_key}

    # Query terms: list or string
    raw_q = qcfg.get(\"q\")
    if isinstance(raw_q, list):
        params[\"q\"] = \" OR \".join(raw_q)
    else:
        params[\"q\"] = raw_q

    # Optional history_days → date range
    history = qcfg.get(\"history_days\", cfg.get(\"newsdata\", {}).get(\"default_history_days\"))
    if history:
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=history)).isoformat()
        params[\"from_date\"] = start
        params[\"to_date\"] = end

    # Optional filters
    if qcfg.get(\"countries\"):
        params[\"country\"] = qcfg.get(\"countries\")
    if qcfg.get(\"language\"):
        params[\"language\"] = qcfg.get(\"language\")

    # Page size (NewsData free tier default is 10, paid up to 50)
    params[\"page\"] = qcfg.get(\"page\", 1)
    params[\"page_size\"] = qcfg.get(\"page_size\", cfg.get(\"newsdata\", {}).get(\"page_size\", 50))

    logger.debug(f\"NewsData.io request params: {params}\")

    # 3) Perform the request
    resp = requests.get(base_url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json().get(\"results\", [])

    # 4) Cache the live result
    try:
        with open(cache_file, \"w\", encoding=\"utf-8\") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f\"Cached {len(data)} items to {cache_file}\")
    except Exception as e:
        logger.warning(f\"Failed to write NewsData.io cache file {cache_file}: {e}\")

    return data
