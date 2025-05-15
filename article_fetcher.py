import logging
import requests
from newspaper import Article as NewspaperArticle
from readability import Document
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" 
                  "AppleWebKit/537.36 (KHTML, like Gecko)" 
                  "Chrome/90.0.4430.93 Safari/537.36"
}


def fetch_with_newspaper(url: str) -> str:
    """Try extracting article text with newspaper3k."""
    try:
        article = NewspaperArticle(url)
        article.download()
        article.parse()
        text = article.text
        if text:
            logger.debug(f"Newspaper3k extracted {len(text)} characters")
            return text
    except Exception as e:
        logger.warning(f"Newspaper3k failed for {url}: {e}")
    return ""


def fetch_with_readability(url: str) -> str:
    """Use readability-lxml to extract main content."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        doc = Document(resp.text)
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "html.parser")
        text = soup.get_text(separator="\n").strip()
        if text:
            logger.debug(f"Readability extracted {len(text)} characters")
            return text
    except Exception as e:
        logger.warning(f"Readability failed for {url}: {e}")
    return ""


def fetch_with_bs4(url: str) -> str:
    """Fallback: simple boilerplate removal via BeautifulSoup."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # remove script and style
        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.decompose()
        texts = soup.stripped_strings
        # heuristic: keep longest continuous block
        blocks = []
        current = []
        for t in texts:
            if len(t) > 80:
                current.append(t)
            else:
                if current:
                    blocks.append(" ".join(current))
                    current = []
        if current:
            blocks.append(" ".join(current))
        if blocks:
            longest = max(blocks, key=len)
            logger.debug(f"BS4 fallback extracted {len(longest)} characters")
            return longest
    except Exception as e:
        logger.warning(f"BS4 fallback failed for {url}: {e}")
    return ""


def fetch_full_text(url: str) -> str:
    """
    Attempt multiple strategies to fetch full article text:
    1. newspaper3k
    2. readability-lxml
    3. BeautifulSoup boilerplate removal
    Returns first successful extraction, or empty string.
    """
    logger.info(f"Fetching full text for URL: {url}")
    for method in (fetch_with_newspaper, fetch_with_readability, fetch_with_bs4):
        text = method(url)
        if text:
            return text
    logger.error(f"All extraction methods failed for {url}")
    return ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch full text of a news article URL.")
    parser.add_argument("url", help="News article URL to fetch")
    parser.add_argument("--print", action="store_true", help="Print text to console")
    parser.add_argument("--output", help="Write text to file")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO
    )

    content = fetch_full_text(args.url)
    if args.print or not args.output:
        print(content)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved content to {args.output}")
