import logging
import requests
from newspaper import Article
from readability import Document
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def fetch_with_newspaper(url: str) -> tuple[str, str]:
    """
    Try extracting with newspaper3k, which also provides top_image.
    Returns (text, image_url) or ("","") on failure.
    """
    try:
        art = Article(url)
        art.download()
        art.parse()
        text = art.text.strip()
        image = art.top_image or ""
        if text:
            logger.debug(f"newspaper3k succeeded for {url}")
            return text, image
    except Exception as e:
        logger.debug(f"newspaper3k failed for {url}: {e}")
    return "", ""

def fetch_with_readability(url: str) -> tuple[str, str]:
    """
    Try extracting using readability-lxml on the raw HTML.
    Returns (text, image_url) or ("","") on failure.
    """
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        doc = Document(resp.text)
        summary_html = doc.summary()   # HTML snippet of main content
        soup = BeautifulSoup(summary_html, "lxml")
        text = soup.get_text(separator="\n").strip()

        # Attempt to get an image from the summary HTML
        image = ""
        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            image = og["content"]

        if text:
            logger.debug(f"readability-lxml succeeded for {url}")
            return text, image
    except Exception as e:
        logger.debug(f"readability-lxml failed for {url}: {e}")
    return "", ""

def fetch_with_bs4(url: str) -> tuple[str, str]:
    """
    Fallback: scrape all <p> tags via BeautifulSoup.
    Returns (text, image_url) or ("","") on failure.
    """
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Concatenate all <p> tag text
        paras = [p.get_text().strip() for p in soup.find_all("p")]
        text = "\n\n".join([p for p in paras if p])

        # Attempt to get an image from the page head
        image = ""
        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            image = og["content"]

        if text:
            logger.debug(f"BeautifulSoup fallback succeeded for {url}")
            return text, image
    except Exception as e:
        logger.debug(f"BeautifulSoup fallback failed for {url}: {e}")
    return "", ""

def fetch_full_text_and_image(url: str) -> tuple[str, str]:
    """
    Attempt multiple strategies in order:
      1. newspaper3k
      2. readability-lxml
      3. BeautifulSoup <p> tags
    Returns the first successful (text, image_url) pair, or ("","") if all fail.
    """
    logger.info(f"Fetching full text and image for URL: {url}")

    for method in (fetch_with_newspaper, fetch_with_readability, fetch_with_bs4):
        text, image = method(url)
        if text:
            logger.info(f"Extraction succeeded with {method.__name__} for {url}")
            return text, image

    logger.error(f"All extraction methods failed for {url}")
    return "", ""
