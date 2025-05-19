import logging
import requests
from newspaper import Article
from readability import Document
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def fetch_with_newspaper(url: str, max_images: int = 3) -> tuple[str, list[str]]:
    """
    Try extracting with newspaper3k, which also provides top_image.
    Returns (text, image_url) or ("","") on failure.
    """
    try:
        art = Article(url)
        art.download()
        art.parse()
        text = art.text.strip()
        # Extract images and limit to `max_images`
        images = list(art.images)[:max_images]
        if text:
            logger.debug(f"newspaper3k succeeded for {url}")
            return text, images
    except Exception as e:
        logger.debug(f"newspaper3k failed for {url}: {e}")
    return "", []

def fetch_with_readability(url: str,  max_images: int = 3) -> tuple[str, list[str]]:
    """
    Try extracting text and multiple images using readability-lxml on the raw HTML.
    Returns (text, image_urls) or ("", []) on failure.

    Args:
        url: The URL of the article to parse.
        max_images: The maximum number of image URLs to retrieve.

    Returns:
        A tuple containing:
          - text: The extracted article text (empty string on failure).
          - images: A list of up to `max_images` image URLs (empty list on failure).
    """
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        doc = Document(resp.text)
        summary_html = doc.summary()   # HTML snippet of main content
        soup = BeautifulSoup(summary_html, "lxml")
        text = soup.get_text(separator="\n").strip()

        # Extract all images from the <img> tags in the main content
        img_tags = soup.find_all("img", limit=max_images)
        images = [img.get("src") for img in img_tags if img.get("src")]

        # Optionally include Open Graph (og:image) if available
        og_image = soup.find("meta", property="og:image")

        if og_image and og_image.get("content") and og_image["content"] not in images:
            images = [og_image["content"]] + images  # Prepend og:image to the list

        # Limit images to `max_images`
        images = images[:max_images]


        if text:
            logger.debug(f"readability-lxml succeeded for {url}. Found {len(images)} image(s).")
            return text, images

    except Exception as e:
        logger.debug(f"readability-lxml failed for {url}: {e}")
    return "", []

def fetch_with_bs4(url: str, max_images: int = 3) -> tuple[str, list[str]]:
    """
    Fallback: scrape all <p> tags via BeautifulSoup and extract multiple images.
    Returns (text, list_of_image_urls) or ("",[]) on failure.

    Args:
        url: The URL of the page to scrape.
        max_images: The maximum number of image URLs to retrieve.

    Returns:
        A tuple containing:
          - text: The concatenated text from all <p> tags of the page.
          - images: A list of up to `max_images` image URLs.
    """
    try:
        # Download the webpage
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Extract article text from <p> tags
        paras = [p.get_text().strip() for p in soup.find_all("p")]
        text = "\n\n".join([p for p in paras if p])  # Join the non-empty paragraphs

        # Extract image URLs from <img> tags
        img_tags = soup.find_all("img", limit=max_images)
        images = [img.get("src") for img in img_tags if img.get("src")]

        # Check for an Open Graph (og:image) image and optionally include it
        og_image = soup.find("meta", property="og:image")
        if og_image and og_image.get("content") and og_image["content"] not in images:
            images = [og_image["content"]] + images  # Prepend og:image

        # Limit to max_images
        images = images[:max_images]

        if text:
            logger.debug(f"BeautifulSoup fallback succeeded for {url}. Found {len(images)} image(s).")
            return text, images
    except Exception as e:
        logger.debug(f"BeautifulSoup fallback failed for {url}: {e}")

    # On failure, return empty text and image list
    return "", []

def fetch_full_text_and_images(url: str, max_images: int = 3) -> tuple[str, list[str]]:
    """
    Attempt multiple strategies in order:
      1. newspaper3k
      2. readability-lxml
      3. BeautifulSoup <p> tags
    Returns the first successful (text, images) pair, or ("", []) if all fail.

    Args:
        url: The URL of the page to extract content from.
        max_images: The maximum number of image URLs to retrieve.

    Returns:
        A tuple containing:
          - text: The article's text if successful, or an empty string if all methods fail.
          - images: A list of image URLs if successful, or an empty list if all methods fail.
    """
    logger.info(f"Fetching full text and up to {max_images} images for URL: {url}")

    # Loop through each extraction method and pass the max_images parameter
    for method in (fetch_with_newspaper, fetch_with_readability, fetch_with_bs4):
        try:
            text, images = method(url, max_images=max_images)  # Pass max_images to the method
            if text:
                logger.info(f"Extraction succeeded with {method.__name__} for {url}")
                return text, images
        except Exception as e:
            logger.warning(f"Method {method.__name__} failed for {url}: {e}")

    logger.error(f"All extraction methods failed for {url}")
    return "", []
