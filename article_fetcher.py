import logging
import requests
from newspaper import Article
from readability import Document
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def fetch_with_newspaper(url: str, max_images: int = 3, diagnostics=None) -> tuple[str, list[str]]:
    """
    Try extracting with newspaper3k, which also provides top_image.
    Ensures the top_image is always the first in the returned list of images.

    Args:
        url (str): The URL of the article to scrape.
        max_images (int): Maximum number of images to return.

    Returns:
        tuple[str, list[str]]:
            - The extracted text of the article.
            - A list of image URLs, with the top_image as the first element (if available).
            Returns ("", []) on failure.
    """
    try:
        art = Article(url)
        art.download()
        art.parse()

        # Extract the main text
        text = art.text.strip()

        # Extract potential images
        images = list(art.images)  # Original image list
        top_image = art.top_image  # Newspaper3k's designated top_image

        # Ensure `top_image` is at the start of the images list
        if top_image and top_image in images:
            images.remove(top_image)  # Avoid duplication
        if top_image:
            images.insert(0, top_image)  # Add to the top
        if diagnostics is not None:
            diagnostics["candidate_images"] = images.copy()
            diagnostics["omitted_by_max_images"] = images[max_images:]

        # Limit the number of images
        images = images[:max_images]

        if text:
            logger.debug(f"newspaper3k succeeded for {url}")
            return text, images
    except Exception as e:
        logger.debug(f"newspaper3k failed for {url}: {e}")
        if diagnostics is not None:
            diagnostics["error"] = str(e)

    return "", []

def fetch_with_readability(url: str,  max_images: int = 3, diagnostics=None) -> tuple[str, list[str]]:
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
        img_tags = soup.find_all("img")
        candidates = [img.get("src") for img in img_tags if img.get("src")]
        images = [img.get("src") for img in img_tags[:max_images] if img.get("src")]

        # Optionally include Open Graph (og:image) if available
        og_image = soup.find("meta", property="og:image")

        if og_image and og_image.get("content") and og_image["content"] not in images:
            images = [og_image["content"]] + images  # Prepend og:image to the list
        if og_image and og_image.get("content") and og_image["content"] not in candidates:
            candidates.insert(0, og_image["content"])
        if diagnostics is not None:
            diagnostics["candidate_images"] = candidates
            diagnostics["omitted_by_max_images"] = [url for url in candidates if url not in images[:max_images]]

        # Limit images to `max_images`
        images = images[:max_images]


        if text:
            logger.debug(f"readability-lxml succeeded for {url}. Found {len(images)} image(s).")
            return text, images

    except Exception as e:
        logger.debug(f"readability-lxml failed for {url}: {e}")
        if diagnostics is not None:
            diagnostics["error"] = str(e)
    return "", []

def fetch_with_bs4(url: str, max_images: int = 3, diagnostics=None) -> tuple[str, list[str]]:
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
        img_tags = soup.find_all("img")
        candidates = [img.get("src") for img in img_tags if img.get("src")]
        images = [img.get("src") for img in img_tags[:max_images] if img.get("src")]

        # Check for an Open Graph (og:image) image and optionally include it
        og_image = soup.find("meta", property="og:image")
        if og_image and og_image.get("content") and og_image["content"] not in images:
            images = [og_image["content"]] + images  # Prepend og:image
        if og_image and og_image.get("content") and og_image["content"] not in candidates:
            candidates.insert(0, og_image["content"])
        if diagnostics is not None:
            diagnostics["candidate_images"] = candidates
            diagnostics["omitted_by_max_images"] = [url for url in candidates if url not in images[:max_images]]

        # Limit to max_images
        images = images[:max_images]

        if text:
            logger.debug(f"BeautifulSoup fallback succeeded for {url}. Found {len(images)} image(s).")
            return text, images
    except Exception as e:
        logger.debug(f"BeautifulSoup fallback failed for {url}: {e}")
        if diagnostics is not None:
            diagnostics["error"] = str(e)

    # On failure, return empty text and image list
    return "", []

def fetch_full_text_and_images(url: str, max_images: int = 3, diagnostics=None) -> tuple[str, list[str]]:
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
        attempt = {"method": method.__name__}
        if diagnostics is not None:
            diagnostics.setdefault("attempts", []).append(attempt)
        try:
            kwargs = {"diagnostics": attempt} if diagnostics is not None else {}
            text, images = method(url, max_images=max_images, **kwargs)
            attempt.update(text_characters=len(text), returned_images=images, succeeded=bool(text))
            if text:
                if diagnostics is not None:
                    diagnostics["selected_method"] = method.__name__
                logger.info(f"Extraction succeeded with {method.__name__} for {url}")
                return text, images
        except Exception as e:
            logger.warning(f"Method {method.__name__} failed for {url}: {e}")
            attempt["error"] = str(e)

    logger.error(f"All extraction methods failed for {url}")
    return "", []
