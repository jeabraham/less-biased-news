import os
import json
import argparse
from datetime import datetime
import textwrap
from datetime import datetime
import html
# Add this lightweight decorator to avoid NameError and emit a warning
import warnings


def deprecated(func):
    """
    Simple deprecation decorator that warns on first call.
    """
    warned = {"done": False}

    def wrapper(*args, **kwargs):
        if not warned["done"]:
            warnings.warn(f"{func.__name__} is deprecated and may be removed in a future release.",
                          DeprecationWarning, stacklevel=2)
            warned["done"] = True
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__dict__.update(func.__dict__)
    return wrapper


import logging

# Constants for image status classification
FEMALE_IMAGE_STATUSES = ("female", "female_majority", "female_prominent")


def setup_logging():
    """
    Set up basic logging for the script.
    """
    import logging
    logger = logging.getLogger("news_formatter")
    logger.setLevel(logging.INFO)

    # Console handler with a basic format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger


# Initialize logger for this module
logger = setup_logging()

FIRST_ARTICLE = True


def load_json(file_path: str):
    """
    Load a JSON file and return its content.
    """
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)


def find_query_files(queries=None, cache_dir="cache"):
    """
    Locate cache files based on queries. If queries aren't specified, find all `_cache.json` files.

    Parameters:
    ----------
    queries : list of str, optional
        List of queries provided by the user. If None, searches for all available `_cache.json` files.

    Returns:
    --------
    dict
        Dictionary where keys are query names, and values are paths to their JSON files.
    """
    cache_files = {}

    # Ensure the cache directory exists
    if not os.path.exists(cache_dir) or not os.path.isdir(cache_dir):
        logger.error(f"The cache directory '{cache_dir}' does not exist.")
        return cache_files

    if queries:
        # Look for files based on user-specified queries
        for query in queries:
            cache_file = os.path.join(cache_dir, f"{query}_cache.json")
            if os.path.exists(cache_file):
                cache_files[query] = cache_file
            else:
                logger.warning(f"No cache file found for query '{query}'")
    else:
        # Find all `_cache.json` files in the cache directory
        for file_name in os.listdir(cache_dir):
            if file_name.endswith("_cache.json"):
                query_name = file_name.replace("_cache.json", "")
                cache_files[query_name] = os.path.join(cache_dir, file_name)

    return cache_files


def extract_articles_from_cache(cache_data):
    """
    Extract articles from cache data, handling both old and new cache formats.
    
    The new cache format is:
    {
        "articles": {
            "article_key": {
                "first_seen": "timestamp",
                "article_data": {...}
            }
        }
    }
    
    The old format was:
    {
        "articles": [...]
    }
    
    Parameters:
    ----------
    cache_data : dict
        The loaded cache data
        
    Returns:
    --------
    list
        List of article dictionaries
    """
    articles_data = cache_data.get('articles', [])
    
    # Check if articles is a dict (new format) or list (old format)
    if isinstance(articles_data, dict):
        # New cache format: extract article_data from each entry
        articles = []
        for article_key, entry in articles_data.items():
            if isinstance(entry, dict) and 'article_data' in entry:
                # Validate that article_data is itself a dict
                article_data = entry['article_data']
                if isinstance(article_data, dict):
                    articles.append(article_data)
                else:
                    logger.warning(f"Invalid article_data type for key: {article_key}, skipping")
            else:
                # Skip invalid entries instead of including them
                logger.warning(f"Unexpected cache entry format for key: {article_key}, skipping")
        return articles
    else:
        # Old format: articles is already a list
        return articles_data if isinstance(articles_data, list) else []


@deprecated
def generate_email_friendly_html(results: dict) -> str:
    """
    Generate email-friendly HTML for filtered news results, including a "New Today" section,
    a table of contents, and "Back to Table of Contents" links.

    Parameters:
    ----------
    results : dict
        Dictionary where keys are query names and values are lists of articles.

    Returns:
    --------
    str
        Email-compliant HTML string.
    """
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Initialize email HTML
    email_html = [
        "<!DOCTYPE html>",
        "<html>",
        "  <head>",
        "    <meta charset='utf-8'>",
        "    <title>News Email</title>",
        "    <style>",
        "      body { font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6; }",
        "      h1 { color: #333; }",
        "      h2 { color: #0066cc; }",
        "      ul { list-style-type: none; padding: 0; }",
        "      li { margin-bottom: 20px; }",
        "      img { display: block; margin-top: 10px; margin-bottom: 10px; }",
        "      p { margin: 5px 0; }",
        "      a { color: #0066cc; text-decoration: none; }",
        "      .back-to-toc { margin-top: 20px; font-size: 12px; }",
        "    </style>",
        "  </head>",
        "  <body>",
        f"    <h1 style='color: #333;'>News Summary - {current_date}</h1>"
    ]

    new_today_articles = get_new_today_articles(results)

    # Build the "Table of Contents"
    email_html.append("<h2>Table of Contents</h2>")
    email_html.append("<ul>")
    if new_today_articles:
        email_html.append("<li><a href='#new_today'>New Today</a></li>")
    for query_name in results.keys():
        email_html.append(f"<li><a href='#{query_name}'>{query_name}</a></li>")
    email_html.append("</ul>")


    # Add the "New Today" section
    if  new_today_articles:
        email_html.append("<hr>")  # Horizontal rule for better separation
        email_html.append("<h2 id='new_today' style='color: #cc0000;'>New Today</h2>")
        email_html.append("<ul>")
        for query_name, article in new_today_articles:
            email_html.append(render_article_to_html_email(article, query_name=query_name))
        email_html.append("</ul>")
        email_html.append("<div class='back-to-toc'><a href='#'>Back to Table of Contents</a></div>")

    # Add the query-specific sections
    for query_name, articles in results.items():
        # Exclude unwanted articles or empty sections
        filtered_articles = [article for article in articles if article.get("status") != "excluded"]
        if not filtered_articles:
            continue

        # Add query name as section heading
        email_html.append("<hr>")  # Horizontal rule for better separation
        email_html.append(f"<h2 id='{query_name}' style='color: #0066cc;'>{query_name}</h2>")
        email_html.append("<ul>")
        for article in filtered_articles:
            email_html.append(render_article_to_html_email(article, query_name=query_name))
        email_html.append("</ul>")
        email_html.append("<div class='back-to-toc'><a href='#'>Back to Table of Contents</a></div>")

    # Add footer
    email_html.append("<hr>")  # Footer separator
    email_html.append("<p style='font-size: 12px; color: #999;'>Generated by news_formatter.py</p>")
    email_html.append("  </body>")
    email_html.append("</html>")

    return "\n".join(email_html)



def generate_text(results: dict, metadata: dict = None, for_email: bool = True) -> str:
    """
    Generate a text summary from the filtered news results, optimized for email readability.

    Parameters:
    ----------
    results : dict
        Dictionary of results, where keys are query names and values are lists of articles.
    metadata : dict, optional
        A dictionary containing query-specific metadata such as query strings.
    for_email : bool, optional
        Whether to format specifically for email (more concise, with special formatting).

    Returns:
    --------
    str
        A structured and readable text output optimized for email viewing.
    """
    # Initialize output lines
    lines = []
    metadata = metadata or {}

    # Add email header if for_email is True
    if for_email:
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        lines.append(f"NEWS SUMMARY - {current_date}")
        lines.append("")
        lines.append("This is an automated summary of recent news articles.")
        lines.append("")

    # Track total articles for summary
    total_articles = sum(len([a for a in articles if a.get("status") != "exclude"])
                         for articles in results.values())

    # Add a brief overview if for_email is True
    if for_email and total_articles > 0:
        topic_count = len(results)
        lines.append(f"OVERVIEW: {total_articles} articles across {topic_count} topics")
        lines.append("")

    # Add "New Today" section first if there are any new articles
    new_today_articles = get_new_today_articles(results)
    if new_today_articles:
        if for_email:
            lines.append(f"## NEW TODAY ##")
        else:
            lines.append("=" * 60)  # Divider line
            lines.append(f"=== NEW TODAY ===")  # Section header
            lines.append("-" * 60)  # Sub-divider for better readability
            lines.append("")

        for query_name, art in new_today_articles:
            process_article_in_text(art, for_email, -1, lines, section=query_name)
            lines.append("")
        lines.append("-" * 40)
        lines.append("")

    # Iterate through the results grouped by query name
    for name, articles in results.items():
        # Filter out excluded articles
        valid_articles = [a for a in articles if a.get("status") != "exclude"]

        # Skip empty sections
        if not valid_articles:
            continue

        # Header for section (query name)
        if for_email:
            lines.append(f"## {name.upper()} ({len(valid_articles)} articles) ##")
        else:
            lines.append("=" * 60)  # Divider line
            lines.append(f"=== {name} ===")  # Section header

        # Include query string metadata if available and not in email mode
        if not for_email:
            query_string = metadata.get(name, "")
            if query_string and query_string!= "":
                lines.append(f"Query: {query_string}")
            lines.append("-" * 60)  # Sub-divider for better readability

        # Iterate through the articles in each section
        for i, art in enumerate(valid_articles, 1):
            process_article_in_text(art, for_email, i, lines)
            if not for_email:
                lines.append("-" * 60)  # Sub-divider for better readability

        # Add a separator between sections
        if for_email:
            lines.append("-" * 40)
        lines.append("")

    # Add footer for email
    if for_email and total_articles > 0:
        lines.append("")
        lines.append("Generated by News Summarizer | " + datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Join the lines into a single text output
    return "\n".join(lines)


def process_article_in_text(article, for_email, article_number, lines, section: str = None):
    # Extract article details
    title = article.get("title", "No title")
    url = article.get("url", "#")
    source = article.get("source", "")
    published = article.get("published_at", article.get("publishedAt", ""))
    # Format the date if available
    date_str = ""
    if published:
        try:
            # Try to parse the date and format it nicely
            dt = datetime.fromisoformat(published.replace('Z', '+00:00'))
            date_str = dt.strftime("%b %d")
        except (ValueError, TypeError):
            # If parsing fails, use the raw string
            date_str = published
    # Combine source and date
    source_date = []
    if source:
        source_date.append(str(source.get("name", source) if isinstance(source, dict) else source))
    if date_str:
        source_date.append(str(date_str))
    source_date_str = " | ".join(source_date)
    # Get content
    content = article.get("content") or article.get("description") or "[No content available]"
    # For email, keep summaries brief
    if for_email and len(content) > 500:
        content = content[:497] + "..."
    # Format article for email
    if for_email:
        if article_number > 0:
            lines.append(f"{article_number}. {title}")
        else:
            lines.append(f"{section or ''}: {title}")
        # Add status information
        status = article.get("status", "")
        if article.get("leader_name"):
            status += f" ({article.get('leader_name')})"
            # Add image status if available
            if article.get("most_relevant_image"):
                img_status = article.get("most_relevant_status", "unknown")
                status += f" | {img_status}"
        if status:
            lines.append(f"   Status: {status}")

        if source_date_str:
            lines.append(f"   {source_date_str}")
        lines.append(f"{url}")

        # Add content as indented paragraph
        content_lines = textwrap.wrap(content, width=70)
        for line in content_lines:
            lines.append(f"   {line}")

        # Add space between articles
        lines.append("")
    else:
        # Original format
        status = article.get("status", "")
        leader_name = article.get("leader_name", "")
        if leader_name:
            status += f" ({leader_name})"

        lines.append(f"- {title}  [{status}]")
        if article.get("most_relevant_image"):
            img_status = article.get("most_relevant_status", "unknown")
            lines.append(f"   Image status: {img_status}")
        lines.append(f"  Link: {url}")

        # Add content in bullet-pointed paragraphs
        for paragraph in content.split("\n\n"):
            paragraph = paragraph.strip()
            if paragraph:  # Avoid blank paragraphs
                lines.append(f"    {paragraph}")

        # Add a blank line after each article for separation
        lines.append("")


def render_article_to_html(article: dict, query_name: str = None, cfg: dict = None) -> str:
    """
    Render an individual article into HTML format.

    Parameters:
    ----------
    article : dict
        The article object containing details such as title, URL, content, images, etc.
    query_name : str, optional
        If provided, appends the query name to the article title for context.
    cfg : dict, optional
        Configuration dict for image sizing

    Returns:
    --------
    str
        A formatted HTML string representing the article.
    """
    html = []

    # Get article details
    title = article.get("title", "No title")
    url = article.get("url", "#")
    content = article.get("content") or article.get("description") or ""
    status = article.get("status", "")
    
    # Build status display string with leader name if applicable
    status_display = ""
    if status:
        status_display = status
        if article.get("leader_name"):
            status_display += f" ({article['leader_name']})"
    
    # Get source and date information
    source = article.get("source", "")
    published = article.get("published_at", article.get("publishedAt", ""))
    
    # Format the date if available
    # Note: NewsAPI and MediaStack return ISO 8601 formatted dates (e.g., '2024-01-15T10:30:00Z')
    date_str = ""
    if published:
        try:
            # Handle ISO 8601 date formats
            # Replace 'Z' with '+00:00' for fromisoformat() compatibility (Python 3.7+)
            date_to_parse = published
            if isinstance(published, str) and published.endswith('Z'):
                date_to_parse = published.replace('Z', '+00:00')
            
            dt = datetime.fromisoformat(date_to_parse)
            date_str = dt.strftime("%b %d, %Y")
        except (ValueError, TypeError, AttributeError):
            # Fallback: Try to extract just the date part from ISO 8601 format
            try:
                if isinstance(published, str) and len(published) >= 10:
                    # Assumes ISO 8601 format where first 10 chars are YYYY-MM-DD
                    date_part = published[:10]
                    dt = datetime.strptime(date_part, "%Y-%m-%d")
                    date_str = dt.strftime("%b %d, %Y")
                else:
                    # If date is very short, just use it as-is (truncated)
                    date_str = str(published)[:10] if published else ""
            except (ValueError, TypeError):
                # Last resort: Display truncated raw string for debugging
                date_str = str(published)[:20] if published else ""
    
    # Build metadata string
    metadata_parts = []
    if source:
        source_name = source.get("name", source) if isinstance(source, dict) else source
        metadata_parts.append(str(source_name))
    if date_str:
        metadata_parts.append(str(date_str))
    metadata_str = " | ".join(metadata_parts)

    # Start HTML for the article with spacing between articles (18pt margin)
    html.append("<li style='margin-bottom: 18pt;'>")
    
    # Article title in bold with metadata in smaller font
    query_display = f" <em>({query_name})</em>" if query_name else ""
    html.append(f"  <a href='{url}' target='_blank' style='font-weight: bold;'>{title}</a>{query_display}")
    
    # Add status and metadata in smaller font
    if status_display or metadata_str:
        status_metadata = []
        if status_display:
            status_metadata.append(f"[{status_display}]")
        if metadata_str:
            status_metadata.append(metadata_str)
        html.append(f"  <div style='font-size: 0.85em; color: #666; margin-top: 4pt; margin-bottom: 8pt'>{' | '.join(status_metadata)}</div>")
    
    # Include the most relevant image first (headline image)
    if article.get("most_relevant_image"):
        size, image_url = get_image_size_and_url(article, cfg=cfg)
        size_style = f"max-width:{size}px;"
        html.append(f"  <img src='{image_url}' alt='Article image' style='{size_style}'>")
    
    # Include all other images that contain women
    image_analysis = article.get("image_analysis", {})
    most_relevant_image = article.get("most_relevant_image")
    for img_url, analysis in image_analysis.items():
        # Skip the most relevant image (already displayed)
        if img_url == most_relevant_image:
            continue
        # Only include images with women
        img_status = analysis.get("status", "")
        if img_status in FEMALE_IMAGE_STATUSES:
            size, _ = get_image_size_and_url(article, image_url=img_url, cfg=cfg)
            size_style = f"max-width:{size}px;"
            html.append(f"  <img src='{img_url}' alt='Article image' style='{size_style}'>")

    # Process and format content into paragraphs with 6pt spacing
    paragraphs = [
        para.strip()
        for block in content.split("\n\n") for para in block.split("\n") if para.strip()
    ]
    for para in paragraphs:
        html.append(f"  <p style='margin-bottom: 4pt; margin-top: 4pt;'>{para}</p>")

    html.append("</li>")
    return "\n".join(html)


def get_image_size_and_url(article, image_url=None, cfg=None):
    """
    Calculate appropriate image size based on article status and image classification.
    
    Parameters:
    ----------
    article : dict
        The article containing status and image information
    image_url : str, optional
        Specific image URL to size (if None, uses most_relevant_image)
    cfg : dict, optional
        Configuration dict with image_sizes parameters
    
    Returns:
    --------
    tuple: (size, image_url)
    """
    if image_url is None:
        image_url = article.get("most_relevant_image")
    
    # Get image status - either from specific image or most_relevant
    if image_url and image_url in article.get("image_analysis", {}):
        image_status = article["image_analysis"][image_url].get("status", "")
    else:
        image_status = article.get("most_relevant_status", "")
    
    # Get config defaults or use hardcoded defaults
    if cfg and "image_sizes" in cfg:
        img_cfg = cfg["image_sizes"]
        female_base = img_cfg.get("female_base", 400)
        no_face_base = img_cfg.get("no_face_base", 200)
        other_base = img_cfg.get("other_base", 150)
        leader_multiplier = img_cfg.get("leader_multiplier", 2)
    else:
        female_base = 400
        no_face_base = 200
        other_base = 150
        leader_multiplier = 2
    
    # Determine image size based on its status
    if image_status in FEMALE_IMAGE_STATUSES:
        base = female_base
    elif image_status == "no_face":
        base = no_face_base
    else:  # male or less relevant images
        base = other_base

    # Multiply size if the article is about a female leader
    if article.get("status") == "female_leader":
        final = base * leader_multiplier
    else:
        final = base

    return final, image_url

@deprecated
def render_article_to_html_email(article: dict, query_name: str = None) -> str:
    """
    Render an article in email-compatible HTML with dynamic image sizing.

    Parameters:
    ----------
    article : dict
        The article object containing title, URL, content, images, etc.
    query_name : str, optional
        If provided, includes the query name for context.

    Returns:
    --------
    str
        Email-compatible HTML for the article.
    """
    html = []

    global FIRST_ARTICLE
    if FIRST_ARTICLE:
        logger.info(f"Processing the first article with URL: {article.get('url', 'No URL')}")

    # Extract article details
    title = article.get("title", "No Title")
    url = article.get("url", "#")
    content = article.get("content") or article.get("description", "Content not available.")
    status = article.get("status", "")

    # Add leader name to the status if applicable
    if article.get("leader_name"):
        status += f" ({article['leader_name']})"

    if FIRST_ARTICLE:
        logger.info(f"Article title: {title}")
        logger.info(f"Article Status: {status}")
        logger.info(f"Query Name: {query_name}")

    # Start rendering the article with updated styles
    html.append("<tr>")
    html.append(f"  <td>")
    html.append(f"    <strong style='font-size: 18px; color: #333;'>{status}</strong><br>")
    html.append(
        f"    <a href='{url}' style='font-size: 20px; font-weight: bold; color: #1a0dab; text-decoration: none;'>{title}</a>")
    if query_name:
        html.append(f"    <br><em style='font-size: 16px; color: #666;'>({query_name})</em>")
    html.append("  </td>")
    html.append("</tr>")

    paragraphs = [
        para.strip()
        for block in content.split("\n\n") for para in block.split("\n") if para.strip()
    ]
    #print("Paragraph count ", len(paragraphs))
    # Optional content snippet
    html.append("<tr>")
    for para in paragraphs:
        if FIRST_ARTICLE:
            logger.info(f"Processing paragraph: {para[:50]}...")

        # Add a row for each paragraph with proper spacing
        html.append("<tr>")
        html.append(f"  <td style='padding: 10px 0;'>")
        html.append(f"    <div style='margin-bottom: 15px; line-height: 1.6;'>{para}</div>")
        html.append("  </td>")
        html.append("</tr>")

    # Add a spacer row before the "Read more" link to ensure visual separation
    html.append("<tr>")
    html.append("  <td style='height: 20px;'></td>")
    html.append("</tr>")

    # # Add the "Read more" link
    # html.append("<tr>")
    # html.append(
    #     f"  <td style='padding: 10px 0; text-align: center;'><a href='{url}' style='text-decoration: none; font-weight: bold; color: blue;'>Read more</a></td>")
    # html.append("</tr>")

    # Attach primary image with dynamically calculated size
    if article.get("most_relevant_image"):
        size, image_url = get_image_size_and_url(article)

        # Add image with calculated size
        html.append("<tr>")
        html.append(
            f"  <td><img src='{image_url}' alt='Article Image' width='{size}' style='display: block; margin-top: 10px;'></td>")
        html.append("</tr>")

    # Add horizontal rule to separate articles
    html.append("<tr><td style='padding-top: 10px;'><hr></td></tr>")

    FIRST_ARTICLE = False
    return "\n".join(html)

def generate_html(results: dict, metadata: dict = None, cfg: dict = None) -> str:
    """
    Generate an enhanced HTML page from the filtered news results.

    Adds a "New Today" section if applicable and organizes articles by queries.

    Parameters:
    ----------
    results : dict
        Dictionary of results, where keys are query names and values are lists of articles.
    metadata : dict, optional
        A dictionary containing additional metadata such as `query_strings`.
    cfg : dict, optional
        Configuration dict for image sizing and other settings

    Returns:
    --------
    str
        A complete HTML string.
    """
    logger.debug("Generating HTML output")

    # Adding optional metadata for query strings
    metadata = metadata or {}

    # Get the current date and time
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Initialize the HTML output
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "  <head>",
        "    <meta charset='utf-8'>",
        "    <title>News</title>",
        "    <style>",
        "      body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0 15px; }",
        "      h1, h2 { color: #333; }",
        "      ul { list-style-type: disc; margin-left: 20px; }",
        "      hr { border: 0; height: 1px; background: #ccc; margin: 20px 0; }",
        "      .toc { margin-bottom: 20px; }",
        "      ",
        "      /* Responsive image layout */",
        "      li { overflow: auto; min-width: 0; }",
        "      li img { float: right; margin: 0 0 1em 1em; max-width: 45%; clear: right; }",
        "      li p { min-width: 200px; }",
        "      ",
        "      /* When window is narrow, stack images below text */",
        "      @media (max-width: 768px) {",
        "        li img { float: none; display: block; max-width: 100%; margin: 1em auto; }",
        "        li p { min-width: 0; }",
        "      }",
        "      ",
        "      /* When images would make text too narrow, stack them */",
        "      @media (min-width: 769px) and (max-width: 1024px) {",
        "        li img { max-width: 40%; }",
        "      }",
        "      ",
        "      p, li { overflow: auto; }",
        "      .toc-link { margin-top: 20px; display: block; font-size: 0.9em; }",
        "    </style>",
        "  </head>",
        "  <body>"
    ]

    # Add page header with date and credit
    html.extend([
        "    <h1>News Summary</h1>",
        f"    <p><strong>Date:</strong> {current_date}</p>",
        "    <p><strong>Processed by:</strong> "
        "<a href='https://github.com/jeabraham/less-biased-news' target='_blank'>less_biased_news</a></p>",
        "    <hr>"
    ])

    # Gather articles tagged as "new_today"
    new_today_articles = get_new_today_articles(results)

    # Generate the Table of Contents (TOC) FIRST
    html.append("    <div class='toc'>")
    html.append("      <h2>Table of Contents</h2>")
    html.append("      <ul>")
    if new_today_articles:
        html.append("        <li><a href='#new_today'>New Today</a></li>")  # Add "New Today" to TOC
    for query_name in results:
        section_id = query_name.replace(" ", "_").lower()  # Generate HTML-safe IDs
        html.append(f"        <li><a href='#{section_id}'>{query_name}</a></li>")
    html.append("      </ul>")
    html.append("    </div>")
    html.append("    <hr>")  # Separate TOC from the content

    # Then add the "New Today" section if applicable
    if new_today_articles:
        html.append("    <section id='new_today'>")
        html.append("      <h2>New Today</h2>")
        html.append("      <ul>")
        for query_name, art in new_today_articles:
            if not isinstance(art, dict):
                continue
            if art.get("status") == "exclude":
                continue  # Skip excluded articles
            html.append(render_article_to_html(art, query_name=query_name, cfg=cfg))
        html.append("      </ul>")
        html.append("      <hr>")
        html.append("    </section>")

    # Add each section for query results
    for query_name, articles in results.items():
        section_id = query_name.replace(" ", "_").lower()  # ID for linking
        query_string = metadata.get(query_name, "Not provided")  # Get query string from metadata

        html.append(f"    <section id='{section_id}'>")
        html.append(f"      <h2>{query_name}</h2>")
        html.append(f"      <p><strong>Query:</strong> {query_string}</p>")
        html.append("      <ul>")

        # Process articles (should always be a list now due to extract_articles_from_cache)
        for art in articles:
            if not isinstance(art, dict):
                continue
            if art.get("status") == "exclude":
                continue
            html.append(render_article_to_html(art, cfg=cfg))

        html.append("      </ul>")
        # Add "Table of Contents" link at the end of the section
        html.append(
            "      <a class='toc-link' href='#' onclick='window.scrollTo({top: 0, behavior: \"smooth\"});'>Back to Table of Contents</a>")
        html.append("    </section>")
        html.append("    <hr>")

    # Close the HTML structure
    html.extend([
        "  </body>",
        "</html>"
    ])

    return "\n".join(html)


def get_new_today_articles(results):
    """
    Get all articles marked as new today across all queries.
    
    Parameters:
    ----------
    results : dict
        Dictionary where keys are query names and values are lists of article dicts
        
    Returns:
    --------
    list of tuple
        List of (query_name, article) tuples for articles marked as new_today
    """
    new_today_articles = []
    for query_name, articles in results.items():
        # articles should always be a list due to extract_articles_from_cache
        if not isinstance(articles, list):
            logger.warning(f"Expected list of articles for query '{query_name}', got {type(articles)}")
            continue
            
        for art in articles:
            if not isinstance(art, dict):
                continue
            if art.get("new_today", False):
                if art.get("status") != "exclude":
                    new_today_articles.append((query_name, art))
    return new_today_articles

def main():
    """
    Main function to run the news formatter with command line arguments.
    """
    parser = argparse.ArgumentParser(description="News Formatter - Generates formatted news summaries")

    # Format options
    parser.add_argument('--format', choices=['email', 'html', 'text'], default='email',
                        help="Output format: 'email' (default email-friendly text), 'text' (detailed text), or 'html'")

    # Content selection options
    parser.add_argument('--queries', nargs='+', help="List of query names to include, e.g., 'Calgary Canada'")
    parser.add_argument('--limit', type=int, default=None,
                        help="Maximum number of articles per query (default: all)")

    # Output options
    parser.add_argument('--output', '-o', type=str, help="Output file path (default: stdout)")
    parser.add_argument('--title', type=str, default="News Summary",
                        help="Title for the summary (used in email and HTML formats)")

    # Testing options
    parser.add_argument('--test', action='store_true', help="Use sample data for testing")
    parser.add_argument('--summarize', action='store_true',
                        help="Summarize article content (requires additional dependencies)")

    args = parser.parse_args()

    # Handle test mode with sample data
    if args.test:
        logger.info("Using sample data for testing")
        results = create_test_data()
        metadata = {}  # Test mode doesn't load metadata from files
    else:
        # Find and load real query files
        query_files = find_query_files(args.queries)

        if not query_files:
            logger.error("No cache files found. Exiting.")
            return

        # Load data from files
        results = {}
        metadata = {}
        for query_name, file_path in query_files.items():
            data = load_json(file_path)
            if data:
                # Extract articles using the common helper function
                articles = extract_articles_from_cache(data)

                # Apply article limit if specified
                if args.limit and args.limit > 0:
                    articles = articles[:args.limit]

                results[query_name] = articles

                # Store query metadata
                metadata[query_name] = data.get('query_string', '')
            else:
                logger.warning(f"Failed to load data from {file_path}")

    # Apply content summarization if requested
    if args.summarize:
        try:
            from summarization import summarize_text_using_local_model
            logger.info("Applying text summarization to article content")

            for query_name, articles in results.items():
                for article in articles:
                    if content := article.get('content'):
                        # Estimate input length and adjust max_length accordingly
                        input_length = len(content) // 4  # rough token estimation
                        max_length = min(100, max(30, input_length // 2))

                        article['original_content'] = content
                        article['content'] = summarize_text_using_local_model(
                            content,
                            max_length=max_length,
                            min_length=min(30, max_length - 10)
                        )
        except ImportError:
            logger.warning("Summarization requested but summarization module not available")

    # Generate output based on format
    if args.format == 'email':
        output = generate_text(results, metadata, for_email=True)
    elif args.format == 'text':
        output = generate_text(results, metadata, for_email=False)
    elif args.format == 'html':
        output = generate_html(results, metadata)
    else:
        output = "Invalid format specified"

    # Write output to file or print to stdout
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        logger.info(f"Output written to {args.output}")
    else:
        print(output)


def create_test_data():
    """Generate sample data for testing the formatter"""
    return {
        "Technology News": [
            {
                "title": "New AI Breakthrough in Natural Language Processing",
                "url": "https://example.com/ai-news",
                "source": "Tech Daily",
                "published_at": "2023-05-27T10:30:00Z",
                "content": "Researchers have announced a significant breakthrough in natural language processing. "
                           "The new model achieves state-of-the-art results on multiple benchmarks while using "
                           "30% fewer parameters than previous models. This development could lead to more "
                           "efficient AI systems that require less computational resources to train and run.",
                "status": "include"
            },
            {
                "title": "Quantum Computing Reaches New Milestone",
                "url": "https://example.com/quantum-news",
                "source": "Science Today",
                "published_at": "2023-05-26T14:15:00Z",
                "content": "Scientists have successfully demonstrated quantum entanglement across a record distance "
                           "of 100 kilometers using a new fiber optic technology. This achievement brings us one step "
                           "closer to a functional quantum internet, which would revolutionize secure communications "
                           "and distributed computing capabilities.",
                "status": "include"
            }
        ],
        "Climate Change": [
            {
                "title": "Global CO2 Levels Reach New High Despite Reduction Efforts",
                "url": "https://example.com/climate-news",
                "source": "Environmental Report",
                "published_at": "2023-05-25T09:45:00Z",
                "content": "Atmospheric carbon dioxide levels have reached 420 parts per million for the first time "
                           "in recorded history, according to measurements from the Mauna Loa Observatory. This represents "
                           "a 50% increase since pre-industrial times and continues the upward trend despite international "
                           "efforts to reduce emissions. Scientists warn that more aggressive action is needed to avoid "
                           "the worst impacts of climate change.",
                "status": "include"
            }
        ]
    }


if __name__ == "__main__":
    setup_logging()
    main()
