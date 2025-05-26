import os
import json
import argparse
from datetime import datetime


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
    Locate cache files based on queries. If queries aren't specified, find all `_current.json` files.

    Parameters:
    ----------
    queries : list of str, optional
        List of queries provided by the user. If None, searches for all available `_current.json` files.

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
            current_file = os.path.join(cache_dir, f"{query}_current.json")
            yesterday_file = os.path.join(cache_dir, f"{query}_yesterday.json")

            if os.path.exists(current_file):
                cache_files[query] = current_file
            elif os.path.exists(yesterday_file):
                cache_files[query] = yesterday_file
            else:
                logger.warning(f"No cache file found for query '{query}'")
    else:
        # Find all `_current.json` files in the current directory
        for file_name in os.listdir(cache_dir):
            if file_name.endswith("_current.json"):
                query_name = file_name.replace("_current.json", "")
                cache_files[query_name] = os.path.join(cache_dir, file_name)

    return cache_files


def generate_email(results: dict) -> str:
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

    new_today_exists = False

    for articles in results.values():
        for article in articles:
            if article.get("status") != "exclude" and article.get("new_today", True):
                new_today_exists = True
                break
        if new_today_exists:
            break

    # Build the "Table of Contents"
    email_html.append("<h2>Table of Contents</h2>")
    email_html.append("<ul>")
    if new_today_exists:
        email_html.append("<li><a href='#new_today'>New Today</a></li>")
    for query_name in results.keys():
        email_html.append(f"<li><a href='#{query_name}'>{query_name}</a></li>")
    email_html.append("</ul>")


    # Add the "New Today" section
    new_today_articles = [
        article
        for articles in results.values()
        for article in articles
        if article.get("status") != "exclude" and article.get("new_today", True)
    ]
    if  new_today_articles:
        email_html.append("<hr>")  # Horizontal rule for better separation
        email_html.append("<h2 id='new_today' style='color: #cc0000;'>New Today</h2>")
        email_html.append("<ul>")
        for article in new_today_articles:
            email_html.append(render_article_to_email(article))
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
            email_html.append(render_article_to_email(article, query_name=query_name))
        email_html.append("</ul>")
        email_html.append("<div class='back-to-toc'><a href='#'>Back to Table of Contents</a></div>")

    # Add footer
    email_html.append("<hr>")  # Footer separator
    email_html.append("<p style='font-size: 12px; color: #999;'>Generated by news_formatter.py</p>")
    email_html.append("  </body>")
    email_html.append("</html>")

    return "\n".join(email_html)




def format_text(results: dict, metadata: dict = None) -> str:
    """
    Generate a text summary from the filtered news results.

    Enhancements:
    - Formats section headers more clearly and consistently.
    - Adds query strings (metadata) below each section header, if available.
    - Improves layout and readability with consistent indentation and markers.
    - Avoids printing empty or useless information.

    Parameters:
    ----------
    results : dict
        Dictionary of results, where keys are query names and values are lists of articles.
    metadata : dict, optional
        A dictionary containing query-specific metadata such as query strings.

    Returns:
    --------
    str
        A structured and readable text output.
    """
    # Initialize output lines
    lines = []
    metadata = metadata or {}

    # Iterate through the results grouped by query name
    for name, articles in results.items():
        # Header for section (query name)
        lines.append("=" * 60)  # Divider line
        lines.append(f"=== {name} ===")  # Section header

        # Include query string metadata if available
        query_string = metadata.get(name, "Unknown query")
        lines.append(f"Query: {query_string}")
        lines.append("-" * 60)  # Sub-divider for better readability

        # Iterate through the articles in each section
        for art in articles:
            # Skip excluded articles
            if art.get("status") == "exclude":
                continue

            # Extract article details
            title = art.get("title", "No title")
            url = art.get("url", "#")
            status = art.get("status", "")
            leader_name = art.get("leader_name", "")
            if leader_name:
                status += f" ({leader_name})"  # Append leader name to status if available
            content = art.get("content") or art.get("description") or "[No content available]"

            # Print article info
            lines.append(f"- {title}  [{status}]")
            lines.append(f"  Link: {url}")

            # Add content in bullet-pointed paragraphs
            for paragraph in content.split("\n\n"):
                paragraph = paragraph.strip()
                if paragraph:  # Avoid blank paragraphs
                    lines.append(f"    {paragraph}")

            # Add a blank line after each article for separation
            lines.append("")

        # Add a space between sections for readability
        lines.append("")

    # Join the lines into a single text output
    return "\n".join(lines)

def render_article_to_html(article: dict, query_name: str = None) -> str:
    """
    Render an individual article into HTML format.

    Parameters:
    ----------
    article : dict
        The article object containing details such as title, URL, content, images, etc.
    query_name : str, optional
        If provided, appends the query name to the article title for context.

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

    # Add leader name to the status if applicable
    if article.get("leader_name"):
        status += f" ({article['leader_name']})"

    # Start HTML for the article
    html.append("<li>")
    query_display = f" <em>({query_name})</em>" if query_name else ""
    html.append(f"  [{status}] <a href='{url}' target='_blank'>{title}</a>{query_display}")

    # Include the most relevant image if available
    if article.get("most_relevant_image"):
        size, image_url = get_image_size_and_url(article)


        size_style = f"max-width:{size}px;"
        html.append(f"  <img src='{image_url}' alt='' style='{size_style}'>")

    # Process and format content into paragraphs
    paragraphs = [
        para.strip()
        for block in content.split("\n\n") for para in block.split("\n") if para.strip()
    ]
    for para in paragraphs:
        html.append(f"  <p>{para}</p>")

    html.append("</li>")
    return "\n".join(html)


def get_image_size_and_url(article):
    image_url = article["most_relevant_image"]
    image_status = article.get("most_relevant_status", "")
    # Determine image size based on its status
    if image_status in ("female", "female_majority", "female_prominent"):
        base = 400
    elif image_status == "no_face":
        base = 200
    else:  # male or less relevant images
        base = 150

    # Double size if the article is about a female leader
    if article.get("status") == "female_leader":
        final = base * 2
    else:
        final = base

    return final, image_url


def render_article_to_email(article: dict, query_name: str = None) -> str:
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

    if FIRST_ARTICLE:
        logger.info(f"Article title: {title}")
        logger.info(f"Article Status: {status}")
        logger.info(f"Query Name: {query_name}")

    # Start rendering the article with updated styles
    html.append("<tr>")
    html.append(f"  <td>")
    html.append(f"    <strong style='font-size: 18px; color: #333;'>[{status}]</strong><br>")
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

def generate_html(results: dict, metadata: dict = None) -> str:
    """
    Generate an enhanced HTML page from the filtered news results.

    Adds a "New Today" section if applicable and organizes articles by queries.

    Parameters:
    ----------
    results : dict
        Dictionary of results, where keys are query names and values are lists of articles.
    metadata : dict, optional
        A dictionary containing additional metadata such as `query_strings`.

    Returns:
    --------
    str
        A complete HTML string.
    """
    logger.debug("Generating HTML output")

    # Adding optional metadata for query strings
    metadata = metadata or {}

    # Get the current date and time
    from datetime import datetime
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
        "      img { float: right; margin: 0 0 1em 1em; max-width: 300px; }",
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
    new_today_articles = []
    for query_name, articles in results.items():
        for art in articles:
            if art.get("new_today", False):
                new_today_articles.append((query_name, art))  # Include the query name with each article

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
            if art.get("status") == "exclude":
                continue  # Skip excluded articles
            html.append(render_article_to_html(art, query_name=query_name))
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

        for art in articles:
            if art.get("status") == "exclude":
                continue  # Skip excluded articles
            html.append(render_article_to_html(art))  # Use the helper function here

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


def main():
    parser = argparse.ArgumentParser(description="News Formatter")
    parser.add_argument('--format', choices=['email', 'html'], default='email',
                        help="Output format: 'email' (default) or 'html'")
    parser.add_argument('--queries', nargs='+', help="List of query names to include, e.g., 'Calgary Canada'")
    args = parser.parse_args()

    query_files = find_query_files(args.queries)

    if not query_files:
        logger.error("No cache files found. Exiting.")
        return

    results = {}
    for query_name, file_path in query_files.items():
        data = load_json(file_path)
        if data:
            results[query_name] = data.get('articles', [])
        else:
            logger.warning(f"Failed to load data from {file_path}")

    output = generate_email(results) if args.format == 'email' else generate_html(results)
    print(output)


if __name__ == "__main__":
    main()
