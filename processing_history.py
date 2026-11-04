"""Persistent, JSON-serializable article processing records and HTML reports."""

from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import difflib
import hashlib
import html
import json
from pathlib import Path
from urllib.parse import urlsplit

_active_article = ContextVar("processing_article", default=None)


@contextmanager
def history_scope(article):
    token = _active_article.set(article)
    try:
        yield
    finally:
        _active_article.reset(token)


def record_model_event(name, details):
    article = _active_article.get()
    if article is not None:
        record_step(article, name, details)


def get_article_id(article):
    key = f"{article.get('url', '')}||{article.get('title', '')}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def start_history(article):
    article["article_id"] = get_article_id(article)
    article["processing_history"] = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "retrieved_data": deepcopy({
            key: value for key, value in article.items()
            if key not in ("processing_history", "processing_page")
        }),
        "steps": [],
    }


def record_step(article, name, details=None, before=None, after=None):
    history = article.get("processing_history")
    if history is None:
        return
    step = {"name": name, "at": datetime.now(timezone.utc).isoformat(),
            "details": deepcopy(details or {})}
    if before is not None and after is not None:
        step.update(before=before, after=after)
    history["steps"].append(step)


def _pre(value):
    if not isinstance(value, str):
        value = json.dumps(value, indent=2, ensure_ascii=False)
    return f"<pre>{html.escape(value)}</pre>"


def _url_link(url):
    escaped = html.escape(str(url), quote=True)
    if urlsplit(str(url)).scheme.lower() not in ("http", "https"):
        return escaped
    return f'<a href="{escaped}" rel="noreferrer">{escaped}</a>'


def render_processing_page(article_id, histories):
    parts = [
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
        "<title>Story processing history</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:1em}"
        ".added{color:#165b24;background:#e6ffed}.removed{color:#8b1515;background:#ffeef0}"
        "section{border-top:1px solid #ccc;margin-top:2em}a{overflow-wrap:anywhere}</style>",
        "</head><body><h1>Story processing history</h1>",
        f"<p>Article ID: {html.escape(article_id)}</p>",
    ]
    for history in histories:
        raw = history["retrieved_data"]
        parts.extend([
            f"<section><h2>{html.escape(str(raw.get('title', 'Untitled')))}</h2>",
            f"<p>Source URL: {_url_link(raw.get('url', ''))}</p>",
            f"<p>Processed: {html.escape(history['processed_at'])}</p>",
            "<details><summary>Retrieved provider data</summary>",
            _pre(raw), "</details>",
        ])
        for number, step in enumerate(history["steps"], 1):
            parts.extend([
                f"<h3>{number}. {html.escape(step['name'])}</h3>",
                f"<p>{html.escape(step['at'])}</p>",
                "<details><summary>Step data</summary>", _pre(step["details"]), "</details>",
            ])
            if step["name"] == "Image analysis and selection":
                parts.append("<table><tr><th>Image</th><th>Status</th><th>Faces</th><th>Prominence</th></tr>")
                for url, analysis in step["details"]["images"].items():
                    parts.append(
                        f"<tr><td>{_url_link(url)}</td><td>{html.escape(analysis['status'])}</td>"
                        f"<td>{len(analysis.get('raw_faces') or [])}</td>"
                        f"<td>{html.escape(str(analysis['prominence_score']))}</td></tr>"
                    )
                parts.append("</table>")
            if "before" in step:
                before, after = step["before"], step["after"]
                delta = len(after) - len(before)
                change = "lengthened" if delta > 0 else "shortened" if delta < 0 else "same length"
                parts.append(
                    f"<p>{len(before.split())} to {len(after.split())} words; "
                    f"{len(before)} to {len(after)} characters ({change}, {delta:+d}).</p>"
                )
                diff = list(difflib.unified_diff(
                    before.splitlines(keepends=True), after.splitlines(keepends=True),
                    fromfile="Before", tofile="After", lineterm="",
                ))
                parts.append("<details open><summary>Text diff</summary><pre>")
                for line in diff:
                    css = "added" if line.startswith("+") else "removed" if line.startswith("-") else ""
                    parts.append(f'<span class="{css}">{html.escape(line)}</span>\n')
                if not diff:
                    parts.append("No text changes.")
                parts.extend(["</pre></details>",
                              "<details><summary>Input text</summary>", _pre(before), "</details>",
                              "<details><summary>Output text</summary>", _pre(after), "</details>"])
        parts.append("</section>")
    parts.append("</body></html>")
    return "\n".join(parts)


def write_processing_pages(results, output_dir, preserve_cached=True):
    """Write beside the summary, retaining identical pages and leaving legacy stories unlinked."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stories = {}
    reprocessed = set()
    for articles in results.values():
        for article in articles:
            article.pop("processing_page", None)
            if article.get("processing_history"):
                article_id = get_article_id(article)
                article["article_id"] = article_id
                article["processing_page"] = f"story-{article_id}.html"
                histories = stories.setdefault(article_id, [])
                if article.get("new_today") is not False:
                    reprocessed.add(article_id)
                if article["processing_history"] not in histories:
                    histories.append(article["processing_history"])
    for article_id, histories in stories.items():
        path = directory / f"story-{article_id}.html"
        if preserve_cached and path.exists() and article_id not in reprocessed:
            continue
        content = render_processing_page(article_id, histories)
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
