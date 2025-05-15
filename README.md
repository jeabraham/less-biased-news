# News Filter Module

A Python tool for fetching and filtering news articles through NewsAPI, enriched with NLP and OpenAI classification to surface stories about women in leadership roles. It supports multiple query configurations and flexible fallback behaviors (`show-full`, `summarize`, `spin-genders`, or `exclude`).

## Prerequisites

* **Python 3.8+**
* **Git** (optional, for cloning the repo)

## Virtual Environment Setup

Create and activate an isolated virtual environment:

```bash
# Create a venv directory
python3 -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

## Installation

1. **Upgrade pip** (optional):

   ```bash
   pip install --upgrade pip
   ```
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Download spaCy model**:

   ```bash
   python -m spacy download en_core_web_sm
   ```

## Configuration

1. Duplicate and rename `config.yaml.example` to `config.yaml`.
2. Fill in your API keys and adjust query settings. Example `config.yaml`:

```yaml
newsapi:
  api_key: "YOUR_NEWSAPI_KEY"
  language: "en"

openai:
  api_key: "YOUR_OPENAI_KEY"
  model: "gpt-3.5-turbo"
  temperature: 0

queries:
  - name: global_news
    q: "global news"
    page_size: 100
    classification: true
    fallback: exclude  # Options: show-full, summarize, spin-genders, exclude
  - name: moose_jaw
    q: '"Moose Jaw, Saskatchewan"'
    page_size: 50
    classification: false
    fallback: show-full
```

### Fallback Options

* **show-full**: Include the full original article.
* **summarize**: Replace the content with a two-sentence summary.
* **spin-genders**: Rewrite the article with a female-centric spin, expanding on women’s roles.
* **exclude**: Drop the article entirely.

## Obtaining API Keys

* **NewsAPI**: Sign up for a free developer account at [https://newsapi.org/](https://newsapi.org/), then copy your API key from the dashboard.
* **OpenAI**: Log in or sign up at [https://platform.openai.com/](https://platform.openai.com/), navigate to **API Keys** under your account settings, and create a new secret key.

## Usage

Run the module as a script or import its functions:

### Command-line

```bash
# Default: plain text saved to news.txt
python news_filter_module.py

# Print to console instead of saving
python news_filter_module.py --print

# Generate HTML output
python news_filter_module.py --format html

# Specify custom config or output file
python news_filter_module.py --config other_config.yaml --output my_news.html
```

### As a Python module

```python
from news_filter_module import fetch_and_filter, generate_html, format_text, load_config

config = load_config("config.yaml")
news = fetch_and_filter(config)
text = format_text(news)
html = generate_html(news)
```

## Requirements

The required Python packages are listed in `requirements.txt`:

```
PyYAML
spacy
newsapi-python
genderize
openai
```

After installing, ensure spaCy’s English model is in place:

```bash
python -m spacy download en_core_web_sm
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
