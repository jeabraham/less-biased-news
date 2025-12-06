# News Filter Module

## Purpose

A Python tool for fetching and filtering news articles through name- and image-based gender classification and OpenAI summarization.

This project addresses the tendency for news sources to focus on male leaders and men in control—reflecting both media biases and the broader patriarchal structures that persist globally. By applying gender-aware filtering and summarization, it highlights and balances representation of women in news coverage.

## Overview

This module processes news articles in several stages:

1. **Article Fetching**  
   `newsdata_fetcher.py` retrieves articles from configured providers (NewsAPI, MediaStack) based on your `config.yaml` queries.

2. **Content Extraction**  
   `article_fetcher.py` uses Newspaper3k, Readability, and BeautifulSoup to extract the full text and lead image from each article.

3. **Name-based Gender Identification**  
    Name-based tools (`genderize`, `gender-guesser`) predict gender from names appearing in news stories.

4. **Image-based Gender Identification**  
   `analyze_image_gender.py` uses DeepFace (with facenet-pytorch and OpenCV) to detect faces in article images and classify gender (configurable confidence threshold).

5. **OpenAI Summarization & Filtering**  
   `openai_utils.py` sends selected articles to OpenAI (GPT-3.5+/GPT-4) to ask if a women is portrayed in a leadership role.  Open AI is also used to generate summaries (long summaries of articles featuring women leaders, or short summaries of articles that don't), and can “spin” the article to focus more on the women.  These options are controlled and defined in `config.yaml`.

6. **Output Generation**  
   `news_filter.py` orchestrates the pipeline and writes HTML or plain-text output.

## Prerequisites

- **Python 3.12** (3.9 is too old, 3.13 doesn't yet have the binary wheels for some of the packages on MacOS.  Yay Python!, you always make things hard!)
- Instructions are based on a Unix environment, such as **macOS**.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jabraham/less-biased-news.git
   cd less-biased-news
   ```

2. Create and activate a virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Updating Dependencies

If dependencies change, regenerate `requirements.txt`:
```bash
pip freeze > requirements.txt
```

## Configuration

Copy the example file and edit your API keys and query settings:

```bash
cp config.yaml.example config.yaml
```

Key sections in `config.yaml`:

- `newsapi`, `mediastack`, `openai`: Global API credentials and parameters.
- `prompts`: OpenAI prompt templates for summarization and gender “spin.”
- `genderize`: API URL or local settings for name-based gender detection.
- `leadership_keywords`: Keywords used to classify leadership mentions.
- `queries`:  
  - `name`: Identifier for this news source or topic.  
  - `provider`: `newsapi` or `mediastack`.  
  - `q`: Search query string (different syntax for each provider).
  - `page_size`: Number of items per request.  
  - `history_days`: How many days back to fetch (default 14) (only for mediastack).
  - `classification`: `true` to apply gender-based filtering; `false` to include all (future functinoality).  
  - `fallback`: One of `show_full`, `short_summary`, `spin_genders`, `exclude`.  
  - `fallback_image_female`: Override fallback if a woman’s face is detected in the lead image.  
  - `summarize_selected`: `true` to summarize articles passing classification, recommended to remove ads, boilerplate, or calls-to-subscribe.

## Usage

Run manually:

```bash
python news_filter.py   --config config.yaml   --format html   --output news_$(date +%Y-%m-%d).html   --log INFO
```

Or use the helper script:

```bash
bin/run_news_filter.sh
```
## Local AI Model Setup

This project supports local AI inference using pre-trained models. For best performance, the system will automatically attempt to download the required local model. However, you can also manually set up a model if desired.

### 1. Automatic Model Download
By default, the project will download the pre-trained model during initialization (if not already present). Ensure the destination path is configured correctly in `config.yaml`:

```yaml
local_model_path: "models/llama-cpp/model.bin"
model_url: "https://huggingface.co/EleutherAI/gpt-neo-2.7B/resolve/main/config.json"
```

When starting the app, the system will:
1. Download the model from `model_url`.
2. Save it at `local_model_path`.

---

### 2. Manual Model Setup
If you prefer manual setup:
1. Download the model manually. For example:
   - From `Hugging Face`: [GPT-Neo 2.7B](https://huggingface.co/EleutherAI/gpt-neo-2.7B)
   - From other repositories or sources.
2. Place the model at the desired path (e.g., `models/llama-cpp/model.bin`) and update the configuration accordingly:

```yaml
local_model_path: "models/your_model_location/model.bin"
```

### 3. Supported Models
- **Transformers-based models** (e.g., GPT-Neo, GPT-J): Uses the `transformers` library for GPUs and CPU inference.
- **Binary models** (e.g., `llama.cpp`): Supports quantized models for CPU-based inference.

With the correct setup, the system will prioritize local inference over fallback methods (e.g., OpenAI API).


## Automation on macOS

### Using `launchd`

1. Create `~/Library/LaunchAgents/com.yourname.newsfilter.plist` with:

   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN"
     "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0">
   <dict>
     <key>Label</key>
     <string>com.yourname.newsfilter</string>
     <key>ProgramArguments</key>
     <array>
       <string>/bin/bash</string>
       <string>/path/to/less-biased-news/bin/run_news_filter.sh</string>
     </array>
     <key>StartCalendarInterval</key>
     <dict>
       <key>Hour</key><integer>7</integer>
       <key>Minute</key><integer>30</integer>
     </dict>
     <key>StandardOutPath</key>
     <string>/tmp/newsfilter.out</string>
     <key>StandardErrorPath</key>
     <string>/tmp/newsfilter.err</string>
   </dict>
   </plist>
   ```

2. Load the job:
   ```bash
   launchctl load ~/Library/LaunchAgents/com.yourname.newsfilter.plist
   ```

3. Unload with:
   ```bash
   launchctl unload ~/Library/LaunchAgents/com.yourname.newsfilter.plist
   ```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
