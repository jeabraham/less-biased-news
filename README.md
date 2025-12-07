# News Filter Module

## Purpose

A Python tool for fetching and filtering news articles through name- and image-based gender classification and AI-powered analysis.

This project addresses the tendency for news sources to focus on male leaders and men in control—reflecting both media biases and the broader patriarchal structures that persist globally. By applying gender-aware filtering and summarization, it highlights and balances representation of women in news coverage.

## 🚀 Recent Enhancements

- **✅ Full Ollama Support** - Run completely locally without OpenAI API keys
- **✅ Task-Specific Models** - Configure different models for classification, summaries, and rewriting
- **✅ Automatic Fallbacks** - Graceful degradation between models if one fails
- **✅ Ubuntu Makefile** - One-command installation and management on Ubuntu
- **✅ Comprehensive Testing** - New test suite for validating LLM prompts
- **✅ Six LLM Operations** - Classification, short/clean summaries, spinning, cleaning, and context addition

**Documentation:**
- See [OLLAMA_SETUP.md](OLLAMA_SETUP.md) for detailed Ollama installation, configuration, troubleshooting, and model selection
- See [MAKEFILE.md](MAKEFILE.md) for Ubuntu Makefile usage, targets, and advanced configuration

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
- Instructions are based on a Unix environment, such as **macOS** or **Ubuntu Linux**.
- For Ubuntu, see the [Quick Ubuntu Installation](#quick-ubuntu-installation-with-makefile) section below.

## Quick Ubuntu Installation with Makefile

For Ubuntu users, we provide a comprehensive Makefile that automates the entire installation process:

```bash
# Clone the repository
git clone https://github.com/jabraham/less-biased-news.git
cd less-biased-news

# Check system requirements
make check-system

# Complete installation (system packages, Python deps, and Ollama)
make install

# Start Ollama and pull the model
make start-ollama
make pull-ollama-model

# Run tests
make test

# Run the application
make run
```

**Available Makefile targets:**
- `make help` - Show all available commands
- `make install` - Complete installation
- `make check-system` - Verify system requirements
- `make install-ollama` - Install Ollama for local LLM
- `make check-ollama` - Check Ollama status
- `make test` - Run test suite
- `make run` - Run the news filter
- `make clean` - Clean temporary files
- `make uninstall` - Remove virtual environment

For detailed information, run `make help` or see [MAKEFILE.md](MAKEFILE.md).

## Manual Installation

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

## Using Ollama (Recommended for Local LLM)

This project has **full support for [Ollama](https://ollama.ai/)** as a local LLM provider, with NO OpenAI API key required. Ollama is the **recommended approach** for several reasons:

- ✅ **No API costs** - Run unlimited queries locally
- ✅ **Complete privacy** - Your data never leaves your machine  
- ✅ **Easy setup** - Simpler than transformers-based models
- ✅ **Task-specific models** - Configure different models for different operations
- ✅ **Fallback support** - Automatic fallback between models
- ✅ **Full feature parity** - All six LLM operations supported:
  - Article classification (leadership detection)
  - Short summaries
  - Clean/detailed summaries  
  - Gender-focused article rewriting ("spinning")
  - Article cleaning (removing ads/boilerplate)
  - Background context addition

**Ubuntu users:** Simply run `make install-ollama` and `make pull-ollama-model`.

### 1. Install Ollama

Follow the installation instructions at [https://ollama.ai/](https://ollama.ai/):

- **macOS/Linux**: `curl -fsSL https://ollama.ai/install.sh | sh`
- **Windows**: Download from [https://ollama.ai/download](https://ollama.ai/download)

### 2. Pull a Model

The project works well with uncensored models like `llama3-lexi-uncensored`. Pull the model:

```bash
ollama pull Godmoded/llama3-lexi-uncensored
```

Other recommended models:
- `llama2` - Good general-purpose model
- `mistral` - Fast and efficient
- `llama3` - Latest version of Llama

### 3. Start Ollama Server

Ollama runs as a local server. Start it with:

```bash
ollama serve
```

By default, it runs on `http://localhost:11434`.

### 4. Configure the Project

In your `config.yaml`, enable Ollama and disable OpenAI. Here's a simplified example showing diverse models:

```yaml
ollama:
  enabled: True
  base_url: "http://localhost:11434"
  model: "llama3.1:8b"  # Default model
  # Task-specific models (NEW FEATURE: Use different models per task!)
  classification_model: "llama3.1:8b"        # Fast, lightweight model for classification
  short_summary_model: "qwen2.5:7b"          # Efficient model for short summaries
  clean_summary_model: "llama3.1:8b"         # Larger model for detailed summaries
  spin_genders_model: "dolphin3:latest"      # Uncensored model for rewriting
  clean_article_model: "mistral:7b"          # Fast model for article cleaning
  add_background_on_women_model: "llama3.1:8b"  # Knowledgeable model for context
  # Fallback models (NEW FEATURE: Automatic fallback on failures)
  spin_genders_fallback: "mistral:7b"        # Reliable fallback for gender spinning
  summary_fallback: "llama3.1:8b"            # Reliable fallback for summaries
  temperature: 0.1
  max_tokens: 8192

openai:
  api_key: ""  # Leave empty when using Ollama
```

**Note:** The `config.yaml.example` file contains additional advanced models and configurations. Start with these simpler models, then customize based on your needs and available resources.

### 5. Test Ollama

You can test that Ollama is working:

```bash
python test_llm_prompts.py --config config.yaml --num-articles 5 --fetch
```

## Local AI Model Setup (Alternative)

This project also supports local AI inference using pre-trained models from Hugging Face. However, Ollama (above) is recommended as it's easier to set up.

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

## Testing LLM Prompts

The project includes a test script to help you fine-tune your LLM prompts for the four main operations: classification, short_summary, spin_genders, and clean_summary. This is especially useful when switching between different LLMs (OpenAI, Ollama, local models).

### Running the Test

```bash
python test_llm_prompts.py --config config.yaml --num-articles 20 --fetch
```

### Command-line Options

- `--config`: Path to configuration file (default: config.yaml)
- `--num-articles`: Number of articles to test (default: 20)
- `--fetch`: Fetch new articles from configured news sources
- `--cache-folder`: Path to cache folder (default: cache)
- `--output`: Output file for results (default: test_results_<timestamp>.json)
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

### Examples

Test with existing cached articles:
```bash
python test_llm_prompts.py --num-articles 10
```

Fetch fresh articles and test:
```bash
python test_llm_prompts.py --fetch --num-articles 20
```

Test with Ollama and save detailed results:
```bash
python test_llm_prompts.py --fetch --num-articles 15 --output my_test_results.json
```

### Understanding the Output

The test script will:
1. Load or fetch the specified number of articles
2. Test each article against all four LLM prompts
3. Measure execution time for each prompt
4. Display results and save them to a JSON file
5. Print a summary with success rates and average timings

Use the results to:
- Verify your LLM is working correctly
- Compare performance between different models
- Fine-tune prompt wording for better results
- Identify which articles cause problems

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
