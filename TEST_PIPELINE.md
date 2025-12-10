# Test Pipeline Tool

## Overview

The `test_pipeline.py` script is a powerful testing tool that allows you to run user-specified sequences of AI processing steps (pipelines) on articles from the cache. This is useful for:

- Testing different processing sequences without fetching new articles
- Validating AI model behavior on specific articles
- Debugging pipeline configurations
- Experimenting with different combinations of processing steps

## Features

- **Multiple Pipelines**: Run multiple different processing sequences on the same articles
- **Query Filtering**: Process specific cached queries or all cache files
- **Date Filtering**: Only process articles newer than a specified date
- **Formatted Output**: Optionally format articles as readable text after each step
- **Comprehensive Logging**: All AI inputs and responses are logged in full to both console and log file

## Installation

The script uses the same dependencies as the main `news_filter.py` application. Ensure you have:

1. Python 3.12 installed
2. All dependencies from `requirements.txt` installed
3. A valid `config.yaml` file with your API keys and settings

## Usage

### Basic Syntax

```bash
python test_pipeline.py [OPTIONS] -p "step1, step2, step3"
```

### Available Pipeline Steps

| Step | Description |
|------|-------------|
| `categorize` | Classify if article is about female leadership |
| `spin_genders` | Rewrite article with gender perspectives |
| `clean_summary` | Generate detailed summary (for female leaders) |
| `short_summary` | Generate brief 2-sentence summary |
| `image_classification` | Analyze images for gender representation |
| `clean_article` | Clean and format article text |
| `add_background` | Add background context about women |

### Command-Line Options

#### Required Options

- `-p`, `--pipeline PIPELINE`: Pipeline of processing steps (comma-separated)
  - Can be specified multiple times for multiple pipelines
  - Example: `-p "categorize, spin_genders, clean_summary"`

#### Optional Options

- `--queries QUERY [QUERY ...]`: List of query names to process
  - Specify one or more query names
  - If not specified, all cache files will be processed
  - Example: `--queries "tech news" "politics"`

- `--newer-than DATE`: Only process articles with first_seen newer than this date
  - Date format: YYYY-MM-DD
  - Example: `--newer-than "2024-12-01"`

- `--format-text`: Format and display article as text after each pipeline step
  - Makes output more readable
  - Uses the same formatting as the email output

- `--config PATH`: Path to configuration file
  - Default: `config.yaml`
  - Example: `--config my_config.yaml`

- `--log LEVEL`: Logging level
  - Choices: DEBUG, INFO, WARNING, ERROR
  - Default: INFO
  - Example: `--log DEBUG`

- `--cache-dir PATH`: Cache directory path
  - Default: `cache`
  - Example: `--cache-dir /path/to/cache`

## Examples

### Example 1: Single Pipeline on All Articles

Process all cached articles with a single pipeline that categorizes, spins gender perspective, and creates a clean summary:

```bash
python test_pipeline.py -p "categorize, spin_genders, clean_summary"
```

### Example 2: Multiple Pipelines on Specific Queries

Run two different pipelines on articles from specific queries:

```bash
python test_pipeline.py \
    --queries "tech news" "politics" \
    -p "categorize, spin_genders, clean_summary" \
    -p "image_classification, categorize, short_summary"
```

This will:
1. Process articles from "tech news" and "politics" caches
2. Run each article through the first pipeline
3. Run each article through the second pipeline

### Example 3: Filter by Date with Formatted Output

Process only recent articles with readable formatting:

```bash
python test_pipeline.py \
    --newer-than "2024-12-01" \
    --format-text \
    -p "categorize, clean_summary"
```

### Example 4: Debug Mode with Custom Config

Run in debug mode with a custom configuration:

```bash
python test_pipeline.py \
    --config config.test.yaml \
    --log DEBUG \
    -p "categorize, spin_genders"
```

### Example 5: Test Image Classification

Test image classification on articles with images:

```bash
python test_pipeline.py \
    --queries "celebrity news" \
    -p "image_classification" \
    --format-text
```

### Example 6: Complex Multi-Step Pipeline

Run a complex pipeline that combines multiple processing steps:

```bash
python test_pipeline.py \
    -p "clean_article, categorize, spin_genders, clean_summary, add_background"
```

## Output Format

The script produces detailed output showing:

1. **Article Information**: Title, URL, first_seen date
2. **Pipeline Steps**: Each step being applied
3. **AI Input**: First 500 characters of input sent to AI
4. **AI Response**: Complete response from the AI model
5. **Formatted Article**: (if `--format-text` is used) Readable article text after each step

### Sample Output

```
################################################################################
# Article 1/2 from query 'tech news'
# Title: Tech CEO breaks barriers
# URL: https://example.com/article1
# First Seen: 2024-12-10T10:00:00
################################################################################

********************************************************************************
* Pipeline 1/1
********************************************************************************

================================================================================
Applying step: categorize
Article: Tech CEO breaks barriers
================================================================================

INPUT (First 500 chars):
Jane Smith, CEO of TechCorp, today announced a revolutionary artificial 
intelligence platform that aims to democratize access to machine learning 
tools. The platform, developed under Smith's leadership over the past three 
years, represents a major breakthrough in the field...

AI RESPONSE:
Is Female Leader: True
Leader Name: Jane Smith

================================================================================
FORMATTED ARTICLE AFTER THIS STEP:
================================================================================

Tech CEO breaks barriers
Status: female_leader (Jane Smith)
https://example.com/article1

Jane Smith, CEO of TechCorp, today announced a revolutionary artificial
intelligence platform that aims to democratize access to machine learning
tools...
```

## Cache File Format

The script expects cache files in the following format:

```json
{
  "query_string": "your search query",
  "articles": {
    "article_key": {
      "first_seen": "2024-12-10T10:00:00",
      "article_data": {
        "title": "Article Title",
        "url": "https://example.com/article",
        "content": "Article content...",
        "body": "Article body...",
        "description": "Article description"
      }
    }
  }
}
```

## Pipeline Step Details

### categorize

Classifies whether an article is about a woman in a leadership position. Uses AI to:
- Detect if the article features a woman leader
- Extract the leader's name if applicable
- Set article status to `female_leader` if confirmed

**Output**: "Is Female Leader: True/False" and optionally "Leader Name: [name]"

### spin_genders

Rewrites the article from a feminist perspective, highlighting women's achievements and contributions. This step:
- Maintains factual accuracy
- Emphasizes women's roles and perspectives
- Balances gender representation in the narrative

**Output**: Rewritten article text

### clean_summary

Creates a detailed, comprehensive summary of the article. Particularly useful for articles about female leaders:
- Focuses on key achievements and contributions
- Maintains important context
- Optimized for longer summaries (multiple paragraphs)

**Output**: Detailed summary text

### short_summary

Generates a brief 2-sentence summary of the article:
- Captures main points
- Concise and readable
- Good for quick article scanning

**Output**: 2-sentence summary

### image_classification

Analyzes images associated with the article for gender representation:
- Detects faces in images
- Classifies gender representation (female, female_majority, female_prominent, etc.)
- Ranks images by prominence of women
- Sets `most_relevant_image` and `most_relevant_status`

**Output**: Image analysis results including status and prominence scores

### clean_article

Cleans and formats the article text by:
- Removing advertisements
- Removing boilerplate text
- Standardizing formatting
- Preserving the actual article content

**Output**: Cleaned article text

### add_background

Adds contextual information about women mentioned in the article:
- Historical context
- Biographical information
- Related achievements
- Connections to broader women's issues

**Output**: Article with added background context

## Tips and Best Practices

1. **Start Simple**: Begin with a single pipeline step to understand the behavior
2. **Use --format-text**: Makes output much more readable during testing
3. **Filter by Date**: Use `--newer-than` to focus on recent articles
4. **Debug Mode**: Use `--log DEBUG` to see detailed processing information
5. **Test Incrementally**: Test each step individually before combining them
6. **Multiple Pipelines**: Use multiple `-p` options to compare different approaches

## Troubleshooting

### No Cache Files Found

**Error**: `No cache files found in cache`

**Solution**: 
- Run `news_filter.py` first to populate the cache
- Check that cache directory exists and contains `*_cache.json` files
- Use `--cache-dir` to specify a different cache location

### Invalid Pipeline Steps

**Error**: `Invalid pipeline steps: unknown_step`

**Solution**: 
- Check the list of valid steps above
- Ensure step names are spelled correctly
- Steps are case-sensitive

### Configuration Not Found

**Error**: `Failed to load configuration: config.yaml does not exist`

**Solution**:
- Copy `config.yaml.example` to `config.yaml`
- Fill in your API keys
- Or use `--config` to specify a different config file

### Date Format Error

**Error**: `Invalid date format: 12-01-2024. Expected YYYY-MM-DD`

**Solution**: Use ISO format: `--newer-than "2024-12-01"`

## Integration with Main Workflow

The test pipeline tool is designed to work alongside the main news filtering workflow:

1. **Fetch and Cache**: Run `news_filter.py` to fetch and cache articles
2. **Test Pipelines**: Use `test_pipeline.py` to test different processing sequences
3. **Refine Configuration**: Adjust your config based on test results
4. **Run Production**: Run `news_filter.py` again with optimized settings

## Performance Notes

- Processing time depends on the AI model used (GPT-4 is slower but higher quality)
- Each pipeline step makes an AI API call (except image_classification)
- Use `--queries` to limit the number of articles processed during testing
- Consider using faster models (gpt-3.5-turbo) for initial testing

## Contributing

If you add new pipeline steps:

1. Add the function to `ai_queries.py` or the appropriate module
2. Import it in `test_pipeline.py`'s `lazy_imports()` function
3. Add a case in `apply_pipeline_step()`
4. Update the `valid_steps` set in `validate_pipeline_steps()`
5. Update this documentation

## See Also

- Main documentation: `README.md`
- Configuration guide: `config.yaml.example`
- Ollama setup: `OLLAMA_SETUP.md`
- Makefile usage: `MAKEFILE.md`
