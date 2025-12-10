# Testing Guide

## Overview

This document describes how to test the Less-Biased News system.

## Test Scripts

### test_pipeline.py - Pipeline Testing Tool

A comprehensive tool for testing article processing pipelines on cached articles. This allows you to:
- Test different processing sequences without fetching new articles
- Validate AI model behavior
- Debug configurations
- Experiment with different pipeline combinations

**Quick Start:**
```bash
# Create sample cache data
mkdir -p cache
# (Add your cache files to the cache/ directory)

# Run a test pipeline
python test_pipeline.py -p "categorize, spin_genders, clean_summary"
```

**Full Documentation:** See [TEST_PIPELINE.md](TEST_PIPELINE.md)

### test_pipeline_validation.py - Structure Validation

Validates that `test_pipeline.py` has the correct structure without requiring all dependencies:

```bash
python test_pipeline_validation.py
```

This checks for:
- Required functions
- Command-line options
- Pipeline steps
- Overall code structure

## Test Data

### Creating Test Cache Files

To test the pipeline tool, you need cache files. You can either:

1. **Use real cache files**: Run `news_filter.py` to fetch and cache real articles
2. **Create test data**: Create JSON files in the `cache/` directory

**Test Cache Format:**
```json
{
  "query_string": "your search query",
  "articles": {
    "https://example.com/article||Article Title": {
      "first_seen": "2024-12-10T10:00:00",
      "article_data": {
        "title": "Article Title",
        "url": "https://example.com/article",
        "description": "Brief description",
        "content": "Full article content...",
        "body": "Article body text...",
        "source": {"name": "News Source"},
        "published_at": "2024-12-10T09:30:00Z"
      }
    }
  }
}
```

Save this as `cache/test_cache.json` to test with the pipeline tool.

## Unit Tests

### Running Existing Tests

The repository includes several test suites:

```bash
# Run all tests in the tests/ directory
python -m pytest tests/

# Run specific test file
python tests/test_cache_functionality.py

# Run LLM prompt tests
python tests/test_llm_prompts.py --config config.yaml --num-articles 5
```

### Test Files

- `test_cache_functionality.py` - Cache management tests
- `test_cache_classifications.py` - Article classification tests
- `test_clean_summary_functionality.py` - Summary generation tests
- `test_spin_genders.py` - Gender perspective rewriting tests
- `test_timing_tracker.py` - Performance tracking tests
- `test_llm_prompts.py` - LLM prompt validation tests

## Integration Testing

### End-to-End Workflow Test

Test the complete workflow from fetching to output:

```bash
# 1. Fetch articles and populate cache
python news_filter.py --config config.yaml

# 2. Test different pipeline configurations
python test_pipeline.py -p "categorize, clean_summary"

# 3. Generate formatted output
python news_formatter.py --format html --output test_output.html
```

### Testing with Different AI Providers

**Test with Ollama (local):**
```bash
# Ensure Ollama is running
ollama serve

# Run with Ollama configuration
python test_pipeline.py --config config.yaml -p "categorize"
```

**Test with OpenAI:**
```bash
# Ensure OpenAI API key is in config.yaml
python test_pipeline.py --config config.yaml -p "categorize"
```

## Validation Scripts

### Validate Configuration

Check that your `config.yaml` is properly formatted:

```bash
python -c "import yaml; yaml.safe_load(open('config.yaml'))" && echo "Config is valid"
```

### Validate Cache Files

Check cache file format:

```bash
python -c "import json; json.load(open('cache/your_cache.json'))" && echo "Cache file is valid"
```

## Performance Testing

### Timing Analysis

The system includes timing tracking. Run with debug logging to see timing information:

```bash
python news_filter.py --log DEBUG --config config.yaml
```

Check the timing summary in `news_filter.log`.

### Benchmarking Different Models

Compare performance of different AI models:

```bash
# Test with gpt-3.5-turbo
# (Edit config.yaml to use gpt-3.5-turbo)
time python test_pipeline.py -p "categorize, clean_summary"

# Test with gpt-4
# (Edit config.yaml to use gpt-4)
time python test_pipeline.py -p "categorize, clean_summary"

# Test with Ollama
# (Edit config.yaml to use Ollama)
time python test_pipeline.py -p "categorize, clean_summary"
```

## Debugging Tips

### Enable Debug Logging

```bash
python test_pipeline.py --log DEBUG -p "categorize"
```

### Check Log Files

Log files are created in the project root:
- `news_filter.log` - Main application log
- Output is also sent to console

### Validate Individual Steps

Test each pipeline step individually:

```bash
# Test categorization
python test_pipeline.py -p "categorize"

# Test summary generation
python test_pipeline.py -p "clean_summary"

# Test gender spinning
python test_pipeline.py -p "spin_genders"
```

### Use Format-Text for Readability

Add `--format-text` to see formatted output after each step:

```bash
python test_pipeline.py -p "categorize, clean_summary" --format-text
```

## Continuous Integration

### GitHub Actions

(If you set up CI/CD, describe it here)

### Pre-commit Checks

Before committing, run:

```bash
# Validate Python syntax
python -m py_compile test_pipeline.py

# Run validation script
python test_pipeline_validation.py

# Run unit tests
python -m pytest tests/
```

## Common Issues

### ModuleNotFoundError

**Problem**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Install all dependencies:
```bash
pip install -r requirements.txt
```

### No Cache Files Found

**Problem**: `No cache files found in cache`

**Solution**: Either run `news_filter.py` first or create test cache files as shown above.

### API Rate Limits

**Problem**: Rate limit errors from AI providers

**Solution**: 
- Use `--queries` to limit the number of articles processed
- Add delays in configuration
- Use Ollama for unlimited local processing

## Contributing Tests

When adding new features:

1. Add validation to `test_pipeline_validation.py` if it's a new pipeline step
2. Create unit tests in `tests/` directory
3. Update this testing guide
4. Document any new test data requirements

## See Also

- [TEST_PIPELINE.md](TEST_PIPELINE.md) - Detailed pipeline testing documentation
- [README.md](README.md) - Main project documentation
- [OLLAMA_SETUP.md](OLLAMA_SETUP.md) - Local AI setup guide
