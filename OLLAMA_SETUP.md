# Ollama Setup Guide

This document provides detailed instructions for setting up and using Ollama as the LLM provider for the less-biased-news project.

## Overview

The project now supports [Ollama](https://ollama.ai/) as a local LLM provider, which eliminates the need for an OpenAI API key. Ollama provides an easy way to run large language models locally.

## Why Ollama?

- **No API costs**: Run models locally without paying per-token
- **Privacy**: Your data never leaves your machine
- **No API key required**: Works completely offline after initial model download
- **Easy to use**: Simple installation and model management
- **Good models available**: Including uncensored models like `llama3-lexi-uncensored`

## Installation

### 1. Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [https://ollama.ai/download](https://ollama.ai/download)

### 2. Pull a Model

The project works well with uncensored models. Recommended model:

```bash
ollama pull Godmoded/llama3-lexi-uncensored
```

**Note:** The model name in Ollama uses a different format. To pull `Godmoded/llama3-lexi-uncensored`, you might need to use:

```bash
ollama pull llama3-lexi-uncensored
```

Other recommended models:
- `llama2` - Good general-purpose model
- `mistral` - Fast and efficient
- `llama3` - Latest version of Llama
- `llama3.1` - Improved version with better reasoning

To see available models:
```bash
ollama list
```

### 3. Start Ollama Server

Start the Ollama server in a terminal:

```bash
ollama serve
```

Leave this running in the background. By default, it runs on `http://localhost:11434`.

**Tip:** On macOS, Ollama may start automatically after installation. Check if it's running:
```bash
curl http://localhost:11434
```

## Configuration

### 1. Edit config.yaml

Copy the example configuration if you haven't already:
```bash
cp config.yaml.example config.yaml
```

### 2. Enable Ollama

In `config.yaml`, update the Ollama section:

```yaml
ollama:
  enabled: True  # Set to True to use Ollama
  base_url: "http://localhost:11434"  # Default Ollama endpoint
  model: "llama3-lexi-uncensored"  # Model for all tasks
  complex_model: "llama3-lexi-uncensored"  # Model for spin_genders and clean_summary
  simple_model: "llama3-lexi-uncensored"  # Model for classification and short_summary
  temperature: 0.1
  max_tokens: 4096
```

### 3. Disable OpenAI (Optional)

Since you're using Ollama, you can leave the OpenAI API key empty:

```yaml
openai:
  api_key: ""  # Leave empty when using Ollama
```

## Testing Your Setup

### Quick Test

Test that Ollama is responding:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3-lexi-uncensored",
  "prompt": "Hello, how are you?",
  "stream": false
}'
```

### Test LLM Prompts

Use the included test script to verify all four prompts work:

```bash
# Test with a few articles from cache
python test_llm_prompts.py --num-articles 5

# Or fetch fresh articles and test
python test_llm_prompts.py --fetch --num-articles 5
```

The test will:
1. Load or fetch articles
2. Test all four prompts (classification, short_summary, clean_summary, spin_genders)
3. Display timing and success rates
4. Save detailed results to a JSON file

### Test Individual Articles

Test the spin_genders prompt on a specific article:

```bash
python test_spin_genders.py test_article.txt
```

## How It Works

The project now has a prioritized LLM provider system:

1. **Ollama** (if enabled) - Used first
2. **Local AI** (transformers) - Used if Ollama not available
3. **OpenAI API** - Used as fallback

When Ollama is enabled:
- No OpenAI API calls are made
- All four prompts (classification, short_summary, clean_summary, spin_genders) use Ollama
- The configured model handles all requests

## Model Selection

You can use different models for different tasks:

```yaml
ollama:
  simple_model: "llama3"  # Fast model for classification and short summaries
  complex_model: "llama3.1"  # Better model for complex rewrites
```

## Troubleshooting

### Ollama Not Found

**Error:** `ModuleNotFoundError: No module named 'ollama'`

**Solution:**
```bash
pip install ollama
```

### Connection Refused

**Error:** Connection refused to `http://localhost:11434`

**Solution:** Make sure Ollama is running:
```bash
ollama serve
```

### Model Not Found

**Error:** Model 'llama3-lexi-uncensored' not found

**Solution:** Pull the model first:
```bash
ollama pull llama3-lexi-uncensored
```

### Slow Performance

If responses are slow:
1. Use a smaller model: `ollama pull llama2`
2. Reduce max_tokens in config.yaml
3. Use GPU if available (Ollama auto-detects)

### Out of Memory

If you get OOM errors:
1. Use a smaller model
2. Close other applications
3. Reduce max_tokens in config.yaml
4. Consider using a quantized model

## Performance Tips

### GPU Acceleration

Ollama automatically uses GPU if available:
- **NVIDIA**: Requires CUDA
- **Apple Silicon**: Uses Metal automatically
- **AMD**: Requires ROCm (Linux only)

Check if GPU is being used:
```bash
ollama ps
```

### Model Size vs. Speed

- Smaller models (7B parameters): Faster, less accurate
- Medium models (13B parameters): Balanced
- Larger models (33B+ parameters): Slower, more accurate

For this project, 7B-13B models work well.

## Running the Full Pipeline

Once Ollama is configured:

```bash
python news_filter.py \
  --config config.yaml \
  --format html \
  --output news_$(date +%Y-%m-%d).html \
  --log INFO
```

Or use the helper script:
```bash
bin/run_news_filter.sh
```

## Comparing Models

Test different models to find the best one:

```bash
# Test with model A
# Edit config.yaml to set model: "llama3"
python test_llm_prompts.py --fetch --num-articles 10 --output results_llama3.json

# Test with model B
# Edit config.yaml to set model: "mistral"
python test_llm_prompts.py --num-articles 10 --output results_mistral.json

# Compare the results files
```

## Fine-tuning Prompts

The test script helps you fine-tune prompts for your chosen model:

1. Run tests with current prompts
2. Review the JSON output
3. Adjust prompts in config.yaml
4. Run tests again
5. Compare results

## Additional Resources

- **Ollama Documentation**: [https://ollama.ai/](https://ollama.ai/)
- **Model Library**: [https://ollama.ai/library](https://ollama.ai/library)
- **Ollama GitHub**: [https://github.com/ollama/ollama](https://github.com/ollama/ollama)

## Getting Help

If you encounter issues:

1. Check Ollama is running: `curl http://localhost:11434`
2. Verify model is installed: `ollama list`
3. Check logs: `tail -f news_filter.log`
4. Test with simple prompt: `ollama run llama3 "Hello"`
5. Run integration test: See "Testing Your Setup" section above

## Reverting to OpenAI

To switch back to OpenAI:

1. Set `ollama.enabled: False` in config.yaml
2. Add your OpenAI API key to `openai.api_key`
3. Restart the application

The system will automatically fall back to OpenAI.
