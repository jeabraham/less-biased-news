# Rejection Prompt Feature

## Overview

The rejection prompt feature allows queries to optionally filter articles using a Large Language Model (LLM) before any other processing occurs. This enables more sophisticated content filtering based on quality, relevance, or custom criteria.

## How It Works

### Configuration

1. **Model Configuration** (`config.yaml.example`)
   - Add a `rejection_model` entry in the ollama section to specify which model to use
   - Default: `"qwen2.5:1.5b"` (fast and lightweight for classification)

2. **Prompt Configuration**
   - Add rejection prompts to the `prompts` section
   - Each prompt should instruct the model to respond with "ACCEPT" or "REJECT" as the first word
   - Example: `reject_classification_prompt_1`

3. **Query Configuration**
   - Add `rejection_prompt: <prompt_name>` to any query that should use rejection filtering
   - This setting is completely optional
   - Example: `rejection_prompt: reject_classification_prompt_1`

### Processing Flow

When an article is processed for a query with a rejection_prompt:

1. **Early Evaluation**: Before classification, summarization, or any other processing, the article body is sent to the rejection model with the specified prompt.

2. **Model Decision**: The model returns a response starting with either:
   - `ACCEPT` - Article proceeds to normal processing
   - `REJECT` - Article is marked as excluded and skipped

3. **Logging**: All rejection decisions are logged at INFO level to `news_filter.log`:
   ```
   Article REJECTED: [Title]
   Rejection response: REJECT - Low-quality promotional content
   ```
   or
   ```
   Article ACCEPTED: [Title]
   ```

4. **Article Status**: Rejected articles are marked with `status: "exclude"` and automatically filtered from HTML/text output.

## Example Configuration

```yaml
# In ollama section
ollama:
  enabled: True
  rejection_model: "qwen2.5:1.5b"
  # ... other models ...

# In prompts section
prompts:
  reject_classification_prompt_1: >-
    Determine whether this article should be accepted or rejected based on its content quality and relevance.

    Return exactly one of the following as the FIRST word of the output:
    ACCEPT
    REJECT

    After the first word, you may optionally add a brief explanation.

    ARTICLE:

# In queries section
queries:
  - name: Canada
    provider: mediastack
    q: "Canada"
    rejection_prompt: reject_classification_prompt_1  # Optional
    fallback: short_summary
    # ... other settings ...
```

## Use Cases

### 1. Quality Filtering
Filter out low-quality content like:
- SEO spam
- Promotional/marketing content
- Clickbait articles
- Duplicate or near-duplicate content

### 2. Relevance Filtering
Ensure articles match the query's intent:
- Remove off-topic articles
- Filter out tangentially related content
- Focus on specific aspects of a topic

### 3. Content Policy Enforcement
Apply custom content policies:
- Remove articles with specific types of content
- Filter based on tone or sentiment
- Apply editorial guidelines

## Prompt Design Guidelines

A good rejection prompt should:

1. **Be Clear and Specific**: Clearly define what should be accepted vs rejected
2. **Start with Format Requirements**: Explicitly state that the first word must be ACCEPT or REJECT
3. **Provide Examples**: If possible, include examples of what should be rejected
4. **Allow Explanations**: Request a brief explanation after the decision for logging purposes

Example prompt structure:
```
Your task is to [describe the classification task].

Return exactly one of the following as the FIRST word of the output:
ACCEPT
REJECT

After the first word, you may optionally add a brief explanation.

[Optional: Provide criteria or examples]

ARTICLE:
```

## Implementation Details

### Code Structure

- **Function**: `reject_classification()` in `ai_queries.py`
- **Integration**: Called at the beginning of `categorize_article_and_generate_content()` in `news_filter.py`
- **Model Selection**: Uses the task parameter "rejection" to select the `rejection_model`

### Backward Compatibility

- Queries without `rejection_prompt` work exactly as before
- No changes to existing functionality
- Fully optional feature

### Error Handling

- If prompt name is not found in config: Returns (False, "") and logs error
- If no LLM provider is available: Returns (False, "ERROR: No LLM provider available") and logs error
- Errors during classification: Falls back to accepting the article

### Performance Considerations

- Uses a lightweight model by default (qwen2.5:1.5b)
- Minimal token generation (200 tokens max)
- Early rejection reduces processing time for excluded articles
- Can significantly reduce downstream processing costs

## Testing

Two test files are provided:

1. **`test_rejection_integration.py`**: Tests configuration structure and imports
   - Verifies config.yaml.example has correct structure
   - Checks that rejection_model and prompts are configured
   - Validates optional nature of rejection_prompt
   - Run: `python3 test_rejection_integration.py`

2. **`test_rejection_prompt.py`**: Functional tests (requires LLM service)
   - Tests classification with good and bad articles
   - Requires Ollama or OpenAI to be configured
   - Run: `python3 test_rejection_prompt.py`

## Logging

All rejection decisions are logged to `news_filter.log`:

```
2025-12-08 17:00:00,000 [INFO] news_filter: Checking article 'Article Title' for rejection using prompt: reject_classification_prompt_1
2025-12-08 17:00:01,500 [INFO] news_filter: Article REJECTED: Article Title
2025-12-08 17:00:01,501 [INFO] news_filter: Rejection response: REJECT - This article is primarily promotional content without substantial news value.
```

This creates an audit trail for understanding which articles were filtered and why.

## Future Enhancements

Possible future improvements:

1. **Multiple Rejection Stages**: Support for sequential rejection prompts
2. **Rejection Statistics**: Add counters to track rejection rates per query
3. **Conditional Rejection**: Rejection based on query-specific criteria
4. **Rejection Caching**: Cache rejection decisions to avoid re-evaluating the same articles
5. **Custom Rejection Models**: Support different models per prompt or per query

## Troubleshooting

### Articles Not Being Rejected
- Check that the prompt name matches exactly
- Verify the prompt is in the config's prompts section
- Check the logs to see what the model is returning
- Ensure the model response starts with "REJECT" or "ACCEPT"

### All Articles Being Rejected
- Review the rejection prompt for clarity
- Test with the provided test script to see how the model classifies different articles
- Consider adjusting the prompt or using a different model

### Performance Issues
- Use a faster/smaller model for rejection (default qwen2.5:1.5b is recommended)
- Reduce max_tokens if the model is generating too much output
- Consider caching rejection decisions (future enhancement)

## Security Considerations

- No user input is directly passed to the LLM (only article content from trusted news sources)
- Rejection responses are logged but not exposed to end users
- No sensitive data is processed or stored
- Security scanning (CodeQL) found no vulnerabilities in the implementation
