# Timing Tracking Feature

## Overview

The timing tracking feature provides detailed performance metrics for `news_filter.py` operations. It tracks the time spent on various tasks like image analysis, AI classification, summarization, and gender spinning, helping you understand where processing time is being spent.

## Features

- **Per-article timing**: Logs timing for each processed article
- **Task-level tracking**: Tracks time for each operation (image analysis, classification, summarization, etc.)
- **Model identification**: Shows which AI model was used for each task
- **Query accumulation**: Accumulates timing by query
- **Total accumulation**: Tracks total timing across all articles
- **Periodic updates**: Writes accumulated timing every 10 minutes
- **Query completion**: Writes accumulated timing when each query completes

## Output Format

The timing information is written to `timing.log` in the following format:

### Per-Article Timing
```
article: Trudeau announces new policy
  classify_leadership (ollama model qwen-2): 0.2s
  clean_article (ollama model llama-3): 0.5s
  image_analysis: 25.0s
  spin_genders (ollama model gpt-4): 18.0s
```

### Query Completion Summary
```
--------------------------------------------------------------------------------
accumulated amount for query "Canada":
  classify_leadership (ollama model qwen-2): 5.2s
  clean_article (ollama model llama-3): 12.5s
  clean_summary (ollama model gpt-4): 45.0s
  image_analysis: 140.0s
  spin_genders (ollama model gpt-4): 824.0s

accumulated amount for all articles so far:
  classify_leadership (ollama model qwen-2): 15.2s
  clean_article (ollama model llama-3): 42.5s
  clean_summary (ollama model gpt-4): 145.0s
  image_analysis: 530.0s
  spin_genders (ollama model gpt-4): 5230.0s
--------------------------------------------------------------------------------
```

### Periodic Update (Every 10 Minutes)
```
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PERIODIC UPDATE (2025-12-07 23:28:40)
accumulated amount for all articles so far:
  classify_leadership (ollama model qwen-2): 15.2s
  clean_article (ollama model llama-3): 42.5s
  image_analysis: 530.0s
  spin_genders (ollama model gpt-4): 5230.0s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

### Final Summary
```
================================================================================
FINAL SUMMARY (2025-12-07 23:28:42)
accumulated amount for all articles:
  classify_leadership (ollama model qwen-2): 15.2s
  clean_article (ollama model llama-3): 42.5s
  clean_summary (ollama model gpt-4): 145.0s
  image_analysis: 530.0s
  spin_genders (ollama model gpt-4): 5230.0s
================================================================================
```

## Usage

The timing feature is **automatically enabled** when you run `news_filter.py`. No additional configuration is needed.

### Running news_filter.py

```bash
python news_filter.py -c config.yaml
```

The timing information will be written to `timing.log` in the same directory.

### Viewing Timing Results

```bash
# View the entire timing log
cat timing.log

# View just the final summary
tail -n 20 timing.log

# Follow the timing log in real-time
tail -f timing.log
```

## Tracked Tasks

The following tasks are automatically tracked:

1. **image_analysis**: Time spent analyzing images for gender detection
2. **classify_leadership**: Time spent classifying articles for female leadership
3. **clean_article**: Time spent cleaning article text
4. **short_summary**: Time spent generating short summaries
5. **clean_summary**: Time spent generating clean summaries
6. **spin_genders**: Time spent applying gender-neutral rewrites
7. **add_background_on_women**: Time spent adding background information

Each task that uses an AI model will include the model name (e.g., `gpt-4`, `qwen-2`, `llama-3`).

## Implementation Details

### Architecture

The timing tracking is implemented using:

- **`timing_tracker.py`**: Core timing tracking module with thread-safe operations
- **Context managers**: `time_task()` context manager for easy timing of code blocks
- **Automatic logging**: Integrated into the article processing pipeline

### Thread Safety

The timing tracker is thread-safe and can be used in multi-threaded environments (though `news_filter.py` currently processes articles sequentially).

### Performance Impact

The timing tracking has minimal performance impact:
- Uses Python's built-in `time.time()` for measurements
- File I/O is buffered and happens after each article (not during tasks)
- No external dependencies required

## Testing

The timing feature includes comprehensive tests:

```bash
# Run unit tests
python test_timing_tracker.py

# Run integration test
python test_timing_integration.py
```

## Customization

### Changing the Log File Location

Edit `news_filter.py` and modify the tracker initialization:

```python
# In fetch_and_filter() function
tracker = get_timing_tracker()
# Change to:
tracker = TimingTracker(log_file="custom_timing.log")
```

### Changing the Periodic Update Interval

The default is 10 minutes (600 seconds). To change it, edit `timing_tracker.py`:

```python
# In TimingTracker.__init__()
self.periodic_interval = 600  # Change to desired seconds
```

### Disabling Timing

To disable timing tracking, you can comment out the timing-related code in `news_filter.py`:

```python
# Comment out these lines:
# tracker.log_article_timing(art.get("title", "Unknown"), name)
# tracker.log_query_complete(name)
# tracker.finalize()
```

## Troubleshooting

### No timing.log File Created

- Ensure you have write permissions in the directory
- Check that `news_filter.py` is running without errors
- Verify that articles are actually being processed

### Timing Seems Inaccurate

- Timing includes all overhead (network requests, disk I/O, etc.)
- Image analysis timing includes downloading the image
- AI query timing includes network latency if using remote APIs

### Log File Growing Too Large

- The log file appends to existing content
- Consider rotating or archiving old timing logs periodically
- Each session starts with a header for easy identification

## Example Output

Here's what you might see after processing articles for a few hours:

```
================================================================================
Timing Log Session Started: 2025-12-07 12:00:00
================================================================================

article: Breaking: New Climate Policy Announced
  image_analysis: 23.5s
  classify_leadership (ollama model qwen-2): 0.3s
  clean_article (ollama model llama-3): 1.2s
  spin_genders (ollama model gpt-4): 15.8s

article: Tech CEO Steps Down After Scandal
  image_analysis: 28.1s
  classify_leadership (ollama model qwen-2): 0.2s
  clean_article (ollama model llama-3): 0.9s
  clean_summary (ollama model gpt-4): 22.3s

[... more articles ...]

--------------------------------------------------------------------------------
accumulated amount for query "Breaking News":
  classify_leadership (ollama model qwen-2): 5.2s
  clean_article (ollama model llama-3): 12.5s
  clean_summary (ollama model gpt-4): 145.0s
  image_analysis: 340.0s
  spin_genders (ollama model gpt-4): 824.0s

accumulated amount for all articles so far:
  classify_leadership (ollama model qwen-2): 15.2s
  clean_article (ollama model llama-3): 42.5s
  clean_summary (ollama model gpt-4): 345.0s
  image_analysis: 1530.0s
  spin_genders (ollama model gpt-4): 2230.0s
--------------------------------------------------------------------------------

[... periodic updates ...]

================================================================================
FINAL SUMMARY (2025-12-07 23:45:30)
accumulated amount for all articles:
  classify_leadership (ollama model qwen-2): 45.2s
  clean_article (ollama model llama-3): 142.5s
  clean_summary (ollama model gpt-4): 1245.0s
  image_analysis: 5530.0s
  spin_genders (ollama model gpt-4): 12230.0s
================================================================================
```

This shows that over the entire run:
- Image analysis took ~1.5 hours total
- Gender spinning took ~3.4 hours total
- Classification was very fast (~45 seconds total)
- The bulk of the time was spent on AI-heavy tasks

## Benefits

1. **Identify bottlenecks**: See which tasks are taking the most time
2. **Optimize model selection**: Compare performance of different AI models
3. **Plan resources**: Understand how long full runs will take
4. **Debug issues**: Identify articles or queries that are unusually slow
5. **Monitor progress**: Watch timing updates in real-time to gauge completion

## Future Enhancements

Possible improvements for future versions:

- CSV export for easier analysis in spreadsheets
- Graphical visualization of timing data
- Comparison between runs
- Alerting for unusually slow operations
- Per-model cost tracking (if using paid APIs)
