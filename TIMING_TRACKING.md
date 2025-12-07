# Timing Tracking Feature

## Overview

The timing tracking feature provides detailed performance metrics for `news_filter.py` operations. It tracks the time spent on various tasks like image analysis, AI classification, summarization, and gender spinning, helping you understand where processing time is being spent.

## Features

- **Per-article timing**: Logs timing for each processed article
- **Task-level tracking**: Tracks time for each operation (image analysis, classification, summarization, etc.)
- **Model identification**: Shows which AI model was used for each task
- **Action counting**: Counts how many times each task was performed
- **Average timing**: Calculates and displays average time per task in milliseconds
- **Query accumulation**: Accumulates timing and counts by query
- **Total accumulation**: Tracks total timing and counts across all articles
- **Periodic updates**: Writes accumulated timing every 10 minutes
- **Query completion**: Writes accumulated timing when each query completes

## Output Format

The timing information is written to `timing.log` in the following format:

### Per-Article Timing
```
article: Trudeau announces new policy
  classify_leadership (model qwen-2): 0.2s
  clean_article (model llama-3): 0.5s
  image_analysis: 25.0s
  spin_genders (model gpt-4): 18.0s
```

### Query Completion Summary
```
--------------------------------------------------------------------------------
accumulated amount for query "Canada":
  classify_leadership (model qwen-2): 5.2s (count: 15, avg: 347ms)
  clean_article (model llama-3): 12.5s (count: 15, avg: 833ms)
  clean_summary (model gpt-4): 45.0s (count: 8, avg: 5625ms)
  image_analysis: 140.0s (count: 15, avg: 9333ms)
  spin_genders (model gpt-4): 824.0s (count: 7, avg: 117714ms)

accumulated amount for all articles so far:
  classify_leadership (model qwen-2): 15.2s (count: 45, avg: 338ms)
  clean_article (model llama-3): 42.5s (count: 45, avg: 944ms)
  clean_summary (model gpt-4): 145.0s (count: 25, avg: 5800ms)
  image_analysis: 530.0s (count: 45, avg: 11778ms)
  spin_genders (model gpt-4): 5230.0s (count: 20, avg: 261500ms)
--------------------------------------------------------------------------------
```

### Periodic Update (Every 10 Minutes)
```
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PERIODIC UPDATE (2025-12-07 23:28:40)
accumulated amount for all articles so far:
  classify_leadership (model qwen-2): 15.2s (count: 45, avg: 338ms)
  clean_article (model llama-3): 42.5s (count: 45, avg: 944ms)
  image_analysis: 530.0s (count: 45, avg: 11778ms)
  spin_genders (model gpt-4): 5230.0s (count: 20, avg: 261500ms)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

### Final Summary
```
================================================================================
FINAL SUMMARY (2025-12-07 23:28:42)
accumulated amount for all articles:
  classify_leadership (model qwen-2): 15.2s (count: 45, avg: 338ms)
  clean_article (model llama-3): 42.5s (count: 45, avg: 944ms)
  clean_summary (model gpt-4): 145.0s (count: 25, avg: 5800ms)
  image_analysis: 530.0s (count: 45, avg: 11778ms)
  spin_genders (model gpt-4): 5230.0s (count: 20, avg: 261500ms)
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
  classify_leadership (model qwen-2): 0.3s
  clean_article (model llama-3): 1.2s
  spin_genders (model gpt-4): 15.8s

article: Tech CEO Steps Down After Scandal
  image_analysis: 28.1s
  classify_leadership (model qwen-2): 0.2s
  clean_article (model llama-3): 0.9s
  clean_summary (model gpt-4): 22.3s

[... more articles ...]

--------------------------------------------------------------------------------
accumulated amount for query "Breaking News":
  classify_leadership (model qwen-2): 5.2s (count: 15, avg: 347ms)
  clean_article (model llama-3): 12.5s (count: 15, avg: 833ms)
  clean_summary (model gpt-4): 145.0s (count: 8, avg: 18125ms)
  image_analysis: 340.0s (count: 15, avg: 22667ms)
  spin_genders (model gpt-4): 824.0s (count: 7, avg: 117714ms)

accumulated amount for all articles so far:
  classify_leadership (model qwen-2): 15.2s (count: 45, avg: 338ms)
  clean_article (model llama-3): 42.5s (count: 45, avg: 944ms)
  clean_summary (model gpt-4): 345.0s (count: 25, avg: 13800ms)
  image_analysis: 1530.0s (count: 45, avg: 34000ms)
  spin_genders (model gpt-4): 2230.0s (count: 20, avg: 111500ms)
--------------------------------------------------------------------------------

[... periodic updates ...]

================================================================================
FINAL SUMMARY (2025-12-07 23:45:30)
accumulated amount for all articles:
  classify_leadership (model qwen-2): 45.2s (count: 135, avg: 335ms)
  clean_article (model llama-3): 142.5s (count: 135, avg: 1056ms)
  clean_summary (model gpt-4): 1245.0s (count: 75, avg: 16600ms)
  image_analysis: 5530.0s (count: 135, avg: 40963ms)
  spin_genders (model gpt-4): 12230.0s (count: 60, avg: 203833ms)
================================================================================
```

This shows that over the entire run:
- Image analysis: 135 operations, ~41 seconds average per image
- Gender spinning: 60 operations, ~3.4 minutes average per article
- Classification: 135 operations, very fast at ~335ms average
- Clean summary: 75 operations, ~17 seconds average
- The bulk of the time was spent on AI-heavy tasks (gender spinning and image analysis)

## Benefits

1. **Identify bottlenecks**: See which tasks are taking the most time on average
2. **Optimize model selection**: Compare performance of different AI models using average times
3. **Plan resources**: Understand how long full runs will take based on operation counts and averages
4. **Debug issues**: Identify articles or queries that are unusually slow compared to average
5. **Monitor progress**: Watch timing updates in real-time to gauge completion
6. **Track utilization**: See how many times each task is being called to understand workflow patterns
7. **Cost estimation**: Use operation counts to estimate API costs for paid services

## Future Enhancements

Possible improvements for future versions:

- CSV export for easier analysis in spreadsheets
- Graphical visualization of timing data
- Comparison between runs
- Alerting for unusually slow operations
- Per-model cost tracking (if using paid APIs)
