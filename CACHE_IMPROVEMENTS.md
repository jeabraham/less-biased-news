# Cache Improvements

## Overview

The cache system has been improved to address issues with recycled news articles appearing as "new" when they're actually old articles from 5-10 days ago that news APIs are republishing.

## Previous Behavior

The old system used two cache files per query:
- `{query_name}_current.json` - Today's articles
- `{query_name}_yesterday.json` - Yesterday's articles

Problems:
1. Articles only stayed in cache for ~2 days maximum
2. Old articles (5-10 days) would appear as "new" when recycled by news APIs
3. Running the program multiple times per day would incorrectly mark earlier articles as "old"
4. Cache was based solely on article title (not robust enough)

## New Behavior

The new system uses a single growing cache file per query:
- `{query_name}_cache.json` - All processed articles with timestamps

Improvements:
1. **Growing Cache**: Articles accumulate indefinitely in cache
2. **Timestamps**: Each article has a `first_seen` timestamp showing when it was first processed
3. **Expiration**: Optional `--expire-older-than` flag to remove old entries
4. **Better Keys**: Articles identified by both URL and title (more unique)
5. **Same-Day Runs**: Multiple runs per day work correctly with `--new-today`

## Cache Structure

```json
{
  "articles": {
    "https://example.com/article||Article Title": {
      "first_seen": "2025-12-07T12:34:56.789012",
      "article_data": {
        "title": "Article Title",
        "url": "https://example.com/article",
        "status": "female_leader",
        "content": "...",
        ...
      }
    }
  }
}
```

## Command Line Usage

### Basic Usage (unchanged)
```bash
# Process all articles
python news_filter.py --config config.yaml

# Only process new articles compared to cache
python news_filter.py --new-today
```

### New Expiration Feature
```bash
# Remove cache entries older than 30 days, then process
python news_filter.py --new-today --expire-older-than 30

# Only expire cache, don't process new articles
python news_filter.py --expire-older-than 30
```

## Migration

The new cache system will automatically start fresh when you first use it:
- Old `*_current.json` and `*_yesterday.json` files will be ignored
- New `*_cache.json` files will be created
- You can delete the old cache files if desired

## Benefits

1. **Accurate New Detection**: Old articles that reappear won't be marked as "new"
2. **Flexible Retention**: Keep articles as long as needed with expiration control
3. **Consistent Behavior**: Multiple runs per day work correctly
4. **Better Tracking**: URL+title combination prevents false matches
5. **Scalable**: Cache grows over time but can be pruned with expiration

## Example Scenarios

### Scenario 1: Daily News Processing
```bash
# Monday: Process articles
python news_filter.py --new-today --expire-older-than 30

# Tuesday: Only new articles from Tuesday are processed
python news_filter.py --new-today --expire-older-than 30

# Even if an article from 10 days ago reappears, it won't be processed
```

### Scenario 2: Multiple Runs Per Day
```bash
# Morning: Process articles
python news_filter.py --new-today

# Evening: Only articles added since morning are processed
python news_filter.py --new-today
# The morning articles are still marked as seen today
```

### Scenario 3: Cache Management
```bash
# Keep cache lean (7 days)
python news_filter.py --new-today --expire-older-than 7

# Keep longer history (90 days)
python news_filter.py --new-today --expire-older-than 90

# No expiration (unlimited cache)
python news_filter.py --new-today
```

## Testing

Run the test suite to verify cache functionality:
```bash
python test_cache_functionality.py
```

Run the demonstration to see cache behavior:
```bash
python demo_cache.py
```

## Technical Details

### Functions Modified
- `get_cache_file_path()` - Returns single cache file path (was `get_cache_file_paths()`)
- `load_cache()` - Loads cache with new structure
- `save_cache()` - Merges new articles, preserves timestamps
- `expire_cache()` - New function to remove old entries
- `get_article_key()` - New function to generate unique article keys
- `fetch_and_filter()` - Updated to use new cache system
- `main()` - Added `expire_days` parameter

### New Command Line Arguments
- `--expire-older-than DAYS` / `-E DAYS` - Expire cache entries older than DAYS

### Backward Compatibility
The new system is not backward compatible with old cache files, but this is intentional as the structure has changed. Old cache files will be ignored and new ones will be created.
