# Implementation Notes: Cache System Improvements

## Problem Statement

The original issue stated:
> The --new-today option shouldn't just compare to "yesterdays" news. The cache of previously fetched articles should be kept longer. What happens is the news apis start recycling old news from 5 or 10 days ago, so it's shows up in the program as new compared to yesterday, but it's not new at all.

## Requirements Met

✅ **Keep a growing cache of all articles** - Implemented with single cache file per query that accumulates articles over time

✅ **Store date when article was processed** - Each article has `first_seen` timestamp

✅ **--expire-older-than X option** - Implemented as `--expire-older-than DAYS` / `-E DAYS`

✅ **Multiple runs per day work correctly** - Cache structure uses timestamps, not dates, so multiple runs per day don't incorrectly mark articles as old

## Technical Implementation

### Cache Structure Change

**Before:**
```
cache/
  query_name_current.json    # Today's articles only
  query_name_yesterday.json  # Yesterday's articles only
```

**After:**
```
cache/
  query_name_cache.json      # All articles with timestamps
```

### Cache File Format

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

### Key Functions Modified

1. **get_cache_file_path()** - Returns single cache file path (was `get_cache_file_paths()` returning tuple)

2. **load_cache()** - Loads cache with new structure containing articles dictionary

3. **save_cache()** - Merges new articles with existing cache, preserving original `first_seen` timestamps

4. **expire_cache()** - New function to remove entries older than specified days

5. **get_article_key()** - New function generating unique keys from URL + title

6. **fetch_and_filter()** - Updated to use new cache structure and check against all cached articles

### Command Line Changes

**New argument:**
- `--expire-older-than DAYS` / `-E DAYS` - Expire cache entries older than DAYS

**Updated help text:**
- `--new-today` - Now says "only process articles not seen in the cache (compares to all previously cached articles)"

## Testing

### Unit Tests
Created `test_cache_functionality.py` with 4 test suites:
1. Basic cache operations (save, load, merge)
2. Timestamp persistence across updates
3. Cache expiration functionality
4. Article key uniqueness

All tests pass ✅

### Demonstration
Created `demo_cache.py` showing:
- Articles accumulating over multiple runs
- New vs. cached article detection
- Timestamp preservation
- Expiration behavior

### Security
CodeQL scan shows 0 alerts ✅

## Documentation

- **CACHE_IMPROVEMENTS.md** - Detailed explanation of changes, usage examples, migration guide
- **README.md** - Updated with new cache features section
- **IMPLEMENTATION_NOTES.md** - This file

## Backward Compatibility

**Breaking change**: Old cache files (`*_current.json`, `*_yesterday.json`) are no longer used.

**Migration**: No manual migration needed. The new system will:
1. Ignore old cache files
2. Create new cache files on first run
3. Start accumulating articles in new format

Users can delete old cache files manually if desired.

## Examples

### Before (Old Behavior)
```bash
# Day 1: Process articles A, B, C
python news_filter.py --new-today
# Cache: current=[A,B,C], yesterday=[]

# Day 2: Process articles B, C, D
python news_filter.py --new-today
# Cache: current=[B,C,D], yesterday=[A,B,C]
# Only D is processed as new (B and C are in yesterday's cache)

# Day 3: Process articles D, E, F
python news_filter.py --new-today
# Cache: current=[D,E,F], yesterday=[B,C,D]
# E and F are processed as new
# Problem: If article A reappears, it would be processed as new!
```

### After (New Behavior)
```bash
# Day 1: Process articles A, B, C
python news_filter.py --new-today
# Cache contains: A, B, C with timestamps

# Day 2: Process articles B, C, D
python news_filter.py --new-today
# Cache contains: A, B, C, D with timestamps
# Only D is processed as new (B and C are already in cache)

# Day 3: Process articles D, E, F
python news_filter.py --new-today
# Cache contains: A, B, C, D, E, F with timestamps
# Only E and F are processed as new

# Day 10: If article A reappears
python news_filter.py --new-today
# Article A is still in cache, NOT processed as new ✅

# With expiration:
python news_filter.py --new-today --expire-older-than 7
# Articles older than 7 days are removed first, then new articles processed
```

## Performance Considerations

- **Cache size**: Grows linearly with unique articles processed
- **Lookup speed**: O(1) for article existence checks (using dictionary)
- **Memory usage**: Entire cache loaded into memory during processing
- **Disk usage**: JSON format with indentation, ~1-2KB per article

For typical usage (processing daily with expiration), cache should remain manageable (<10MB for months of articles).

## Future Enhancements

Potential improvements:
1. Add statistics command to show cache size, oldest/newest articles
2. Support cache cleanup by query (instead of just by age)
3. Add option to export/import cache
4. Implement cache compression for very large caches
5. Add cache deduplication tool to fix any inconsistencies

## Related Issues

This implementation addresses the problem described where news APIs recycle old articles, causing them to appear as "new" when they're actually 5-10 days old.

## Testing Commands

```bash
# Run unit tests
python test_cache_functionality.py

# Run demonstration
python demo_cache.py

# Test actual program (requires full dependencies)
python news_filter.py --new-today --expire-older-than 30 --config config.yaml
```

## Commit History

1. Initial plan and structure analysis
2. Implement growing cache with timestamps and expiration
3. Add documentation and demonstration
4. Fix code review issues (remove duplicate imports)
