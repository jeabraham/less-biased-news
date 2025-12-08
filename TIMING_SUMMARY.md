# Timing Tracking Feature - Quick Start

## What Was Added

This PR adds comprehensive performance timing to `news_filter.py` to help understand where processing time is being spent during long-running operations.

## Key Features

✅ **Per-article timing** - See time breakdown for each article
✅ **Action counting** - Track how many times each task runs  
✅ **Average timing** - Calculate average time per operation in milliseconds
✅ **Model identification** - See which AI model was used (e.g., "gpt-4", "qwen-2")
✅ **Query summaries** - Accumulated stats when each query completes
✅ **Periodic updates** - Automatic logging every 10 minutes
✅ **Final summary** - Complete breakdown at end of run

## Quick Example

After running `python news_filter.py`, check `timing.log`:

```
article: Trudeau announces new policy
  classify_leadership (model qwen-2): 0.2s
  image_analysis: 25.0s
  spin_genders (model gpt-4): 18.0s

accumulated amount for query "Canada":
  classify_leadership (model qwen-2): 5.2s (count: 15, avg: 347ms)
  image_analysis: 140.0s (count: 15, avg: 9333ms)
  spin_genders (model gpt-4): 824.0s (count: 7, avg: 117714ms)
```

This tells you:
- Classification happens 15 times, ~347ms each
- Image analysis happens 15 times, ~9.3 seconds each  
- Gender spinning happens 7 times, ~118 seconds each

## What This Helps With

1. **Find bottlenecks** - See which tasks take the longest on average
2. **Optimize models** - Compare performance of different AI models
3. **Plan resources** - Estimate run times based on counts and averages
4. **Debug slow articles** - Identify specific articles that take longer than average
5. **Estimate costs** - Use operation counts for API cost calculations

## No Configuration Needed

The timing feature is **automatically enabled**. Just run `news_filter.py` as normal:

```bash
python news_filter.py -c config.yaml
```

Results are written to `timing.log` in the same directory.

## More Information

See [TIMING_TRACKING.md](TIMING_TRACKING.md) for complete documentation including:
- Detailed output format explanation
- Customization options
- Troubleshooting guide
- Testing instructions

## Testing

Run the tests to see examples:

```bash
# Unit tests
python test_timing_tracker.py

# Integration test with realistic scenarios
python test_timing_integration.py
```

## Implementation Details

**Files Modified:**
- `news_filter.py` - Integrated timing logging
- `ai_queries.py` - Instrumented all AI functions
- `analyze_image_gender.py` - Instrumented image analysis

**Files Added:**
- `timing_tracker.py` - Core timing module (240 lines)
- `test_timing_tracker.py` - Unit tests (195 lines)
- `test_timing_integration.py` - Integration tests (160 lines)
- `TIMING_TRACKING.md` - Complete documentation (300+ lines)
- `TIMING_SUMMARY.md` - This quick start guide

All tests pass ✓
