#!/usr/bin/env python3
"""
Demonstration of the new cache functionality.
Shows:
1. Articles being cached with timestamps
2. Detection of new vs. cached articles
3. Cache expiration
"""

import os
import sys
import json
import tempfile
from datetime import datetime, timedelta

# Inline cache functions
def get_cache_file_path(query_name: str, base_dir: str = "cache") -> str:
    """Get the path for the cache file for a given query name."""
    cache_file = os.path.join(base_dir, f"{query_name}_cache.json")
    return cache_file

def load_cache(file_path: str) -> dict:
    """Load the cache from the specified file."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {"articles": {}}

def get_article_key(article: dict) -> str:
    """Generate a unique key for an article based on URL and title."""
    url = article.get("url", "")
    title = article.get("title", "")
    return f"{url}||{title}"

def save_cache(file_path: str, data: dict):
    """Save the cache data to a file. Merges new articles with existing cache."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Load existing cache
    existing_cache = load_cache(file_path)
    
    # Get current timestamp
    current_time = datetime.now().isoformat()
    
    # Merge new articles with existing ones
    new_articles = data.get("articles", [])
    for article in new_articles:
        article_key = get_article_key(article)
        
        # Only add if not already in cache (preserves first_seen timestamp)
        if article_key not in existing_cache["articles"]:
            existing_cache["articles"][article_key] = {
                "first_seen": current_time,
                "article_data": article
            }
            print(f"  ✓ Added to cache: {article['title']}")
        else:
            # Update article data but keep original first_seen timestamp
            existing_cache["articles"][article_key]["article_data"] = article
            first_seen = existing_cache["articles"][article_key]["first_seen"]
            print(f"  ⏭  Already in cache (first seen {first_seen[:10]}): {article['title']}")
    
    # Save merged cache
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(existing_cache, file, indent=4)

def expire_cache(file_path: str, expire_days: int):
    """Remove cache entries older than the specified number of days."""
    if not os.path.exists(file_path):
        print(f"Cache file {file_path} does not exist, nothing to expire")
        return
    
    cache = load_cache(file_path)
    
    # Calculate cutoff date
    cutoff = datetime.now() - timedelta(days=expire_days)
    
    # Filter out expired entries
    expired_keys = []
    
    for key, entry in cache["articles"].items():
        first_seen_str = entry.get("first_seen", "")
        if first_seen_str:
            try:
                first_seen = datetime.fromisoformat(first_seen_str)
                if first_seen < cutoff:
                    expired_keys.append(key)
            except ValueError:
                print(f"Invalid timestamp format for cache entry: {first_seen_str}")
    
    # Remove expired entries
    for key in expired_keys:
        title = cache["articles"][key]["article_data"]["title"]
        del cache["articles"][key]
        print(f"  ✗ Expired: {title}")
    
    if expired_keys:
        print(f"\n  Expired {len(expired_keys)} cache entries older than {expire_days} days")
        # Save updated cache
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(cache, file, indent=4)
    else:
        print(f"  No cache entries to expire")


def demo():
    """Demonstrate the cache functionality."""
    print("=" * 70)
    print("CACHE FUNCTIONALITY DEMONSTRATION")
    print("=" * 70)
    
    # Use temp directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = get_cache_file_path("test_news", cache_dir)
        
        # Day 1: Process first batch of articles
        print("\n📅 DAY 1: Processing initial articles")
        print("-" * 70)
        
        articles_day1 = [
            {"title": "Breaking: Important Event", "url": "https://news.com/article1"},
            {"title": "Tech Update Released", "url": "https://tech.com/article2"},
            {"title": "Sports: Team Wins Championship", "url": "https://sports.com/article3"},
        ]
        
        save_cache(cache_file, {"articles": articles_day1})
        
        cache = load_cache(cache_file)
        print(f"\n  Total articles in cache: {len(cache['articles'])}")
        
        # Day 2: Process articles again (some new, some repeats)
        print("\n📅 DAY 2: Processing articles (mix of new and repeated)")
        print("-" * 70)
        
        articles_day2 = [
            {"title": "Breaking: Important Event", "url": "https://news.com/article1"},  # repeat
            {"title": "New Development in Politics", "url": "https://politics.com/article4"},  # new
            {"title": "Weather Alert Issued", "url": "https://weather.com/article5"},  # new
        ]
        
        save_cache(cache_file, {"articles": articles_day2})
        
        cache = load_cache(cache_file)
        print(f"\n  Total articles in cache: {len(cache['articles'])}")
        
        # Manually set an old timestamp for one article to demonstrate expiration
        print("\n🕐 Simulating old article (15 days ago)...")
        print("-" * 70)
        cache = load_cache(cache_file)
        
        # Make one article appear to be 15 days old
        for key, entry in cache["articles"].items():
            if "Sports" in entry["article_data"]["title"]:
                old_time = (datetime.now() - timedelta(days=15)).isoformat()
                entry["first_seen"] = old_time
                print(f"  ⏰ Set '{entry['article_data']['title']}' timestamp to {old_time[:10]}")
                break
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=4)
        
        # Day 3: Process more articles
        print("\n📅 DAY 3: Processing more articles")
        print("-" * 70)
        
        articles_day3 = [
            {"title": "Tech Update Released", "url": "https://tech.com/article2"},  # repeat
            {"title": "International Summit Begins", "url": "https://world.com/article6"},  # new
        ]
        
        save_cache(cache_file, {"articles": articles_day3})
        
        cache = load_cache(cache_file)
        print(f"\n  Total articles in cache: {len(cache['articles'])}")
        
        # Show what's in the cache
        print("\n📋 Current cache contents:")
        print("-" * 70)
        for key, entry in cache["articles"].items():
            title = entry["article_data"]["title"]
            first_seen = entry["first_seen"][:10]
            print(f"  • {title} (first seen: {first_seen})")
        
        # Demonstrate expiration
        print("\n🗑️  Expiring articles older than 10 days...")
        print("-" * 70)
        
        expire_cache(cache_file, expire_days=10)
        
        cache = load_cache(cache_file)
        print(f"\n  Articles remaining in cache: {len(cache['articles'])}")
        
        print("\n📋 Cache contents after expiration:")
        print("-" * 70)
        for key, entry in cache["articles"].items():
            title = entry["article_data"]["title"]
            first_seen = entry["first_seen"][:10]
            print(f"  • {title} (first seen: {first_seen})")
        
        print("\n" + "=" * 70)
        print("DEMONSTRATION COMPLETE")
        print("=" * 70)
        print("\n✅ Key improvements:")
        print("  1. Cache grows over time (not limited to yesterday)")
        print("  2. Articles tracked by URL + title for better uniqueness")
        print("  3. Timestamps preserve when articles were first seen")
        print("  4. Expiration removes old entries based on first_seen date")
        print("  5. Multiple runs per day don't affect new/old detection")
        print("\n💡 Command line usage:")
        print("  python news_filter.py --new-today")
        print("    → Only process articles not in cache")
        print("\n  python news_filter.py --new-today --expire-older-than 30")
        print("    → Remove cache entries older than 30 days, then process new articles")


if __name__ == "__main__":
    demo()
