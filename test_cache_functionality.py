#!/usr/bin/env python3
"""
Test the new cache functionality with timestamps and expiration.
Tests:
1. Cache article with timestamp
2. Load cached article
3. Detect new vs. cached articles
4. Expire old cache entries
"""

import os
import sys
import json
import tempfile
import shutil
from datetime import datetime, timedelta, date
from pathlib import Path

# Inline cache functions for testing (copied from news_filter.py to avoid dependencies)

def get_cache_file_path(query_name: str) -> str:
    """Get the path for the cache file for a given query name."""
    base_dir = "cache"
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
        else:
            # Update article data but keep original first_seen timestamp
            existing_cache["articles"][article_key]["article_data"] = article
    
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
    original_count = len(cache["articles"])
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
        del cache["articles"][key]
    
    if expired_keys:
        print(f"Expired {len(expired_keys)} cache entries older than {expire_days} days")
        # Save updated cache
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(cache, file, indent=4)
    else:
        print(f"No cache entries to expire (checked {original_count} entries)")


def test_cache_basic_operations():
    """Test basic cache save and load operations."""
    print("\n=== Testing Basic Cache Operations ===")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override the cache directory for testing
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Test cache file path generation
        cache_file = os.path.join(cache_dir, "test_query_cache.json")
        
        # Test article data
        article1 = {
            "title": "Test Article 1",
            "url": "https://example.com/article1",
            "description": "This is a test article",
            "status": "female_leader",
            "content": "Test content 1"
        }
        
        article2 = {
            "title": "Test Article 2",
            "url": "https://example.com/article2",
            "description": "Another test article",
            "status": "short_summary",
            "content": "Test content 2"
        }
        
        # Test saving to cache
        print("✓ Saving articles to cache...")
        save_cache(cache_file, {"articles": [article1, article2]})
        
        # Verify file was created
        assert os.path.exists(cache_file), "Cache file was not created"
        print("✓ Cache file created successfully")
        
        # Test loading from cache
        print("✓ Loading cache...")
        cache = load_cache(cache_file)
        
        assert "articles" in cache, "Cache missing 'articles' key"
        assert len(cache["articles"]) == 2, f"Expected 2 articles, got {len(cache['articles'])}"
        print(f"✓ Loaded {len(cache['articles'])} articles from cache")
        
        # Verify timestamps exist
        for key, entry in cache["articles"].items():
            assert "first_seen" in entry, f"Article {key} missing 'first_seen' timestamp"
            assert "article_data" in entry, f"Article {key} missing 'article_data'"
            print(f"✓ Article '{entry['article_data']['title']}' has timestamp: {entry['first_seen']}")
        
        # Test adding a new article to existing cache
        article3 = {
            "title": "Test Article 3",
            "url": "https://example.com/article3",
            "description": "Third test article",
            "status": "spin_genders",
            "content": "Test content 3"
        }
        
        print("✓ Adding new article to existing cache...")
        save_cache(cache_file, {"articles": [article1, article2, article3]})
        
        # Verify cache was updated, not replaced
        cache = load_cache(cache_file)
        assert len(cache["articles"]) == 3, f"Expected 3 articles after merge, got {len(cache['articles'])}"
        print(f"✓ Cache now contains {len(cache['articles'])} articles")
        
        # Verify article key generation
        key1 = get_article_key(article1)
        key2 = get_article_key(article2)
        assert key1 != key2, "Article keys should be different"
        print(f"✓ Article keys are unique")
        
        print("\n✅ All basic cache operations tests passed!")
        return True


def test_cache_timestamp_persistence():
    """Test that timestamps persist across multiple saves."""
    print("\n=== Testing Timestamp Persistence ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "test_query_cache.json")
        
        article = {
            "title": "Persistent Article",
            "url": "https://example.com/persistent",
            "description": "Test persistence",
            "status": "female_leader",
            "content": "Content"
        }
        
        # First save
        print("✓ Saving article for the first time...")
        save_cache(cache_file, {"articles": [article]})
        
        # Load and get the timestamp
        cache = load_cache(cache_file)
        article_key = get_article_key(article)
        first_timestamp = cache["articles"][article_key]["first_seen"]
        print(f"✓ First timestamp: {first_timestamp}")
        
        # Wait a moment and save again
        import time
        time.sleep(0.1)
        
        # Update the article with new content
        article["content"] = "Updated content"
        print("✓ Updating article content...")
        save_cache(cache_file, {"articles": [article]})
        
        # Load and verify timestamp hasn't changed
        cache = load_cache(cache_file)
        second_timestamp = cache["articles"][article_key]["first_seen"]
        print(f"✓ Second timestamp: {second_timestamp}")
        
        assert first_timestamp == second_timestamp, "Timestamp should not change on update"
        assert cache["articles"][article_key]["article_data"]["content"] == "Updated content", \
            "Article content should be updated"
        
        print("✓ Timestamp persisted correctly while content was updated")
        print("\n✅ Timestamp persistence test passed!")
        return True


def test_cache_expiration():
    """Test cache expiration functionality."""
    print("\n=== Testing Cache Expiration ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "test_query_cache.json")
        
        # Create cache with manually set timestamps
        old_time = (datetime.now() - timedelta(days=15)).isoformat()
        recent_time = (datetime.now() - timedelta(days=3)).isoformat()
        current_time = datetime.now().isoformat()
        
        cache = {
            "articles": {
                "old_article": {
                    "first_seen": old_time,
                    "article_data": {
                        "title": "Old Article",
                        "url": "https://example.com/old",
                        "content": "Old content"
                    }
                },
                "recent_article": {
                    "first_seen": recent_time,
                    "article_data": {
                        "title": "Recent Article",
                        "url": "https://example.com/recent",
                        "content": "Recent content"
                    }
                },
                "current_article": {
                    "first_seen": current_time,
                    "article_data": {
                        "title": "Current Article",
                        "url": "https://example.com/current",
                        "content": "Current content"
                    }
                }
            }
        }
        
        # Save the cache
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=4)
        
        print(f"✓ Created cache with 3 articles (ages: 15 days, 3 days, current)")
        
        # Test expiring entries older than 10 days
        print("✓ Expiring entries older than 10 days...")
        expire_cache(cache_file, expire_days=10)
        
        # Load and verify
        cache = load_cache(cache_file)
        assert len(cache["articles"]) == 2, f"Expected 2 articles after expiration, got {len(cache['articles'])}"
        
        # Verify the old article was removed
        assert "old_article" not in cache["articles"], "Old article should be expired"
        assert "recent_article" in cache["articles"], "Recent article should still be in cache"
        assert "current_article" in cache["articles"], "Current article should still be in cache"
        
        print("✓ Old article (15 days) was removed")
        print("✓ Recent article (3 days) was kept")
        print("✓ Current article was kept")
        
        # Test expiring with shorter threshold (2 days - should only keep current)
        print("✓ Expiring entries older than 2 days...")
        expire_cache(cache_file, expire_days=2)
        
        cache = load_cache(cache_file)
        assert len(cache["articles"]) == 1, f"Expected 1 article after second expiration, got {len(cache['articles'])}"
        assert "current_article" in cache["articles"], "Only current article should remain"
        assert "recent_article" not in cache["articles"], "Recent article (3 days) should be expired"
        
        print("✓ Recent article (3 days) was removed")
        print("✓ Only current article remains")
        print("\n✅ Cache expiration test passed!")
        return True


def test_article_key_uniqueness():
    """Test that article keys are properly unique."""
    print("\n=== Testing Article Key Uniqueness ===")
    
    # Same URL, different title
    article1 = {"url": "https://example.com/article", "title": "Title 1"}
    article2 = {"url": "https://example.com/article", "title": "Title 2"}
    
    key1 = get_article_key(article1)
    key2 = get_article_key(article2)
    
    assert key1 != key2, "Different titles should produce different keys"
    print("✓ Same URL, different titles produce different keys")
    
    # Different URL, same title
    article3 = {"url": "https://example.com/article1", "title": "Same Title"}
    article4 = {"url": "https://example.com/article2", "title": "Same Title"}
    
    key3 = get_article_key(article3)
    key4 = get_article_key(article4)
    
    assert key3 != key4, "Different URLs should produce different keys"
    print("✓ Same title, different URLs produce different keys")
    
    # Same article should produce same key
    article5 = {"url": "https://example.com/same", "title": "Same Article"}
    article6 = {"url": "https://example.com/same", "title": "Same Article"}
    
    key5 = get_article_key(article5)
    key6 = get_article_key(article6)
    
    assert key5 == key6, "Identical articles should produce same key"
    print("✓ Identical articles produce same key")
    
    print("\n✅ Article key uniqueness test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("CACHE FUNCTIONALITY TESTS")
    print("=" * 60)
    
    tests = [
        ("Basic Operations", test_cache_basic_operations),
        ("Timestamp Persistence", test_cache_timestamp_persistence),
        ("Cache Expiration", test_cache_expiration),
        ("Article Key Uniqueness", test_article_key_uniqueness),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n❌ Test '{test_name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, error in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
