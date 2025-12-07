#!/usr/bin/env python3
"""
Integration test to demonstrate timing tracking in a realistic scenario.
This simulates how timing would work when processing news articles.
"""

import os
import time
import tempfile
from timing_tracker import TimingTracker


def simulate_image_analysis(tracker, num_images=2):
    """Simulate image analysis task."""
    for i in range(num_images):
        with tracker.time_task("image_analysis"):
            time.sleep(0.05)  # Simulate image processing


def simulate_classify_leadership(tracker, model="qwen-2"):
    """Simulate leadership classification task."""
    with tracker.time_task("classify_leadership", model):
        time.sleep(0.02)  # Simulate classification


def simulate_spin_genders(tracker, model="gpt-4"):
    """Simulate gender spinning task."""
    with tracker.time_task("spin_genders", model):
        time.sleep(0.15)  # Simulate gender spinning


def simulate_clean_summary(tracker, model="gpt-4"):
    """Simulate clean summary task."""
    with tracker.time_task("clean_summary", model):
        time.sleep(0.1)  # Simulate summarization


def simulate_clean_article(tracker, model="llama-3"):
    """Simulate article cleaning task."""
    with tracker.time_task("clean_article", model):
        time.sleep(0.08)  # Simulate cleaning


def simulate_article_processing(tracker, article_title, query_name, is_female_leader=False):
    """Simulate processing a single article."""
    # Every article gets image analysis
    simulate_image_analysis(tracker, num_images=2)
    
    # Classify leadership
    simulate_classify_leadership(tracker, "qwen-2")
    
    # Clean article
    simulate_clean_article(tracker, "llama-3")
    
    if is_female_leader:
        # Female leader articles get clean summary
        simulate_clean_summary(tracker, "gpt-4")
    else:
        # Other articles might get gender spinning
        simulate_spin_genders(tracker, "gpt-4")
    
    # Log the article timing
    tracker.log_article_timing(article_title, query_name)


def main():
    """Run integration test."""
    print("=" * 80)
    print("Timing Tracker Integration Test")
    print("Simulating news article processing with various AI tasks")
    print("=" * 80)
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        temp_log = f.name
    
    try:
        tracker = TimingTracker(log_file=temp_log)
        
        # Simulate processing articles for "Canada" query
        print("\nProcessing articles for query 'Canada'...")
        simulate_article_processing(tracker, "Trudeau announces new policy", "Canada", is_female_leader=False)
        simulate_article_processing(tracker, "Chrystia Freeland leads economic reform", "Canada", is_female_leader=True)
        simulate_article_processing(tracker, "Canadian housing crisis deepens", "Canada", is_female_leader=False)
        
        # Complete Canada query
        tracker.log_query_complete("Canada")
        print("✓ Query 'Canada' completed")
        
        # Simulate processing articles for "Tech" query
        print("\nProcessing articles for query 'Tech'...")
        simulate_article_processing(tracker, "Meta CEO discusses AI strategy", "Tech", is_female_leader=False)
        simulate_article_processing(tracker, "YouTube CEO Susan Wojcicki steps down", "Tech", is_female_leader=True)
        
        # Complete Tech query
        tracker.log_query_complete("Tech")
        print("✓ Query 'Tech' completed")
        
        # Finalize
        tracker.finalize()
        print("\n✓ Timing tracking finalized")
        
        # Display the timing log
        print("\n" + "=" * 80)
        print("Generated timing.log content:")
        print("=" * 80)
        with open(temp_log, 'r') as f:
            content = f.read()
            print(content)
        
        # Verify key elements are present
        print("=" * 80)
        print("Verification:")
        print("=" * 80)
        
        checks = [
            ("article: Trudeau announces new policy", "✓ Article titles logged"),
            ("image_analysis:", "✓ Image analysis timings logged"),
            ("spin_genders (ollama model gpt-4):", "✓ Spin genders with model logged"),
            ("classify_leadership (ollama model qwen-2):", "✓ Classification with model logged"),
            ("clean_summary (ollama model gpt-4):", "✓ Clean summary with model logged"),
            ('accumulated amount for query "Canada"', "✓ Query accumulation logged"),
            ('accumulated amount for query "Tech"', "✓ Tech query logged"),
            ("accumulated amount for all articles so far:", "✓ Total accumulation logged"),
            ("FINAL SUMMARY", "✓ Final summary logged"),
        ]
        
        all_passed = True
        for check_str, msg in checks:
            if check_str in content:
                print(msg)
            else:
                print(f"✗ Failed: {msg}")
                all_passed = False
        
        if all_passed:
            print("\n" + "=" * 80)
            print("All verification checks passed! ✓")
            print("=" * 80)
            return 0
        else:
            print("\n✗ Some verification checks failed")
            return 1
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up
        if os.path.exists(temp_log):
            os.unlink(temp_log)


if __name__ == "__main__":
    exit(main())
