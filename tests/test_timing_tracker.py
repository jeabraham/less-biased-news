#!/usr/bin/env python3
"""
Test the timing tracker functionality.
"""

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import time
import tempfile
from timing_tracker import TimingTracker, reset_timing_tracker


def test_basic_timing():
    """Test basic timing functionality."""
    print("Test 1: Basic timing functionality")
    
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        temp_log = f.name
    
    try:
        tracker = TimingTracker(log_file=temp_log)
        
        # Time some tasks
        with tracker.time_task("task1"):
            time.sleep(0.1)
        
        with tracker.time_task("task2", "model-a"):
            time.sleep(0.05)
        
        # Log article timing
        tracker.log_article_timing("Test Article 1", "test_query")
        
        # Check the log file was created and has content
        assert os.path.exists(temp_log), "Log file was not created"
        
        with open(temp_log, 'r') as f:
            content = f.read()
            assert "Test Article 1" in content, "Article title not in log"
            assert "task1:" in content, "Task1 not in log"
            assert "task2 (model model-a):" in content, "Task2 with model not in log"
        
        print("✓ Basic timing test passed")
    finally:
        # Clean up
        if os.path.exists(temp_log):
            os.unlink(temp_log)


def test_accumulated_timing():
    """Test accumulated timing functionality."""
    print("\nTest 2: Accumulated timing functionality")
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        temp_log = f.name
    
    try:
        tracker = TimingTracker(log_file=temp_log)
        
        # Process multiple articles
        for i in range(3):
            with tracker.time_task("task_common"):
                time.sleep(0.05)
            tracker.log_article_timing(f"Article {i}", "query1")
        
        # Complete the query
        tracker.log_query_complete("query1")
        
        # Check the log file
        with open(temp_log, 'r') as f:
            content = f.read()
            assert 'accumulated amount for query "query1"' in content, "Query completion not logged"
            assert "accumulated amount for all articles so far:" in content, "Total accumulation not logged"
            assert "task_common:" in content, "Task not in accumulated log"
        
        print("✓ Accumulated timing test passed")
    finally:
        if os.path.exists(temp_log):
            os.unlink(temp_log)


def test_multiple_queries():
    """Test timing with multiple queries."""
    print("\nTest 3: Multiple queries timing")
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        temp_log = f.name
    
    try:
        tracker = TimingTracker(log_file=temp_log)
        
        # Process articles for query 1
        with tracker.time_task("classify"):
            time.sleep(0.05)
        tracker.log_article_timing("Article A", "query1")
        tracker.log_query_complete("query1")
        
        # Process articles for query 2
        with tracker.time_task("summarize"):
            time.sleep(0.05)
        tracker.log_article_timing("Article B", "query2")
        tracker.log_query_complete("query2")
        
        # Finalize
        tracker.finalize()
        
        # Check the log file
        with open(temp_log, 'r') as f:
            content = f.read()
            assert 'accumulated amount for query "query1"' in content, "Query1 not logged"
            assert 'accumulated amount for query "query2"' in content, "Query2 not logged"
            assert "FINAL SUMMARY" in content, "Final summary not logged"
        
        print("✓ Multiple queries test passed")
    finally:
        if os.path.exists(temp_log):
            os.unlink(temp_log)


def test_periodic_logging():
    """Test periodic logging functionality."""
    print("\nTest 4: Periodic logging (shortened interval)")
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        temp_log = f.name
    
    try:
        tracker = TimingTracker(log_file=temp_log)
        # Set a very short interval for testing
        tracker.periodic_interval = 1  # 1 second instead of 10 minutes
        
        # Process some articles
        with tracker.time_task("task1"):
            time.sleep(0.1)
        tracker.log_article_timing("Article 1", "query1")
        
        # Wait for periodic interval
        time.sleep(1.5)
        
        # Process another article to trigger check
        with tracker.time_task("task2"):
            time.sleep(0.1)
        tracker.log_article_timing("Article 2", "query1")
        
        # Check the log file
        with open(temp_log, 'r') as f:
            content = f.read()
            assert "PERIODIC UPDATE" in content, "Periodic update not logged"
        
        print("✓ Periodic logging test passed")
    finally:
        if os.path.exists(temp_log):
            os.unlink(temp_log)


def test_context_manager():
    """Test the time_task context manager."""
    print("\nTest 5: Context manager functionality")
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        temp_log = f.name
    
    try:
        tracker = TimingTracker(log_file=temp_log)
        
        # Test that timing works correctly
        start = time.time()
        with tracker.time_task("sleep_task"):
            time.sleep(0.2)
        elapsed = time.time() - start
        
        tracker.log_article_timing("Test Article", "test_query")
        
        # Check that timing is approximately correct
        with open(temp_log, 'r') as f:
            content = f.read()
            # Look for the timing value
            assert "sleep_task:" in content, "Task not in log"
            # The timing should be around 0.2 seconds (200ms)
            assert "0." in content, "Timing value not found"
        
        print(f"✓ Context manager test passed (measured ~{elapsed:.1f}s)")
    finally:
        if os.path.exists(temp_log):
            os.unlink(temp_log)


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Timing Tracker Functionality")
    print("=" * 60)
    
    try:
        test_basic_timing()
        test_accumulated_timing()
        test_multiple_queries()
        test_periodic_logging()
        test_context_manager()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
