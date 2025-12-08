"""
Timing tracker for news_filter.py performance monitoring.

This module provides utilities to track and log timing information for various tasks
like image analysis, AI queries, etc. It writes timing data to timing.log with:
- Per-article timings
- Accumulated timings per query
- Accumulated timings for all articles
- Periodic updates (every 10 minutes)
"""

import time
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Optional
import threading

logger = logging.getLogger(__name__)


class TimingTracker:
    """
    Tracks timing information for various tasks across articles and queries.
    Thread-safe for concurrent operations.
    """

    def __init__(self, log_file: str = "timing.log"):
        self.log_file = log_file
        self.lock = threading.Lock()
        
        # Per-article timing (temporary, cleared after logging)
        self.current_article_timings: Dict[str, float] = {}
        
        # Accumulated timings per query
        self.query_timings: Dict[str, Dict[str, float]] = {}
        
        # Accumulated timings for all articles
        self.total_timings: Dict[str, float] = {}
        
        # Count of operations per query
        self.query_counts: Dict[str, Dict[str, int]] = {}
        
        # Count of operations for all articles
        self.total_counts: Dict[str, int] = {}
        
        # Track last periodic log time
        self.last_periodic_log = time.time()
        self.periodic_interval = 600  # 10 minutes in seconds
        
        # Initialize the timing log file
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize the timing log file with a header."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Timing Log Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")
        except Exception as e:
            logger.error(f"Failed to initialize timing log file: {e}")
    
    @contextmanager
    def time_task(self, task_name: str, model_name: Optional[str] = None):
        """
        Context manager to time a task.
        
        Args:
            task_name: Name of the task being timed (e.g., 'image_analysis', 'spin_genders')
            model_name: Optional model name (e.g., 'gpt-4', 'qwen-2')
        
        Usage:
            with tracker.time_task('spin_genders', 'gpt-4'):
                # do work
                pass
        """
        # Build full task key including model if provided
        task_key = f"{task_name} (model {model_name})" if model_name else task_name
        
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            with self.lock:
                # Add to current article timings and increment count
                self.current_article_timings[task_key] = \
                    self.current_article_timings.get(task_key, 0) + elapsed
    
    def log_article_timing(self, article_title: str, query_name: str):
        """
        Log timing for the current article and accumulate to query and total timings.
        
        Args:
            article_title: Title of the article
            query_name: Name of the query this article belongs to
        """
        with self.lock:
            if not self.current_article_timings:
                return  # Nothing to log
            
            try:
                # Write per-article timing
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\narticle: {article_title}\n")
                    for task, duration in sorted(self.current_article_timings.items()):
                        f.write(f"  {task}: {duration:.1f}s\n")
                
                # Accumulate to query timings and counts
                if query_name not in self.query_timings:
                    self.query_timings[query_name] = {}
                if query_name not in self.query_counts:
                    self.query_counts[query_name] = {}
                
                for task, duration in self.current_article_timings.items():
                    # Accumulate timing
                    self.query_timings[query_name][task] = \
                        self.query_timings[query_name].get(task, 0) + duration
                    self.total_timings[task] = self.total_timings.get(task, 0) + duration
                    
                    # Increment counts
                    self.query_counts[query_name][task] = \
                        self.query_counts[query_name].get(task, 0) + 1
                    self.total_counts[task] = self.total_counts.get(task, 0) + 1
                
                # Clear current article timings
                self.current_article_timings.clear()
                
                # Check if we should do periodic logging
                self._check_periodic_log()
                
            except Exception as e:
                logger.error(f"Failed to log article timing: {e}")
    
    def log_query_complete(self, query_name: str):
        """
        Log accumulated timings for a completed query.
        
        Args:
            query_name: Name of the query that just completed
        """
        with self.lock:
            if query_name not in self.query_timings:
                return  # No timings for this query
            
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'-'*80}\n")
                    f.write(f"accumulated amount for query \"{query_name}\":\n")
                    for task in sorted(self.query_timings[query_name].keys()):
                        duration = self.query_timings[query_name][task]
                        count = self.query_counts[query_name].get(task, 0)
                        f.write(self._format_timing_line(task, duration, count))
                    
                    # Also log total timings so far
                    f.write(f"\naccumulated amount for all articles so far:\n")
                    for task in sorted(self.total_timings.keys()):
                        duration = self.total_timings[task]
                        count = self.total_counts.get(task, 0)
                        f.write(self._format_timing_line(task, duration, count))
                    f.write(f"{'-'*80}\n\n")
            except Exception as e:
                logger.error(f"Failed to log query completion: {e}")
    
    def _format_timing_line(self, task: str, duration: float, count: int) -> str:
        """
        Format a timing line with duration, count, and average.
        
        Args:
            task: Task name
            duration: Total duration in seconds
            count: Number of times the task was performed
            
        Returns:
            Formatted string with timing information
        """
        if count > 0:
            avg_ms = (duration / count) * 1000
            return f"  {task}: {duration:.1f}s (count: {count}, avg: {avg_ms:.0f}ms)\n"
        else:
            # Should not happen but handle gracefully
            return f"  {task}: {duration:.1f}s (count: 0, avg: N/A)\n"
    
    def _check_periodic_log(self):
        """Check if we should write a periodic accumulated timing log."""
        current_time = time.time()
        if current_time - self.last_periodic_log >= self.periodic_interval:
            self._write_accumulated_log()
            self.last_periodic_log = current_time
    
    def _write_accumulated_log(self):
        """Write accumulated timings for all articles (periodic update)."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'~'*80}\n")
                f.write(f"PERIODIC UPDATE ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
                f.write(f"accumulated amount for all articles so far:\n")
                for task in sorted(self.total_timings.keys()):
                    duration = self.total_timings[task]
                    count = self.total_counts.get(task, 0)
                    f.write(self._format_timing_line(task, duration, count))
                f.write(f"{'~'*80}\n\n")
        except Exception as e:
            logger.error(f"Failed to write periodic log: {e}")
    
    def finalize(self):
        """Write final accumulated timings and close the log."""
        with self.lock:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"FINAL SUMMARY ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
                    f.write(f"accumulated amount for all articles:\n")
                    for task in sorted(self.total_timings.keys()):
                        duration = self.total_timings[task]
                        count = self.total_counts.get(task, 0)
                        f.write(self._format_timing_line(task, duration, count))
                    f.write(f"{'='*80}\n\n")
            except Exception as e:
                logger.error(f"Failed to write final summary: {e}")


# Global timing tracker instance
_timing_tracker: Optional[TimingTracker] = None


def get_timing_tracker() -> TimingTracker:
    """Get or create the global timing tracker instance."""
    global _timing_tracker
    if _timing_tracker is None:
        _timing_tracker = TimingTracker()
    return _timing_tracker


def reset_timing_tracker():
    """Reset the global timing tracker (useful for testing)."""
    global _timing_tracker
    _timing_tracker = None
