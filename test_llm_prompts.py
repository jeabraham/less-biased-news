#!/usr/bin/env python3
"""
Test LLM prompts (classification, short_summary, spin_genders, clean_summary) 
against saved news articles to help fine-tune prompts for the chosen LLM.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict
import yaml

from ai_utils import AIUtils
from ai_queries import classify_leadership, short_summary, clean_summary, spin_genders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    logger.debug(f"Loading config from {path}")
    if not os.path.exists(path):
        logger.error(f"Config file not found: {path}")
        raise FileNotFoundError(f"{path} does not exist")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Configuration loaded successfully")
    return cfg


def fetch_full_text_and_images(url: str):
    """
    Fetch full text and images from a URL using article_fetcher.
    This is a lazy import wrapper to avoid heavy dependencies at module load.
    """
    try:
        from article_fetcher import fetch_full_text_and_images as _fetch
        return _fetch(url)
    except ImportError as e:
        logger.error(f"Could not import article_fetcher: {e}")
        raise


def fetch_articles(query_cfg: dict, cfg: dict):
    """
    Fetch articles using news_filter's fetch function.
    This is a lazy import wrapper to avoid heavy dependencies at module load.
    """
    try:
        from news_filter import fetch_articles as _fetch
        return _fetch(query_cfg, cfg)
    except ImportError as e:
        logger.error(f"Could not import news_filter: {e}")
        raise


class LLMPromptTester:
    """Test LLM prompts against cached news articles."""
    
    def __init__(self, cfg: dict, num_articles: int = 20, cache_folder: str = "cache"):
        """
        Initialize the test system.
        
        Args:
            cfg: Configuration dictionary
            num_articles: Number of articles to test (default: 20)
            cache_folder: Path to the cache folder for saved articles
        """
        self.cfg = cfg
        self.num_articles = num_articles
        self.cache_folder = Path(cache_folder)
        self.cache_folder.mkdir(exist_ok=True)
        
        # Initialize AI utilities
        self.ai_util = AIUtils(cfg)
        
        logger.info(f"Initialized LLM Prompt Tester")
        logger.info(f"Ollama enabled: {self.ai_util.ollama_enabled}")
        logger.info(f"OpenAI enabled: {self.ai_util.openai_client is not None}")
        logger.info(f"Local AI enabled: {self.ai_util.local_capable}")
    
    def load_cached_articles(self) -> List[Dict]:
        """
        Load cached articles from the cache folder.
        
        Returns:
            List of article dictionaries
        """
        articles = []
        
        # Load all JSON files from cache
        for file_path in self.cache_folder.glob("*.json"):
            logger.info(f"Loading articles from {file_path}")
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, dict):
                    # Format with "articles" key
                    raw_articles = data.get("articles", [])
                    if isinstance(raw_articles, list):
                        articles.extend(raw_articles)
                elif isinstance(data, list):
                    # List format
                    articles.extend(data)
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(articles)} articles from cache")
        return articles
    
    def fetch_new_articles(self) -> List[Dict]:
        """
        Fetch new articles from configured news sources.
        
        Returns:
            List of fetched article dictionaries
        """
        logger.info("Fetching new articles from news sources...")
        
        all_articles = []
        queries = self.cfg.get("queries", [])
        
        for query_cfg in queries:
            try:
                logger.info(f"Fetching from source: {query_cfg.get('name', 'Unknown')}")
                articles = fetch_articles(query_cfg, self.cfg)
                all_articles.extend(articles)
                
                # Stop if we have enough
                if len(all_articles) >= self.num_articles:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching articles: {e}")
        
        # Save to cache
        if all_articles:
            cache_file = self.cache_folder / f"fetched_{int(time.time())}.json"
            try:
                with open(cache_file, "w") as f:
                    json.dump({
                        "date": time.strftime("%Y-%m-%d"),
                        "articles": all_articles
                    }, f, indent=2)
                logger.info(f"Saved {len(all_articles)} articles to {cache_file}")
            except Exception as e:
                logger.error(f"Error saving articles: {e}")
        
        return all_articles
    
    def ensure_full_text(self, article: Dict) -> bool:
        """
        Ensure article has full body text, fetching if necessary.
        
        Args:
            article: Article dictionary
            
        Returns:
            True if article has body text, False otherwise
        """
        if article.get("body"):
            return True
        
        # Try to fetch full text
        url = article.get("url")
        if not url:
            return False
        
        try:
            logger.info(f"Fetching full text for: {article.get('title', 'Untitled')}")
            body, images = fetch_full_text_and_images(url)
            if body:
                article["body"] = body
                article["images"] = images
                return True
        except Exception as e:
            logger.error(f"Error fetching full text: {e}")
        
        return False
    
    def test_prompts(self, articles: List[Dict]) -> List[Dict]:
        """
        Test all four LLM prompts against the articles.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of test results
        """
        results = []
        
        for i, article in enumerate(articles[:self.num_articles], 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing article {i}/{min(self.num_articles, len(articles))}")
            logger.info(f"Title: {article.get('title', 'Untitled')}")
            logger.info(f"{'='*60}")
            
            # Ensure we have full text
            if not self.ensure_full_text(article):
                logger.warning("Skipping article: no body text available")
                continue
            
            body = article["body"]
            
            # Initialize result
            result = {
                "title": article.get("title", "Untitled"),
                "url": article.get("url", ""),
                "body_length": len(body),
                "tests": {},
                "timings": {}
            }
            
            # Test 1: Classification
            logger.info("\n--- Testing classification prompt ---")
            start = time.time()
            try:
                is_leader, leader_name = classify_leadership(body, self.cfg, self.ai_util)
                result["tests"]["classification"] = {
                    "is_leader": is_leader,
                    "leader_name": leader_name,
                    "success": True
                }
                logger.info(f"Result: is_leader={is_leader}, leader_name={leader_name}")
            except Exception as e:
                logger.error(f"Classification failed: {e}")
                result["tests"]["classification"] = {
                    "success": False,
                    "error": str(e)
                }
            result["timings"]["classification"] = time.time() - start
            
            # Test 2: Short Summary
            logger.info("\n--- Testing short_summary prompt ---")
            start = time.time()
            try:
                summary = short_summary(body, self.cfg, self.ai_util)
                result["tests"]["short_summary"] = {
                    "summary": summary,
                    "length": len(summary),
                    "success": True
                }
                logger.info(f"Summary (length={len(summary)}): {summary[:200]}...")
            except Exception as e:
                logger.error(f"Short summary failed: {e}")
                result["tests"]["short_summary"] = {
                    "success": False,
                    "error": str(e)
                }
            result["timings"]["short_summary"] = time.time() - start
            
            # Test 3: Clean Summary
            logger.info("\n--- Testing clean_summary prompt ---")
            start = time.time()
            try:
                leader_name = result["tests"]["classification"].get("leader_name")
                cleaned = clean_summary(body, self.cfg, self.ai_util, leader_name)
                result["tests"]["clean_summary"] = {
                    "summary": cleaned,
                    "length": len(cleaned),
                    "success": True
                }
                logger.info(f"Clean summary (length={len(cleaned)}): {cleaned[:200]}...")
            except Exception as e:
                logger.error(f"Clean summary failed: {e}")
                result["tests"]["clean_summary"] = {
                    "success": False,
                    "error": str(e)
                }
            result["timings"]["clean_summary"] = time.time() - start
            
            # Test 4: Spin Genders
            logger.info("\n--- Testing spin_genders prompt ---")
            start = time.time()
            try:
                spun = spin_genders(body, self.cfg, self.ai_util)
                result["tests"]["spin_genders"] = {
                    "text": spun,
                    "length": len(spun),
                    "success": True
                }
                logger.info(f"Spun text (length={len(spun)}): {spun[:200]}...")
            except Exception as e:
                logger.error(f"Spin genders failed: {e}")
                result["tests"]["spin_genders"] = {
                    "success": False,
                    "error": str(e)
                }
            result["timings"]["spin_genders"] = time.time() - start
            
            results.append(result)
            
            # Show timing summary
            logger.info("\nTiming summary:")
            for test_name, duration in result["timings"].items():
                logger.info(f"  {test_name}: {duration:.2f}s")
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str = None):
        """
        Save test results to a JSON file.
        
        Args:
            results: List of test result dictionaries
            output_file: Output file path (default: test_results_<timestamp>.json)
        """
        if not output_file:
            output_file = f"test_results_{int(time.time())}.json"
        
        output_path = Path(output_file)
        
        # Prepare summary
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_articles_tested": len(results),
            "ollama_enabled": self.ai_util.ollama_enabled,
            "openai_enabled": self.ai_util.openai_client is not None,
            "local_ai_enabled": self.ai_util.local_capable,
            "results": results
        }
        
        try:
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"\nTest results saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def print_summary(self, results: List[Dict]):
        """Print a summary of test results."""
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        
        total_tests = len(results) * 4  # 4 prompts per article
        successful_tests = 0
        
        for result in results:
            for test_name, test_data in result["tests"].items():
                if test_data.get("success"):
                    successful_tests += 1
        
        logger.info(f"Articles tested: {len(results)}")
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Successful: {successful_tests}")
        logger.info(f"Failed: {total_tests - successful_tests}")
        logger.info(f"Success rate: {successful_tests/total_tests*100:.1f}%")
        
        # Average timings
        timings = {}
        for result in results:
            for test_name, duration in result["timings"].items():
                if test_name not in timings:
                    timings[test_name] = []
                timings[test_name].append(duration)
        
        logger.info("\nAverage timings:")
        for test_name, durations in timings.items():
            avg = sum(durations) / len(durations)
            logger.info(f"  {test_name}: {avg:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Test LLM prompts against news articles"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--num-articles",
        type=int,
        default=20,
        help="Number of articles to test (default: 20)"
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch new articles from news sources"
    )
    parser.add_argument(
        "--cache-folder",
        type=str,
        default="cache",
        help="Path to cache folder for saved articles (default: cache)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for test results (default: test_results_<timestamp>.json)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    try:
        cfg = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Initialize tester
    tester = LLMPromptTester(cfg, args.num_articles, args.cache_folder)
    
    # Get articles
    if args.fetch:
        articles = tester.fetch_new_articles()
        if not articles:
            logger.error("Failed to fetch articles. Check your API keys and configuration.")
            sys.exit(1)
    else:
        articles = tester.load_cached_articles()
        
        if not articles:
            logger.error(f"No cached articles found in '{args.cache_folder}'.")
            logger.error("Please run with --fetch to fetch new articles, or check your cache folder.")
            sys.exit(1)
    
    # Check if we have enough articles
    if len(articles) < args.num_articles:
        logger.warning(f"Only {len(articles)} articles available, but {args.num_articles} requested.")
        if not args.fetch:
            logger.error(f"Please run with --fetch to fetch more articles.")
            sys.exit(1)
    
    # Run tests
    logger.info(f"\nStarting tests on {min(args.num_articles, len(articles))} articles...")
    results = tester.test_prompts(articles)
    
    # Save and display results
    tester.save_results(results, args.output)
    tester.print_summary(results)
    
    logger.info("\nTest complete!")


if __name__ == "__main__":
    main()
