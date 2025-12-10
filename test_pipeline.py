#!/usr/bin/env python3
"""
Test procedure that runs user-specified pipelines on articles from the cache.

This script allows testing different processing sequences (pipelines) on cached articles
to validate AI processing steps without fetching new articles.

Usage examples:
    # Process all cache files with a single pipeline
    python test_pipeline.py -p "categorize, spin_genders, clean_summary"
    
    # Process specific queries with multiple pipelines
    python test_pipeline.py --queries "tech news" "politics" \
        -p "categorize, spin_genders, clean_summary" \
        -p "image_classification, spin_genders, categorize, short_summary"
    
    # Filter by date and format output
    python test_pipeline.py --newer-than "2024-12-01" --format-text
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, TYPE_CHECKING

# Setup logging early
logger = logging.getLogger(__name__)

# Type checking imports
if TYPE_CHECKING:
    from ai_utils import AIUtils

# Lazy imports - will be imported when needed to avoid dependency issues
# This allows --help to work without all dependencies installed
def lazy_imports():
    """Import heavy dependencies only when actually running the script."""
    global AIUtils, classify_leadership, short_summary, clean_summary
    global spin_genders, clean_article, add_background_on_women
    global find_query_files, load_json, extract_articles_from_cache, process_article_in_text
    global load_config, setup_logging, process_article_images, analyze_image_gender
    
    from ai_utils import AIUtils
    from ai_queries import (
        classify_leadership,
        short_summary,
        clean_summary,
        spin_genders,
        clean_article,
        add_background_on_women,
    )
    from news_formatter import (
        find_query_files,
        load_json,
        extract_articles_from_cache,
        process_article_in_text,
    )
    from news_filter import (
        load_config,
        setup_logging as nf_setup_logging,
        process_article_images,
    )
    from analyze_image_gender import analyze_image_gender
    
    # Re-export setup_logging
    globals()['setup_logging'] = nf_setup_logging


def parse_pipeline(pipeline_str: str) -> List[str]:
    """
    Parse a pipeline string into a list of processing steps.
    
    Args:
        pipeline_str: Comma-separated string of processing steps
        
    Returns:
        List of step names
    """
    steps = [step.strip() for step in pipeline_str.split(',')]
    return [step for step in steps if step]  # Remove empty strings


def validate_pipeline_steps(steps: List[str]) -> bool:
    """
    Validate that all pipeline steps are recognized.
    
    Args:
        steps: List of pipeline step names
        
    Returns:
        True if all steps are valid, False otherwise
    """
    valid_steps = {
        'categorize',
        'spin_genders',
        'clean_summary',
        'short_summary',
        'image_classification',
        'clean_article',
        'add_background',
    }
    
    invalid_steps = [step for step in steps if step not in valid_steps]
    
    if invalid_steps:
        logger.error(f"Invalid pipeline steps: {', '.join(invalid_steps)}")
        logger.error(f"Valid steps are: {', '.join(sorted(valid_steps))}")
        return False
    
    return True


def apply_pipeline_step(step: str, article: Dict[str, Any], cfg: dict, ai_util, 
                       format_text: bool = False) -> str:
    """
    Apply a single pipeline step to an article.
    
    Args:
        step: The name of the processing step
        article: The article dictionary
        cfg: Configuration dictionary
        ai_util: AI utilities instance
        format_text: Whether to format output as readable text
        
    Returns:
        The result of the processing step
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Applying step: {step}")
    logger.info(f"Article: {article.get('title', 'Unknown')}")
    logger.info(f"{'='*80}\n")
    
    # Get the current content to process
    content = article.get('content') or article.get('body') or article.get('description', '')
    
    if not content:
        logger.warning(f"No content available for article: {article.get('title', 'Unknown')}")
        return ""
    
    result = ""
    
    try:
        if step == 'categorize':
            # Classify if article is about female leadership
            logger.info("INPUT (First 500 chars):")
            logger.info(content[:500] + "..." if len(content) > 500 else content)
            
            is_leader, leader_name = classify_leadership(content, cfg, ai_util)
            
            result = f"Is Female Leader: {is_leader}"
            if leader_name:
                result += f"\nLeader Name: {leader_name}"
                article['leader_name'] = leader_name
                article['status'] = 'female_leader'
            
            logger.info("\nAI RESPONSE:")
            logger.info(result)
            
        elif step == 'spin_genders':
            logger.info("INPUT (First 500 chars):")
            logger.info(content[:500] + "..." if len(content) > 500 else content)
            
            result = spin_genders(content, cfg, ai_util)
            article['content'] = result
            
            logger.info("\nAI RESPONSE:")
            logger.info(result)
            
        elif step == 'clean_summary':
            logger.info("INPUT (First 500 chars):")
            logger.info(content[:500] + "..." if len(content) > 500 else content)
            
            leader_name = article.get('leader_name')
            result = clean_summary(content, cfg, ai_util, leader_name=leader_name)
            article['content'] = result
            
            logger.info("\nAI RESPONSE:")
            logger.info(result)
            
        elif step == 'short_summary':
            logger.info("INPUT (First 500 chars):")
            logger.info(content[:500] + "..." if len(content) > 500 else content)
            
            result = short_summary(content, cfg, ai_util)
            article['content'] = result
            
            logger.info("\nAI RESPONSE:")
            logger.info(result)
            
        elif step == 'image_classification':
            image_urls = article.get('image_urls', [])
            if not image_urls and article.get('urlToImage'):
                image_urls = [article['urlToImage']]
            
            logger.info(f"INPUT: Processing {len(image_urls)} image(s)")
            for url in image_urls:
                logger.info(f"  - {url}")
            
            if image_urls:
                process_article_images(article, image_urls)
                
                result = f"Most Relevant Image: {article.get('most_relevant_image', 'None')}\n"
                result += f"Most Relevant Status: {article.get('most_relevant_status', 'unknown')}\n"
                result += f"\nAll Image Analysis:\n"
                for img_url, analysis in article.get('image_analysis', {}).items():
                    result += f"  {img_url}\n"
                    result += f"    Status: {analysis.get('status', 'unknown')}\n"
                    result += f"    Prominence: {analysis.get('prominence_score', 0)}\n"
            else:
                result = "No images to process"
            
            logger.info("\nAI RESPONSE:")
            logger.info(result)
            
        elif step == 'clean_article':
            logger.info("INPUT (First 500 chars):")
            logger.info(content[:500] + "..." if len(content) > 500 else content)
            
            result = clean_article(content, cfg, ai_util)
            article['content'] = result
            
            logger.info("\nAI RESPONSE:")
            logger.info(result)
            
        elif step == 'add_background':
            logger.info("INPUT (First 500 chars):")
            logger.info(content[:500] + "..." if len(content) > 500 else content)
            
            result = add_background_on_women(content, cfg, ai_util)
            article['content'] = result
            
            logger.info("\nAI RESPONSE:")
            logger.info(result)
            
        else:
            logger.error(f"Unknown step: {step}")
            result = ""
        
        # Format article text if requested
        if format_text and result:
            logger.info("\n" + "="*80)
            logger.info("FORMATTED ARTICLE AFTER THIS STEP:")
            logger.info("="*80 + "\n")
            lines = []
            process_article_in_text(article, for_email=False, article_number=-1, lines=lines)
            formatted = "\n".join(lines)
            logger.info(formatted)
            print(formatted)
            
    except Exception as e:
        logger.error(f"Error applying step '{step}': {e}", exc_info=True)
        result = f"Error: {str(e)}"
    
    return result


def process_article_with_pipeline(article: Dict[str, Any], pipeline: List[str], 
                                  cfg: dict, ai_util, format_text: bool = False):
    """
    Process an article through a complete pipeline.
    
    Args:
        article: The article dictionary
        pipeline: List of processing steps
        cfg: Configuration dictionary
        ai_util: AI utilities instance
        format_text: Whether to format output after each step
    """
    logger.info(f"\n{'#'*80}")
    logger.info(f"# Processing Article: {article.get('title', 'Unknown')}")
    logger.info(f"# Pipeline: {' -> '.join(pipeline)}")
    logger.info(f"{'#'*80}\n")
    
    for step in pipeline:
        apply_pipeline_step(step, article, cfg, ai_util, format_text)


def filter_articles_by_date(articles: List[Dict[str, Any]], newer_than: str) -> List[Dict[str, Any]]:
    """
    Filter articles to only include those newer than the specified date.
    
    Args:
        articles: List of article dictionaries with cache metadata
        newer_than: ISO format date string (YYYY-MM-DD)
        
    Returns:
        Filtered list of articles
    """
    try:
        cutoff_date = datetime.fromisoformat(newer_than)
    except ValueError:
        logger.error(f"Invalid date format: {newer_than}. Expected YYYY-MM-DD")
        return articles
    
    filtered = []
    for article in articles:
        first_seen = article.get('first_seen', '')
        if first_seen:
            try:
                article_date = datetime.fromisoformat(first_seen)
                if article_date >= cutoff_date:
                    filtered.append(article)
            except ValueError:
                logger.warning(f"Invalid first_seen date for article: {article.get('title', 'Unknown')}")
    
    logger.info(f"Filtered {len(articles)} articles to {len(filtered)} articles newer than {newer_than}")
    return filtered


def main():
    """Main entry point for the test pipeline script."""
    parser = argparse.ArgumentParser(
        description='Test procedure to run pipelines on cached articles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  categorize          - Classify if article is about female leadership
  spin_genders        - Rewrite article with gender perspectives
  clean_summary       - Generate detailed summary (for female leaders)
  short_summary       - Generate brief 2-sentence summary
  image_classification - Analyze images for gender representation
  clean_article       - Clean and format article text
  add_background      - Add background context about women

Examples:
  # Single pipeline on all cache files
  python test_pipeline.py -p "categorize, spin_genders, clean_summary"
  
  # Multiple pipelines on specific queries
  python test_pipeline.py --queries "tech news" "politics" \\
      -p "categorize, spin_genders, clean_summary" \\
      -p "image_classification, categorize, short_summary"
  
  # Filter by date with formatted output
  python test_pipeline.py --newer-than "2024-12-01" --format-text \\
      -p "categorize, clean_summary"
        """
    )
    
    parser.add_argument(
        '--queries',
        nargs='+',
        help='List of query names to process (e.g., "tech news" "politics"). '
             'If not specified, all cache files will be processed.'
    )
    
    parser.add_argument(
        '-p', '--pipeline',
        action='append',
        dest='pipelines',
        required=True,
        help='Pipeline of processing steps (comma-separated). '
             'Can be specified multiple times for multiple pipelines. '
             'Example: -p "categorize, spin_genders, clean_summary"'
    )
    
    parser.add_argument(
        '--newer-than',
        help='Only process articles with first_seen newer than this date (YYYY-MM-DD format)'
    )
    
    parser.add_argument(
        '--format-text',
        action='store_true',
        help='Format and display article as text after each pipeline step'
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--log',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--cache-dir',
        default='cache',
        help='Cache directory path (default: cache)'
    )
    
    args = parser.parse_args()
    
    # Import dependencies now that we know we're actually running (not just --help)
    lazy_imports()
    
    # Setup logging
    setup_logging(args.log)
    logger.info("Starting test pipeline procedure")
    
    # Load configuration
    try:
        cfg = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Parse and validate pipelines
    pipelines = []
    for pipeline_str in args.pipelines:
        steps = parse_pipeline(pipeline_str)
        if not validate_pipeline_steps(steps):
            sys.exit(1)
        pipelines.append(steps)
        logger.info(f"Pipeline {len(pipelines)}: {' -> '.join(steps)}")
    
    # Initialize AI utilities
    try:
        ai_util = AIUtils(cfg)
        logger.info("Initialized AI utilities")
    except Exception as e:
        logger.error(f"Failed to initialize AI utilities: {e}")
        sys.exit(1)
    
    # Find cache files
    cache_files = find_query_files(args.queries, cache_dir=args.cache_dir)
    
    if not cache_files:
        logger.error(f"No cache files found in {args.cache_dir}")
        if args.queries:
            logger.error(f"Searched for queries: {', '.join(args.queries)}")
        sys.exit(1)
    
    logger.info(f"Found {len(cache_files)} cache file(s): {', '.join(cache_files.keys())}")
    
    # Process each cache file
    total_articles = 0
    for query_name, cache_file in cache_files.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing query: {query_name}")
        logger.info(f"Cache file: {cache_file}")
        logger.info(f"{'='*80}\n")
        
        # Load cache data
        cache_data = load_json(cache_file)
        if not cache_data:
            logger.warning(f"Failed to load cache file: {cache_file}")
            continue
        
        # Extract articles - need to handle both old and new cache formats
        articles_data = cache_data.get('articles', {})
        
        articles = []
        if isinstance(articles_data, dict):
            # New format: articles is a dict with article_key -> {first_seen, article_data}
            for article_key, entry in articles_data.items():
                if isinstance(entry, dict) and 'article_data' in entry:
                    article = entry['article_data']
                    # Add first_seen to article for filtering
                    article['first_seen'] = entry.get('first_seen', '')
                    articles.append(article)
        else:
            # Old format: articles is a list
            articles = articles_data if isinstance(articles_data, list) else []
        
        logger.info(f"Loaded {len(articles)} article(s) from cache")
        
        # Filter by date if requested
        if args.newer_than:
            articles = filter_articles_by_date(articles, args.newer_than)
        
        if not articles:
            logger.warning(f"No articles to process for query: {query_name}")
            continue
        
        # Process each article with each pipeline
        for article_idx, article in enumerate(articles, 1):
            logger.info(f"\n{'#'*80}")
            logger.info(f"# Article {article_idx}/{len(articles)} from query '{query_name}'")
            logger.info(f"# Title: {article.get('title', 'Unknown')}")
            logger.info(f"# URL: {article.get('url', 'N/A')}")
            if article.get('first_seen'):
                logger.info(f"# First Seen: {article['first_seen']}")
            logger.info(f"{'#'*80}\n")
            
            for pipeline_idx, pipeline in enumerate(pipelines, 1):
                logger.info(f"\n{'*'*80}")
                logger.info(f"* Pipeline {pipeline_idx}/{len(pipelines)}")
                logger.info(f"{'*'*80}\n")
                
                # Create a copy of the article for this pipeline to avoid interference
                import copy
                article_copy = copy.deepcopy(article)
                
                process_article_with_pipeline(
                    article_copy,
                    pipeline,
                    cfg,
                    ai_util,
                    args.format_text
                )
            
            total_articles += 1
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Test pipeline completed")
    logger.info(f"Processed {total_articles} article(s) with {len(pipelines)} pipeline(s)")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
