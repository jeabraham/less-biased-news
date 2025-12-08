import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import time
import json
import logging
import argparse
from ai_utils import AIUtils
from ai_queries import classify_leadership, short_summary, clean_summary, spin_genders, add_background_on_women, clean_article
from article_fetcher import fetch_full_text_and_images

import openai as openai_pkg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCacheClassifications:
    def __init__(self, cache_folder: str, cfg: dict, ai_util=None, use_openai=True, openai_limit=None, no_fetch_bodies=False, num_articles=20, fetch_articles=False):
        """
        Initialize the test system for running classifications on cached data.

        Args:
            cache_folder (str): Path to the cache folder.
            cfg (dict): Configuration dictionary containing OpenAI and Local AI configurations.
            ai_util (AIUtils, optional): AI utility instance for local AI testing.
            use_openai (bool): Whether to run tests using OpenAI.
            openai_limit (int): Maximum number of OpenAI queries allowed. Defaults to None (no limit).
            no_fetch_bodies (bool): If True, don't fetch missing article bodies from URLs.
            num_articles (int): Number of articles to test (default: 20).
            fetch_articles (bool): If True, fetch fresh articles from news sources.
        """
        self.cache_folder = Path(cache_folder)
        self.cfg = cfg
        self.ai_util = ai_util
        self.use_openai = use_openai
        self.openai_limit = openai_limit if openai_limit is not None else float("inf")
        self.openai_count = 0
        self.no_fetch_bodies = no_fetch_bodies
        self.num_articles = num_articles
        self.fetch_articles = fetch_articles
        self.output_file = self.cache_folder / "test_results.json"

    def load_cache(self):
        """
            Load all cached articles from the cache folder, supporting both JSON formats:
            - Dict format with "articles" key and optional "date".
            - List format containing article dictionaries directly.

            Returns:
                list: A list of valid articles (dictionaries) loaded from the cache.
            """
        articles = []
        for file in self.cache_folder.glob("*.json"):  # Assuming cached articles are saved as JSON
            logger.info(f"Attempting to load {file}...")
            try:
                # Load the JSON file
                with open(file, "r") as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    # Format with "articles" key and optional "date"
                    raw_articles = data.get("articles", [])
                    if isinstance(raw_articles, list):
                        valid_articles = [
                            article for article in raw_articles
                            if isinstance(article, dict) and "description" in article
                        ]
                        articles.extend(valid_articles)
                        logger.info(f"Found {len(valid_articles)} valid articles in {file}")

                elif isinstance(data, list):
                    # List format where each item is an article
                    valid_articles = [
                        article for article in data
                        if isinstance(article, dict) and "description" in article
                    ]
                    articles.extend(valid_articles)
                    logger.info(f"Found {len(valid_articles)} valid articles in {file}")

                else:
                    logger.warning(f"Unexpected top-level structure in {file}: {type(data)}")

            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON in {file}: Possibly malformed JSON.")
            except Exception as e:
                logger.error(f"Unexpected error while loading {file}: {e}")

        logger.info(f"Total valid articles loaded: {len(articles)}")
        return articles

    def fetch_articles_from_sources(self, num_to_fetch=20):
        """
        Fetch articles from news sources configured in config.yaml.
        
        Args:
            num_to_fetch (int): Number of articles to fetch
            
        Returns:
            list: List of fetched articles
        """
        logger.info(f"Fetching {num_to_fetch} articles from news sources...")
        
        try:
            # Import news fetching functionality
            from news_filter import fetch_articles
            
            all_articles = []
            queries = self.cfg.get("queries", [])
            
            if not queries:
                logger.warning("No queries configured in config.yaml. Cannot fetch articles.")
                return []
            
            for query_cfg in queries:
                try:
                    logger.info(f"Fetching from source: {query_cfg.get('name', 'Unknown')}")
                    # Use the new fetch_articles function
                    articles = fetch_articles(query_cfg, self.cfg, use_cache=False)
                    
                    # Fetch full text for each article
                    for art in articles:
                        if not art.get('body') and art.get('url'):
                            try:
                                body, images = fetch_full_text_and_images(art['url'])
                                art['body'] = body
                                art['images'] = images
                            except Exception as e:
                                logger.debug(f"Failed to fetch body for {art.get('url')}: {e}")
                    
                    all_articles.extend(articles)
                    
                    # Stop if we have enough
                    if len(all_articles) >= num_to_fetch:
                        break
                        
                except Exception as e:
                    logger.error(f"Error fetching from source: {e}")
            
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
            
            return all_articles[:num_to_fetch]
            
        except ImportError as e:
            logger.error(f"Could not import fetch_articles from news_filter: {e}")
            return []
        except Exception as e:
            logger.error(f"Error in fetch_articles_from_sources: {e}")
            return []

    def run_tests(self):
        """
        Run classification and summarization tests on cached articles.
        """

        openai_client = None

        if self.use_openai:
            openai_pkg.api_key = cfg["openai"]["api_key"]
            openai_client = openai_pkg

        # Get list of prompts dynamically from config
        available_prompts = list(self.cfg.get("prompts", {}).keys())
        logger.info(f"Available prompts from config: {available_prompts}")
        
        # Map prompt names to functions
        prompt_functions = {
            "classification": classify_leadership,
            "short_summary": short_summary,
            "clean_summary": clean_summary,
            "clean_article": clean_article,
            "spin_genders": spin_genders,
            "add_background_on_women": add_background_on_women,
        }

        # Fetch articles if requested
        if self.fetch_articles:
            logger.info("Fetching articles from news sources...")
            articles = self.fetch_articles_from_sources(self.num_articles)
            if not articles:
                logger.error("Failed to fetch articles. Trying cache instead...")
                articles = self.load_cache()
        else:
            logger.info("Loading cached articles...")
            articles = self.load_cache()

        results = []

        if not articles:
            logger.error("No articles found. Exiting...")
            return
        
        # Check if we have enough articles
        if len(articles) < self.num_articles:
            logger.warning(f"Only {len(articles)} articles found, but {self.num_articles} requested.")
            if not self.fetch_articles:
                logger.warning(f"Consider running with --fetch to fetch more articles, or reduce --num-articles.")
            
        logger.info(f"Found {len(articles)} articles. Will test up to {self.num_articles}...")

        for idx, article in enumerate(articles[:self.num_articles], 1):
            # Validate article structure
            if not isinstance(article, dict):
                logger.warning(f"Skipping non-dict entry: {article}")
                continue

            # Skip invalid articles
            if "body" not in article or not article["body"]:
                if not self.no_fetch_bodies:
                    logger.info(f"Fetching full text for article: {article.get('title', 'Untitled')}...")
                    try:
                        # Use fetch_full_text_and_images to retrieve the full text
                        fetched_body, images = fetch_full_text_and_images(article.get("url"))
                        article["body"] = fetched_body  # Update 'body' field with fetched text
                    except Exception as e:
                        logger.error(f"Failed to fetch article body for: {article.get('url')}. Error: {e}")
                        continue
                else:
                    logger.warning(f"Skipping article without 'body' due to --no-fetch-bodies: {article.get('url')}")
                    continue

            if self.openai_count >= self.openai_limit:
                logger.warning("Reached the OpenAI query limit. Skipping further OpenAI queries.")
                self.use_openai = False

            # Validate 'body' before passing article to tests
            body = article.get("body")
            if not body:
                logger.warning(f"Skipping article without valid 'body': {article}")
                continue

            result = {
                "title": article.get("title", "Untitled"),
                "url": article.get("url", ""),
                "body": body,  # Include article body in results
                "prompts": {},
                "performance": {},
            }

            logger.info(f"\n{'='*60}")
            logger.info(f"Testing article {idx}/{min(self.num_articles, len(articles))}: {result['title']}")
            logger.info(f"{'='*60}")

            # Dynamically test all prompts that have corresponding functions
            for prompt_name in available_prompts:
                if prompt_name not in prompt_functions:
                    logger.debug(f"No function available for prompt: {prompt_name}")
                    continue
                
                if not (self.use_openai or self.ai_util):
                    continue
                
                func = prompt_functions[prompt_name]
                start_time = time.time()
                
                try:
                    # Handle special cases for different function signatures
                    if prompt_name == "classification":
                        is_leader, leader_name = func(body, self.cfg, self.ai_util)
                        result["prompts"][prompt_name] = {"is_leader": is_leader, "leader_name": leader_name}
                        logger.info(f"  {prompt_name}: is_leader={is_leader}, leader_name={leader_name}")
                    elif prompt_name == "clean_summary":
                        leader_name = None
                        if "classification" in result["prompts"]:
                            leader_name = result["prompts"]["classification"].get("leader_name")
                        output = func(body, self.cfg, self.ai_util, leader_name=leader_name)
                        result["prompts"][prompt_name] = output
                        logger.info(f"  {prompt_name}: {output[:100]}..." if len(output) > 100 else f"  {prompt_name}: {output}")
                    else:
                        output = func(body, self.cfg, self.ai_util)
                        result["prompts"][prompt_name] = output
                        logger.info(f"  {prompt_name}: {output[:100]}..." if len(output) > 100 else f"  {prompt_name}: {output}")
                    
                    self.openai_count += 1
                except Exception as e:
                    logger.error(f"  Error running {prompt_name}: {e}")
                    result["prompts"][prompt_name] = None
                
                result["performance"][f"{prompt_name}_time"] = time.time() - start_time

            # Log the result
            logger.info(f"\nCompleted article: {result['title']}")
            results.append(result)
            
            # Save results after each article
            self.save_results(results)

        # Save results to file
        self.save_results(results)
        logger.info("Test run completed.")

    def save_results(self, results):
        """
        Save the test results to a file.

        Args:
            results (list): List of test results.
        """
        output_file = self.cache_folder / "test_results.json"
        try:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Test results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test AI classifications on cached articles.")
    parser.add_argument("--cache-folder", type=str, default='cache', help="Path to the cache folder.")
    parser.add_argument("--use-openai", action="store_true", help="Enable OpenAI-backed classifications.")
    parser.add_argument("--openai-limit", type=int, default=None, help="Limit the number of OpenAI queries.")
    parser.add_argument("--config", type=str, default='config.yaml', help="Path to the configuration YAML file.")
    parser.add_argument("--fetch", action="store_true", help="Fetch fresh articles from configured news sources.")
    parser.add_argument("--no-fetch-bodies", action="store_true", help="Disable fetching full article text from URLs for articles with missing body fields.")
    parser.add_argument("--num-articles", type=int, default=20, help="Number of articles to test (default: 20).")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",  # Default log level
        help="Set the logging level (default: INFO)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),  # Set logging level
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger("test_cache_classifications")
    logger.info("Starting test_cache_classifications with log level: %s", args.log_level)

    # Load configuration
    import yaml

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Initialize AIUtils for local AI testing
    ai_util = AIUtils(cfg)

    # Initialize the test system
    test_system = TestCacheClassifications(
        cache_folder=args.cache_folder,
        cfg=cfg,
        ai_util=ai_util,
        use_openai=args.use_openai,
        openai_limit=args.openai_limit,
        no_fetch_bodies=args.no_fetch_bodies,
        num_articles=args.num_articles,
        fetch_articles=args.fetch,
    )

    # Run the tests
    test_system.run_tests()
