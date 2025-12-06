import os
import time
import json
import logging
import argparse
from pathlib import Path
from ai_utils import AIUtils
from ai_queries import classify_leadership, short_summary, clean_summary, spin_genders
from article_fetcher import fetch_full_text_and_images

import openai as openai_pkg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCacheClassifications:
    def __init__(self, cache_folder: str, cfg: dict, ai_util=None, use_openai=True, openai_limit=None, no_fetch=False, num_articles=20):
        """
        Initialize the test system for running classifications on cached data.

        Args:
            cache_folder (str): Path to the cache folder.
            cfg (dict): Configuration dictionary containing OpenAI and Local AI configurations.
            ai_util (AIUtils, optional): AI utility instance for local AI testing.
            use_openai (bool): Whether to run tests using OpenAI.
            openai_limit (int): Maximum number of OpenAI queries allowed. Defaults to None (no limit).
            no_fetch (bool): If True, don't fetch missing article bodies.
            num_articles (int): Number of articles to test (default: 20).
        """
        self.cache_folder = Path(cache_folder)
        self.cfg = cfg
        self.ai_util = ai_util
        self.use_openai = use_openai
        self.openai_limit = openai_limit if openai_limit is not None else float("inf")
        self.openai_count = 0
        self.no_fetch = no_fetch
        self.num_articles = num_articles

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

    def run_tests(self):
        """
        Run classification and summarization tests on cached articles.
        """

        openai_client = None

        if self.use_openai:
            openai_pkg.api_key = cfg["openai"]["api_key"]
            openai_client = openai_pkg

        logger.info("Loading cached articles...")
        articles = self.load_cache()
        results = []

        if not articles:
            logger.error("No articles found in the cache folder. Exiting...")
            return
        
        # Check if we have enough articles
        if len(articles) < self.num_articles:
            logger.warning(f"Only {len(articles)} articles found in cache, but {self.num_articles} requested.")
            logger.warning(f"Please run with --fetch to fetch more articles, or reduce --num-articles.")
            if not self.no_fetch:
                logger.info("Attempting to fetch more articles from news sources...")
                # This would require implementing fetch logic similar to test_llm_prompts.py
                # For now, we'll proceed with what we have
            
        logger.info(f"Found {len(articles)} articles. Will test up to {self.num_articles}...")

        for article in articles[:self.num_articles]:
            # Validate article structure
            if not isinstance(article, dict):
                logger.warning(f"Skipping non-dict entry: {article}")
                continue

            # Skip invalid articles
            if "body" not in article or not article["body"]:
                if not self.no_fetch:
                    logger.info(f"Fetching full text for article: {article.get('title', 'Untitled')}...")
                    try:
                        # Use fetch_full_text_and_images to retrieve the full text
                        fetched_body, images = fetch_full_text_and_images(article.get("url"))
                        article["body"] = fetched_body  # Update 'body' field with fetched text
                    except Exception as e:
                        logger.error(f"Failed to fetch article body for: {article.get('url')}. Error: {e}")
                        continue
                else:
                    logger.warning(f"Skipping article without 'body' due to --no-fetch: {article.get('url')}")
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
                "classification": None,
                "short_summary": None,
                "clean_summary": None,
                "spin_genders": None,
                "performance": {},
            }

            # Test classify_leadership
            if self.use_openai or self.ai_util:
                start_time = time.time()
                try:
                    is_leader, leader_name = classify_leadership(
                        body, self.cfg, self.ai_util
                    )
                    result["classification"] = {"is_leader": is_leader, "leader_name": leader_name}
                    self.openai_count += 1
                except Exception as e:
                    logger.error(f"Error running classify_leadership: {e}")
                result["performance"]["classification_time"] = time.time() - start_time

            # Test short_summary
            if self.use_openai or self.ai_util:
                start_time = time.time()
                try:
                    result["short_summary"] = short_summary(
                        body, self.cfg, self.ai_util
                    )
                    self.openai_count += 1
                except Exception as e:
                    logger.error(f"Error running short_summary: {e}")
                result["performance"]["short_summary_time"] = time.time() - start_time

            # Test clean_summary
            if self.use_openai or self.ai_util:
                start_time = time.time()
                try:
                    leader_name = result["classification"].get("leader_name") if result["classification"] else None
                    result["clean_summary"] = clean_summary(
                        body, self.cfg, self.ai_util, leader_name=leader_name
                    )
                    self.openai_count += 1
                except Exception as e:
                    logger.error(f"Error running clean_summary: {e}")
                result["performance"]["clean_summary_time"] = time.time() - start_time

            # Test spin_genders
            if self.use_openai or self.ai_util:
                start_time = time.time()
                try:
                    result["spin_genders"] = spin_genders(body, self.cfg, self.ai_util)
                    self.openai_count += 1
                except Exception as e:
                    logger.error(f"Error running spin_genders: {e}")
                result["performance"]["spin_genders_time"] = time.time() - start_time

            # Log the result
            logger.info(f"Processed article: {result['title']}")
            results.append(result)

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
    parser.add_argument("--no-fetch", action="store_true", help="Disable fetching full text for missing 'body' fields.")
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
        no_fetch=args.no_fetch,
        num_articles=args.num_articles,
    )

    # Run the tests
    test_system.run_tests()
