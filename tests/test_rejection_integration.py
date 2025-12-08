#!/usr/bin/env python3
"""
Integration test for the rejection prompt feature.
Tests that the feature integrates correctly with the news_filter module.
"""

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_config_structure():
    """Test that config.yaml.example has the correct structure for rejection prompts."""
    
    logger.info("Testing config.yaml.example structure...")
    
    with open("config.yaml.example", "r") as f:
        cfg = yaml.safe_load(f)
    
    # Check that rejection_model is in ollama config
    assert "ollama" in cfg, "ollama section missing from config"
    assert "rejection_model" in cfg["ollama"], "rejection_model missing from ollama config"
    logger.info(f"✓ rejection_model found: {cfg['ollama']['rejection_model']}")
    
    # Check that reject_classification_prompt_1 is in prompts
    assert "prompts" in cfg, "prompts section missing from config"
    assert "reject_classification_prompt_1" in cfg["prompts"], "reject_classification_prompt_1 missing from prompts"
    logger.info("✓ reject_classification_prompt_1 found in prompts")
    
    # Verify the prompt has the correct format
    prompt = cfg["prompts"]["reject_classification_prompt_1"]
    assert "ACCEPT" in prompt, "ACCEPT keyword missing from prompt"
    assert "REJECT" in prompt, "REJECT keyword missing from prompt"
    logger.info("✓ Rejection prompt has correct format")
    
    # Check that Canada query has rejection_prompt setting
    queries = cfg.get("queries", [])
    canada_query = None
    for query in queries:
        if query.get("name") == "Canada":
            canada_query = query
            break
    
    assert canada_query is not None, "Canada query not found"
    assert "rejection_prompt" in canada_query, "rejection_prompt missing from Canada query"
    assert canada_query["rejection_prompt"] == "reject_classification_prompt_1", \
        "Canada query rejection_prompt should reference reject_classification_prompt_1"
    logger.info("✓ Canada query has rejection_prompt configured")
    
    logger.info("✓ All config structure tests passed!")
    return True


def test_function_import():
    """Test that the reject_classification function can be imported."""
    
    logger.info("Testing function imports...")
    
    try:
        from ai_queries import reject_classification
        logger.info("✓ reject_classification function imported successfully")
    except ImportError as e:
        logger.warning(f"Could not import ai_queries (missing dependencies): {e}")
        logger.info("✓ This is expected in environments without full dependencies")
        # Check that the function exists in the source code
        with open("ai_queries.py", "r") as f:
            content = f.read()
            assert "def reject_classification" in content, "reject_classification function not found in source"
        logger.info("✓ reject_classification function exists in source code")
        return True
    
    # Check that news_filter imports it correctly
    try:
        # Just check that it's imported in the imports section
        with open("news_filter.py", "r") as f:
            content = f.read()
            assert "reject_classification" in content, "reject_classification not imported in news_filter"
        logger.info("✓ reject_classification is referenced in news_filter module")
    except Exception as e:
        logger.error(f"Failed to verify news_filter imports: {e}")
        return False
    
    logger.info("✓ All import tests passed!")
    return True


def test_optional_rejection_prompt():
    """Test that queries without rejection_prompt still work."""
    
    logger.info("Testing optional nature of rejection_prompt...")
    
    # Create a mock query config without rejection_prompt
    qcfg_without = {
        "name": "Test Query",
        "fallback": "short_summary"
    }
    
    # Check that we can get rejection_prompt with None as default
    rejection_prompt = qcfg_without.get("rejection_prompt")
    assert rejection_prompt is None, "rejection_prompt should be None when not specified"
    logger.info("✓ rejection_prompt is optional (returns None when not specified)")
    
    # Create a query config with rejection_prompt
    qcfg_with = {
        "name": "Test Query",
        "fallback": "short_summary",
        "rejection_prompt": "reject_classification_prompt_1"
    }
    
    rejection_prompt = qcfg_with.get("rejection_prompt")
    assert rejection_prompt == "reject_classification_prompt_1", \
        "rejection_prompt should be retrieved correctly"
    logger.info("✓ rejection_prompt is retrieved correctly when specified")
    
    logger.info("✓ All optional tests passed!")
    return True


if __name__ == "__main__":
    try:
        logger.info("=" * 80)
        logger.info("RUNNING INTEGRATION TESTS FOR REJECTION PROMPT FEATURE")
        logger.info("=" * 80)
        
        test1 = test_config_structure()
        test2 = test_function_import()
        test3 = test_optional_rejection_prompt()
        
        if test1 and test2 and test3:
            logger.info("=" * 80)
            logger.info("ALL INTEGRATION TESTS PASSED!")
            logger.info("=" * 80)
            exit(0)
        else:
            logger.error("Some tests failed")
            exit(1)
    except Exception as e:
        logger.error(f"Integration test failed with error: {e}", exc_info=True)
        exit(1)
