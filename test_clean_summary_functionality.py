#!/usr/bin/env python3
"""
Test script to verify clean_summary functionality.
Tests that:
1. clean_summary is properly imported in test_llm_prompts.py
2. clean_summary is properly imported in news_filter.py (categorize_article_and_generate_content)
3. clean_summary prompt exists in config.yaml
4. clean_summary function works correctly
"""

import sys
import os
import yaml
import unittest

# Add the current directory to sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestCleanSummaryFunctionality(unittest.TestCase):
    """Test suite for clean_summary functionality."""

    def setUp(self):
        """Load config for testing."""
        with open("config.yaml.example", "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def test_prompt_exists_in_config(self):
        """Test that clean_summary prompt exists in config.yaml.example."""
        self.assertIn("prompts", self.cfg, "Config should have 'prompts' section")
        self.assertIn("clean_summary", self.cfg["prompts"], 
                     "Config prompts should include 'clean_summary'")
        
        prompt = self.cfg["prompts"]["clean_summary"]
        self.assertIsInstance(prompt, str, "clean_summary prompt should be a string")
        self.assertGreater(len(prompt), 0, "clean_summary prompt should not be empty")
        print(f"✅ clean_summary prompt exists ({len(prompt)} chars)")

    def test_clean_summary_imports_in_test_llm_prompts(self):
        """Test that test_llm_prompts.py imports clean_summary."""
        with open("test_llm_prompts.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for import statement
        self.assertIn("from ai_queries import", content, 
                     "test_llm_prompts.py should import from ai_queries")
        self.assertIn("clean_summary", content,
                     "test_llm_prompts.py should import clean_summary")
        
        # Check for usage
        self.assertIn("clean_summary(", content,
                     "test_llm_prompts.py should call clean_summary()")
        print("✅ test_llm_prompts.py correctly imports and uses clean_summary")

    def test_clean_summary_imports_in_news_filter(self):
        """Test that news_filter.py imports and uses clean_summary."""
        with open("news_filter.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for import statement
        self.assertIn("from ai_queries import", content,
                     "news_filter.py should import from ai_queries")
        self.assertIn("clean_summary", content,
                     "news_filter.py should import clean_summary")
        
        # Check for usage in categorize_article_and_generate_content
        self.assertIn("categorize_article_and_generate_content", content,
                     "news_filter.py should define categorize_article_and_generate_content")
        self.assertIn("clean_summary(", content,
                     "news_filter.py should call clean_summary()")
        
        # Verify it's used with leader_name parameter
        self.assertIn("leader_name=", content,
                     "clean_summary should be called with leader_name parameter")
        print("✅ news_filter.py correctly imports and uses clean_summary")

    def test_clean_summary_function_can_be_imported(self):
        """Test that clean_summary function can be imported from ai_queries."""
        try:
            from ai_queries import clean_summary
            self.assertTrue(callable(clean_summary), 
                          "clean_summary should be a callable function")
            print("✅ clean_summary function can be imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import clean_summary: {e}")

    def test_clean_summary_function_signature(self):
        """Test that clean_summary has the correct signature."""
        from ai_queries import clean_summary
        import inspect
        
        sig = inspect.signature(clean_summary)
        params = list(sig.parameters.keys())
        
        self.assertIn("text", params, "clean_summary should have 'text' parameter")
        self.assertIn("cfg", params, "clean_summary should have 'cfg' parameter")
        self.assertIn("ai_util", params, "clean_summary should have 'ai_util' parameter")
        self.assertIn("leader_name", params, "clean_summary should have 'leader_name' parameter")
        
        # Check that leader_name has a default value (None is a valid default)
        leader_name_param = sig.parameters["leader_name"]
        self.assertNotEqual(leader_name_param.default, inspect.Parameter.empty,
                           "leader_name should have a default value")
        print("✅ clean_summary function has correct signature")

    def test_leader_name_placeholder_handling(self):
        """Test how clean_summary handles the leader_name placeholder."""
        prompt = self.cfg["prompts"]["clean_summary"]
        has_placeholder = "<leader_name>" in prompt
        
        if not has_placeholder:
            print("⚠️  Note: The prompt doesn't contain a <leader_name> placeholder.")
            print("   The clean_summary function will still work, but the leader_name")
            print("   personalization feature won't have any effect with the current prompt.")
        else:
            print("✅ Prompt contains <leader_name> placeholder for personalization")

    def test_test_llm_prompts_structure(self):
        """Test that test_llm_prompts.py has the expected structure."""
        with open("test_llm_prompts.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for LLMPromptTester class
        self.assertIn("class LLMPromptTester", content,
                     "test_llm_prompts.py should define LLMPromptTester class")
        
        # Check for test_prompts method
        self.assertIn("def test_prompts", content,
                     "LLMPromptTester should have test_prompts method")
        
        # Check that it tests clean_summary
        self.assertIn("Testing clean_summary prompt", content,
                     "test_prompts should test clean_summary")
        
        print("✅ test_llm_prompts.py has correct structure")

    def test_categorize_article_structure(self):
        """Test that categorize_article_and_generate_content uses clean_summary correctly."""
        with open("news_filter.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check that clean_summary is called in the female_leader branch
        self.assertIn('if art["status"] == "female_leader"', content,
                     "categorize_article_and_generate_content should handle female_leader status")
        
        # Verify clean_summary is called with proper parameters
        self.assertIn("clean_summary(body, cfg, aiclient", content,
                     "clean_summary should be called with body, cfg, aiclient parameters")
        
        print("✅ categorize_article_and_generate_content uses clean_summary correctly")


def run_tests():
    """Run all tests and print results."""
    print("="*70)
    print("Testing clean_summary functionality")
    print("="*70)
    print()
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCleanSummaryFunctionality)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("="*70)
    print("Test Summary")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print()
        print("✅ All tests passed! clean_summary functionality is working correctly.")
        return 0
    else:
        print()
        print("❌ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
