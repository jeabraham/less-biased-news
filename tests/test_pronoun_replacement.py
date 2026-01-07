"""
Tests for the replace_male_pronouns_with_neutral function.

This module tests the pronoun replacement logic to ensure correct handling
of possessive determiners (his → their) vs. possessive pronouns (his → theirs).
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from news_filter import replace_male_pronouns_with_neutral


class TestPronounReplacement(unittest.TestCase):
    """Test cases for pronoun replacement functionality."""

    def test_possessive_pronoun_at_end(self):
        """Test 'his' as possessive pronoun at end of sentence."""
        self.assertEqual(
            replace_male_pronouns_with_neutral("The book is his."),
            "The book is theirs."
        )

    def test_possessive_pronoun_before_punctuation(self):
        """Test 'his' as possessive pronoun before exclamation."""
        self.assertEqual(
            replace_male_pronouns_with_neutral("This is his!"),
            "This is theirs!"
        )

    def test_determiner_before_noun(self):
        """Test 'his' as determiner before a simple noun."""
        self.assertEqual(
            replace_male_pronouns_with_neutral("his book"),
            "their book"
        )

    def test_determiner_before_adjective_noun(self):
        """Test 'his' as determiner before adjective + noun."""
        self.assertEqual(
            replace_male_pronouns_with_neutral("his lifelong dream"),
            "their lifelong dream"
        )

    def test_determiner_before_country(self):
        """Test 'his' as determiner in the original failing example."""
        self.assertEqual(
            replace_male_pronouns_with_neutral("his country"),
            "their country"
        )

    def test_subject_and_object_pronouns(self):
        """Test replacement of he → they and him → them."""
        self.assertEqual(
            replace_male_pronouns_with_neutral("He gave it to him."),
            "They gave it to them."
        )

    def test_reflexive_pronoun(self):
        """Test replacement of himself → themself."""
        self.assertEqual(
            replace_male_pronouns_with_neutral("He himself did it."),
            "They themself did it."
        )

    def test_possessive_pronoun_with_question_mark(self):
        """Test 'his' as possessive pronoun with question mark."""
        self.assertEqual(
            replace_male_pronouns_with_neutral("Is this his?"),
            "Is this theirs?"
        )

    def test_mixed_usage(self):
        """Test sentence with both determiner and possessive pronoun."""
        self.assertEqual(
            replace_male_pronouns_with_neutral("His bike is red, and that one is his too."),
            "Their bike is red, and that one is theirs too."
        )

    def test_multiple_determiners(self):
        """Test multiple uses of 'his' as determiner."""
        self.assertEqual(
            replace_male_pronouns_with_neutral("his car, his house"),
            "their car, their house"
        )

    def test_mixed_possessive_pronouns(self):
        """Test multiple uses of 'his' as possessive pronoun."""
        self.assertEqual(
            replace_male_pronouns_with_neutral("The victory was his, his alone."),
            "The victory was theirs, theirs alone."
        )

    def test_only_as_adjective(self):
        """Test 'only' used as an adjective (not adverb)."""
        self.assertEqual(
            replace_male_pronouns_with_neutral("his only wish"),
            "their only wish"
        )

    def test_alone_as_adverb(self):
        """Test 'alone' used as an adverb after possessive pronoun."""
        self.assertEqual(
            replace_male_pronouns_with_neutral("That is his alone."),
            "That is theirs alone."
        )

    def test_only_as_adverb(self):
        """Test 'only' used as an adverb at end of sentence."""
        self.assertEqual(
            replace_male_pronouns_with_neutral("It was his only."),
            "It was theirs only."
        )

    def test_adjective_before_noun(self):
        """Test determiner before adjective + noun."""
        self.assertEqual(
            replace_male_pronouns_with_neutral("his wonderful achievement"),
            "their wonderful achievement"
        )

    def test_capitalization_preserved(self):
        """Test that capitalization is preserved."""
        self.assertEqual(
            replace_male_pronouns_with_neutral("His book is great."),
            "Their book is great."
        )
        self.assertEqual(
            replace_male_pronouns_with_neutral("HIS BOOK"),
            "THEIR BOOK"
        )

    def test_original_failing_example(self):
        """Test the original failing example from the problem statement."""
        input_text = (
            "Bo Horvat was named to Team Canada's Olympic roster after a successful "
            "World Championships performance and strong regular season, fulfilling his "
            "lifelong dream of representing his country. Hockey Canada selected Horvat "
            "as one of four forwards not on the 4 Nations team, with management citing "
            "his versatility and faceoff skills as key factors in his inclusion."
        )
        expected_text = (
            "Bo Horvat was named to Team Canada's Olympic roster after a successful "
            "World Championships performance and strong regular season, fulfilling their "
            "lifelong dream of representing their country. Hockey Canada selected Horvat "
            "as one of four forwards not on the 4 Nations team, with management citing "
            "their versatility and faceoff skills as key factors in their inclusion."
        )
        self.assertEqual(
            replace_male_pronouns_with_neutral(input_text),
            expected_text
        )


if __name__ == "__main__":
    unittest.main()
