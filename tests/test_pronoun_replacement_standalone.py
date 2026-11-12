#!/usr/bin/env python3
"""
Standalone test for the replace_male_pronouns_with_neutral function.
This test can be run without installing the full project dependencies.

Note: This file intentionally duplicates the replace_male_pronouns_with_neutral
function from news_filter.py to allow standalone testing without requiring the
installation of spacy and other heavy dependencies. This makes it easy to verify
the pronoun replacement logic in isolation.
"""

import re


def replace_male_pronouns_with_neutral(text):
    """
    Replace masculine pronouns with gender-neutral ones,
    preserving capitalization automatically and distinguishing
    'his' determiner vs. 'his' possessive pronoun.
    
    The logic:
    - "his" followed by adverbs at end/before punctuation → "theirs" (possessive pronoun)
    - "his" followed by a word → "their" (determiner)
    - "his" at end or before punctuation → "theirs" (possessive pronoun)
    """

    if not isinstance(text, str):
        raise ValueError("The input argument 'text' must be a string.")

    # Order matters: we need to handle 'his' carefully to distinguish determiner vs. possessive pronoun
    replacements = [
        # himself
        (r"\bhimself\b", "themself"),

        # 'his' before common adverbs that typically come at END (followed by punctuation or end of string)
        # This handles cases like "his too." or "his alone," or "his also!"
        (r"\bhis\b(?=\s+(?:too|alone|also|as well|entirely|exclusively|only)(?:\s*[,;.!?]|\s*$))", "theirs"),

        # 'his' determiner (followed by a word or quoted phrase) → their
        # This catches determiners before words and quoted phrases.
        (r"\bhis\b(?=\s+(?:\w|[\"'“‘]))", "their"),

        # standalone 'his' → theirs (possessive pronoun)
        # This catches everything else: end of string, before punctuation, etc.
        (r"\bhis\b", "theirs"),

        # Repair malformed possessives returned by the rewrite model.
        (r"\btheirs\b(?=\s+(?!(?:is|are|was|were|has|have|does|do|did|can|could|will|would|should|might|must|too|alone|also|entirely|exclusively|only)\b)(?!as\s+well\b)(?:\w|[\"'“‘]))", "their"),

        # him → them
        (r"\bhim\b", "them"),

        # he → they
        (r"\bhe\b", "they"),
    ]

    def match_case(repl, original):
        """Return repl but matching the capitalization pattern of original."""
        if original.isupper():
            return repl.upper()
        if original[0].isupper():
            return repl.capitalize()
        return repl

    def replacer(match, repl):
        word = match.group(0)
        return match_case(repl, word)

    for pattern, repl in replacements:
        text = re.sub(
            pattern,
            lambda m, r=repl: replacer(m, r),
            text,
            flags=re.IGNORECASE
        )

    verb_replacements = {
        "does": "do",
        "doesn't": "don't",
        "is": "are",
        "was": "were",
        "has": "have",
    }

    def replace_singular_verb(match):
        return match.group(1) + match_case(verb_replacements[match.group(2).lower()], match.group(2))

    text = re.sub(
        r"\b(they\s+)(doesn't|does|is|was|has)\b",
        replace_singular_verb,
        text,
        flags=re.IGNORECASE,
    )

    return text


def run_tests():
    """Run all test cases and report results."""
    
    test_cases = [
        # (input, expected_output, description)
        (
            "The book is his.",
            "The book is theirs.",
            "Possessive pronoun at end"
        ),
        (
            "This is his!",
            "This is theirs!",
            "Possessive pronoun before punctuation"
        ),
        (
            "his book",
            "their book",
            "Determiner before noun"
        ),
        (
            "his lifelong dream",
            "their lifelong dream",
            "Determiner before adjective+noun"
        ),
        (
            "his country",
            "their country",
            "Determiner before noun"
        ),
        (
            "He gave it to him.",
            "They gave it to them.",
            "Subject and object pronouns"
        ),
        (
            "He himself did it.",
            "They themself did it.",
            "Reflexive pronoun"
        ),
        (
            'And speaking to Fox News on Tuesday night, Trump touted theirs "ultimate power" '
            "of veto to shut down bills passed by Congress that they does not like.",
            'And speaking to Fox News on Tuesday night, Trump touted their "ultimate power" '
            "of veto to shut down bills passed by Congress that they do not like.",
            "Repair possessive and verb agreement"
        ),
        (
            "Theirs is the final decision.",
            "Theirs is the final decision.",
            "Preserve possessive pronoun before verb"
        ),
        (
            "Is this his?",
            "Is this theirs?",
            "Possessive pronoun with question mark"
        ),
        (
            "His bike is red, and that one is his too.",
            "Their bike is red, and that one is theirs too.",
            "Mixed usage"
        ),
        (
            "his car, his house",
            "their car, their house",
            "Multiple determiners"
        ),
        (
            "The victory was his, his alone.",
            "The victory was theirs, theirs alone.",
            "Mixed possessive pronouns"
        ),
        (
            "his only wish",
            "their only wish",
            "'only' as adjective"
        ),
        (
            "That is his alone.",
            "That is theirs alone.",
            "'alone' as adverb"
        ),
        (
            "It was his only.",
            "It was theirs only.",
            "'only' as adverb at end"
        ),
        (
            "his wonderful achievement",
            "their wonderful achievement",
            "Adjective before noun"
        ),
        (
            "His book is great.",
            "Their book is great.",
            "Capitalization preserved"
        ),
        (
            "HIS BOOK",
            "THEIR BOOK",
            "All caps preserved"
        ),
        (
            "Bo Horvat was named to Team Canada's Olympic roster after a successful "
            "World Championships performance and strong regular season, fulfilling his "
            "lifelong dream of representing his country. Hockey Canada selected Horvat "
            "as one of four forwards not on the 4 Nations team, with management citing "
            "his versatility and faceoff skills as key factors in his inclusion.",
            "Bo Horvat was named to Team Canada's Olympic roster after a successful "
            "World Championships performance and strong regular season, fulfilling their "
            "lifelong dream of representing their country. Hockey Canada selected Horvat "
            "as one of four forwards not on the 4 Nations team, with management citing "
            "their versatility and faceoff skills as key factors in their inclusion.",
            "Original failing example from problem statement"
        ),
    ]
    
    passed = 0
    failed = 0
    
    print("Running pronoun replacement tests...\n")
    
    for input_text, expected, description in test_cases:
        result = replace_male_pronouns_with_neutral(input_text)
        if result == expected:
            print(f"✓ PASS: {description}")
            passed += 1
        else:
            print(f"✗ FAIL: {description}")
            print(f"  Input:    '{input_text}'")
            print(f"  Expected: '{expected}'")
            print(f"  Got:      '{result}'")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print(f"{'='*60}\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
