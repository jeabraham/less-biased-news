#!/usr/bin/env python3
"""
Simple validation script for test_pipeline.py
Tests that the script structure is correct without requiring full dependencies.
"""

import ast
import sys

def validate_test_pipeline():
    """Validate the structure of test_pipeline.py"""
    print("Validating test_pipeline.py structure...")
    
    errors = []
    
    # Read the file
    try:
        with open('test_pipeline.py', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("ERROR: test_pipeline.py not found")
        return False
    
    # Parse the AST
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"ERROR: Syntax error in test_pipeline.py: {e}")
        return False
    
    # Check for required functions
    required_functions = [
        'parse_pipeline',
        'validate_pipeline_steps',
        'apply_pipeline_step',
        'process_article_with_pipeline',
        'filter_articles_by_date',
        'lazy_imports',
        'main'
    ]
    
    found_functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            found_functions.append(node.name)
    
    for func in required_functions:
        if func in found_functions:
            print(f"✓ Found function: {func}")
        else:
            errors.append(f"Missing function: {func}")
            print(f"✗ Missing function: {func}")
    
    # Check for argparse usage
    has_argparse = 'argparse' in content
    if has_argparse:
        print("✓ Uses argparse")
    else:
        errors.append("Does not use argparse")
        print("✗ Does not use argparse")
    
    # Check for required options
    required_options = [
        '--queries',
        '--pipeline',
        '--newer-than',
        '--format-text'
    ]
    
    for option in required_options:
        if option in content:
            print(f"✓ Has option: {option}")
        else:
            errors.append(f"Missing option: {option}")
            print(f"✗ Missing option: {option}")
    
    # Check for pipeline steps documentation
    pipeline_steps = [
        'categorize',
        'spin_genders',
        'clean_summary',
        'short_summary',
        'image_classification'
    ]
    
    for step in pipeline_steps:
        if step in content:
            print(f"✓ Mentions pipeline step: {step}")
        else:
            errors.append(f"Missing pipeline step: {step}")
            print(f"✗ Missing pipeline step: {step}")
    
    # Summary
    print("\n" + "="*60)
    if errors:
        print(f"VALIDATION FAILED with {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("VALIDATION PASSED - All required components found!")
        return True

if __name__ == '__main__':
    success = validate_test_pipeline()
    sys.exit(0 if success else 1)
