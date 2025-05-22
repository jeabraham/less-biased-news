
"""
Test the spin_genders function using a sample input from a text file and configuration from config.yaml.
"""
import yaml
from pathlib import Path
from ai_queries import spin_genders

from ai_utils import AIUtils

# Load text from test file
test_file = Path("test_article.txt")
text = test_file.read_text(encoding="utf-8").strip()

# Load configuration from YAML file
config_file = Path("config.yaml")
with config_file.open("r", encoding="utf-8") as file:
    cfg = yaml.safe_load(file)

# Mock or initialize an AIUtils instance (update this to match AIUtils implementation)
ai_util = AIUtils(cfg)

# Call the function with the loaded text and config
result = spin_genders(text, cfg, ai_util)

# Output or assert the result (customize assertion/validation as needed)
print("Spun genders result:")
print(result)
