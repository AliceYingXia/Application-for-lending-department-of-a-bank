"""
Configuration management for Gemini API and experiment settings.
Loads environment variables from .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from PART_2 directory
PART_2_DIR = Path(__file__).parent
ENV_FILE = PART_2_DIR / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
    print(f"✓ Loaded environment variables from {ENV_FILE}")
else:
    print(f"⚠️  Warning: .env file not found at {ENV_FILE}")
    print(f"   Please copy .env.example to .env and add your API key")

# ==================== Gemini API Configuration ==================== #

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
    raise ValueError(
        "GEMINI_API_KEY not set! Please create .env file with your API key.\n"
        "See .env.example for template."
    )

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))
GEMINI_TOP_P = float(os.getenv("GEMINI_TOP_P", "0.95"))

# ==================== Cost Controls ==================== #

MAX_API_BUDGET_USD = float(os.getenv("MAX_API_BUDGET_USD", "20.00"))
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "15"))

# ==================== Experiment Settings ==================== #

TEST_SAMPLE_SIZE = int(os.getenv("TEST_SAMPLE_SIZE", "50"))
FORECAST_YEAR = int(os.getenv("FORECAST_YEAR", "2024"))

# ==================== Directory Paths ==================== #

PART_1_DIR = PART_2_DIR.parent / "PART_1"
DATA_DIR = PART_2_DIR / "data"
FORECASTS_DIR = PART_2_DIR / "forecasts"
RESULTS_DIR = PART_2_DIR / "results"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
FORECASTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

(DATA_DIR / "historical").mkdir(exist_ok=True)
(DATA_DIR / "actual").mkdir(exist_ok=True)
(FORECASTS_DIR / "gemini").mkdir(exist_ok=True)
(RESULTS_DIR / "plots").mkdir(exist_ok=True)

# ==================== Part 1 Results File ==================== #

PART_1_RESULTS = PART_1_DIR / "S&P500_results_20251120_142343.csv"
if not PART_1_RESULTS.exists():
    print(f"⚠️  Warning: Part 1 results not found at {PART_1_RESULTS}")

# ==================== Gemini Client Helpers ==================== #

def get_gemini_client():
    """
    Initialize and return a Gemini client configured with API key.
    Uses the new google.genai package (replaces deprecated google.generativeai).

    Returns:
        Configured Gemini client
    """
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    return client

def get_generation_config():
    """
    Get the generation configuration for Gemini API calls.

    Returns:
        dict: Configuration parameters
    """
    return {
        "temperature": GEMINI_TEMPERATURE,
        "top_p": GEMINI_TOP_P,
        "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
        "response_mime_type": "application/json",  # For structured output
    }

# ==================== Display Configuration ==================== #

def print_config():
    """Print current configuration (without exposing API key)"""
    print("\n" + "=" * 70)
    print("GEMINI FORECASTING CONFIGURATION")
    print("=" * 70)
    print(f"Model: {GEMINI_MODEL}")
    print(f"Temperature: {GEMINI_TEMPERATURE}")
    print(f"Max Output Tokens: {GEMINI_MAX_OUTPUT_TOKENS}")
    print(f"API Key: {'*' * 20}{GEMINI_API_KEY[-8:] if GEMINI_API_KEY else 'NOT SET'}")
    print(f"\nExperiment Settings:")
    print(f"  Test Sample Size: {TEST_SAMPLE_SIZE}")
    print(f"  Forecast Year: {FORECAST_YEAR}")
    print(f"  Max Budget: ${MAX_API_BUDGET_USD:.2f}")
    print(f"\nDirectories:")
    print(f"  PART_2: {PART_2_DIR}")
    print(f"  Data: {DATA_DIR}")
    print(f"  Forecasts: {FORECASTS_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    # Test configuration
    print_config()

    # Test Gemini client initialization
    try:
        print("Testing Gemini client initialization...")
        client = get_gemini_client()
        print(f"✅ Successfully created Gemini client")
        print(f"   Model to use: {GEMINI_MODEL}")
    except Exception as e:
        print(f"❌ Failed to create Gemini client: {e}")