#!/usr/bin/env python3
"""
Gemini-Based Financial Forecasting Script

Uses Gemini 2.5 Flash with Chain-of-Thought reasoning to forecast 2024 financial statements
based on historical data (2020-2023).
"""

import json
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import config
from typing import Dict, Any


# Rate limiting
class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.calls = []

    def wait_if_needed(self):
        """Wait if we've hit the rate limit"""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < 60]

        if len(self.calls) >= self.max_per_minute:
            # Wait until the oldest call is 60 seconds old
            sleep_time = 60 - (now - self.calls[0])
            if sleep_time > 0:
                print(f"  ⏳ Rate limit reached, waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        self.calls.append(time.time())


# Cost tracking
class CostTracker:
    """Track API costs"""
    def __init__(self, max_budget_usd: float):
        self.max_budget_usd = max_budget_usd
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

        # Gemini 2.5 Flash pricing (as of Feb 2026)
        # FREE during preview, but we'll track as if it costs money for safety
        self.input_price_per_1m = 0.0  # Currently FREE
        self.output_price_per_1m = 0.0  # Currently FREE

    def add_usage(self, input_tokens: int, output_tokens: int):
        """Add token usage"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        input_cost = (input_tokens / 1_000_000) * self.input_price_per_1m
        output_cost = (output_tokens / 1_000_000) * self.output_price_per_1m
        self.total_cost_usd += input_cost + output_cost

    def check_budget(self) -> bool:
        """Check if we're within budget"""
        return self.total_cost_usd < self.max_budget_usd

    def get_summary(self) -> str:
        """Get cost summary"""
        return (
            f"Total Input Tokens: {self.total_input_tokens:,}\n"
            f"Total Output Tokens: {self.total_output_tokens:,}\n"
            f"Total Cost: ${self.total_cost_usd:.2f} / ${self.max_budget_usd:.2f}"
        )


def create_forecast_prompt(ticker: str, historical_data: Dict[str, Any]) -> str:
    """
    Create Chain-of-Thought prompt for Gemini to forecast financial statements.
    Optimized to be concise and avoid JSON truncation.

    Args:
        ticker: Stock ticker symbol
        historical_data: Historical financial data (2020-2023)

    Returns:
        Prompt string
    """
    years = sorted(historical_data.keys())

    # Create compact historical data summary
    prompt = f"""Forecast 2024 financials for {ticker} based on {years[0]}-{years[-1]} data.

Historical Data:
"""

    for year in years:
        data = historical_data[year]
        is_data = data['income_statement']
        bs_data = data['balance_sheet']

        prompt += f"\n{year}: "
        if is_data.get('revenue'):
            prompt += f"Rev ${is_data['revenue']:,.0f}, "
        if is_data.get('net_income'):
            prompt += f"NI ${is_data['net_income']:,.0f}, "
        if bs_data.get('total_assets'):
            prompt += f"Assets ${bs_data['total_assets']:,.0f}, "
        if bs_data.get('equity'):
            prompt += f"Equity ${bs_data['equity']:,.0f}"
        prompt += "\n"

    prompt += """
Task: Analyze trends and forecast 2024.

Output ONLY this JSON (NUMBERS ONLY, NO TEXT):

{
  "forecast_2024": {
    "income_statement": {
      "revenue": <number>,
      "cogs": <number>,
      "gross_profit": <number>,
      "operating_expense": <number>,
      "operating_income": <number>,
      "net_income": <number>
    },
    "balance_sheet": {
      "total_assets": <number>,
      "current_assets": <number>,
      "cash": <number>,
      "total_liabilities": <number>,
      "current_liabilities": <number>,
      "equity": <number>
    },
    "cash_flow": {
      "operating_cash_flow": <number>,
      "capex": <number>,
      "free_cash_flow": <number>
    }
  }
}

Return ONLY valid JSON with numbers. NO text fields."""

    return prompt


def repair_json_with_llm(client, malformed_json: str, ticker: str, rate_limiter: RateLimiter) -> Dict[str, Any]:
    """
    Use LLM to extract forecast numbers from malformed JSON.

    Args:
        client: Gemini client
        malformed_json: The malformed JSON string
        ticker: Stock ticker
        rate_limiter: Rate limiter instance

    Returns:
        Repaired forecast data dictionary
    """
    rate_limiter.wait_if_needed()

    repair_prompt = f"""The following JSON is malformed but contains forecast data for {ticker}:

{malformed_json}

Extract ALL the forecast numbers and return ONLY valid JSON in this EXACT format:

{{
  "forecast_2024": {{
    "income_statement": {{
      "revenue": <number>,
      "cogs": <number>,
      "gross_profit": <number>,
      "operating_expense": <number>,
      "operating_income": <number>,
      "net_income": <number>
    }},
    "balance_sheet": {{
      "total_assets": <number>,
      "current_assets": <number>,
      "cash": <number>,
      "total_liabilities": <number>,
      "current_liabilities": <number>,
      "equity": <number>
    }},
    "cash_flow": {{
      "operating_cash_flow": <number>,
      "capex": <number>,
      "free_cash_flow": <number>
    }}
  }}
}}

Extract the numbers from the malformed JSON and return ONLY valid JSON."""

    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=repair_prompt,
            config={
                "temperature": 0.0,
                "max_output_tokens": 8192,  # Increased to fix truncation issues
                "response_mime_type": "application/json",
            }
        )

        repair_text = response.text.strip()

        # Clean up
        if not repair_text.startswith('{'):
            start = repair_text.find('{')
            if start != -1:
                repair_text = repair_text[start:]

        if not repair_text.endswith('}'):
            end = repair_text.rfind('}')
            if end != -1:
                repair_text = repair_text[:end+1]

        return json.loads(repair_text)

    except Exception as e:
        print(f"    Repair attempt failed: {e}")
        return None


def forecast_with_gemini(ticker: str, historical_data: Dict[str, Any], client, rate_limiter: RateLimiter, cost_tracker: CostTracker) -> Dict[str, Any]:
    """
    Use Gemini to forecast 2024 financial statements.

    Args:
        ticker: Stock ticker symbol
        historical_data: Historical financial data
        client: Gemini client
        rate_limiter: Rate limiter instance
        cost_tracker: Cost tracker instance

    Returns:
        Forecast dictionary
    """
    # Rate limiting
    rate_limiter.wait_if_needed()

    # Create prompt
    prompt = create_forecast_prompt(ticker, historical_data)

    max_retries = 3  # Increased to 3 for temperature variation
    for attempt in range(max_retries):
        try:
            # Call Gemini API with timeout
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("API call timeout")

            # Set 60 second timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)

            # Use higher temperature on retries to get different outputs
            temperature = config.GEMINI_TEMPERATURE if attempt == 0 else 0.3

            try:
                response = client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=prompt,
                    config={
                        "temperature": temperature,
                        "top_p": config.GEMINI_TOP_P,
                        "max_output_tokens": 8192,  # Increased from 4096 to fix AMZN/MET truncation
                        "response_mime_type": "application/json",
                    }
                )
            finally:
                signal.alarm(0)  # Cancel timeout

            # Parse response with cleanup
            response_text = response.text.strip()

            # Clean up common JSON issues
            if not response_text.startswith('{'):
                start = response_text.find('{')
                if start != -1:
                    response_text = response_text[start:]

            if not response_text.endswith('}'):
                end = response_text.rfind('}')
                if end != -1:
                    response_text = response_text[:end+1]

            forecast_data = json.loads(response_text)

            # Track usage (Gemini API response includes usage metadata)
            if hasattr(response, 'usage_metadata'):
                cost_tracker.add_usage(
                    input_tokens=response.usage_metadata.prompt_token_count,
                    output_tokens=response.usage_metadata.candidates_token_count
                )

            return {
                "ticker": ticker,
                "status": "success",
                "forecast": forecast_data
            }

        except TimeoutError:
            if attempt < max_retries - 1:
                print(f"  ⏱️  Timeout for {ticker}, retrying ({attempt+1}/{max_retries})...")
                time.sleep(5)
                continue
            else:
                print(f"  ❌ Timeout for {ticker} after {max_retries} attempts")
                return {
                    "ticker": ticker,
                    "status": "timeout_error",
                    "error": f"API call timeout after {max_retries} attempts"
                }

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"  ⚠️  JSON error for {ticker} (attempt {attempt+1}/{max_retries}), retrying...")
                time.sleep(3)
                continue
            else:
                # Last resort: Use LLM to repair the JSON
                print(f"  🔧 Attempting LLM-based JSON repair for {ticker}...")
                try:
                    repaired_data = repair_json_with_llm(client, response_text, ticker, rate_limiter)
                    if repaired_data:
                        print(f"  ✅ Successfully repaired JSON for {ticker}")
                        return {
                            "ticker": ticker,
                            "status": "success",
                            "forecast": repaired_data
                        }
                except Exception as repair_error:
                    print(f"  ❌ JSON repair failed for {ticker}: {repair_error}")

                print(f"  ❌ JSON parsing error for {ticker} after {max_retries} attempts: {e}")
                return {
                    "ticker": ticker,
                    "status": "json_error",
                    "error": str(e),
                    "raw_response": response_text if 'response_text' in locals() else None
                }

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠️  Error for {ticker}, retrying ({attempt+1}/{max_retries}): {e}")
                time.sleep(5)
                continue
            else:
                print(f"  ❌ API error for {ticker}: {e}")
                return {
                    "ticker": ticker,
                    "status": "api_error",
                    "error": str(e)
                }


def run_forecasting_experiment():
    """
    Run the Gemini forecasting experiment on all sample companies.
    """
    print("=" * 70)
    print("GEMINI FORECASTING EXPERIMENT")
    print("=" * 70)
    print(f"Model: {config.GEMINI_MODEL}")
    print(f"Temperature: {config.GEMINI_TEMPERATURE}")
    print(f"Max Budget: ${config.MAX_API_BUDGET_USD:.2f}")
    print(f"Rate Limit: {config.MAX_REQUESTS_PER_MINUTE} requests/minute")
    print("=" * 70)

    # Load test sample
    sample_file = config.DATA_DIR / "test_sample.csv"
    sample = pd.read_csv(sample_file)
    print(f"\nCompanies to forecast: {len(sample)}")

    # Initialize Gemini client
    print("\nInitializing Gemini client...")
    client = config.get_gemini_client()
    print("✓ Client initialized")

    # Initialize rate limiter and cost tracker
    rate_limiter = RateLimiter(config.MAX_REQUESTS_PER_MINUTE)
    cost_tracker = CostTracker(config.MAX_API_BUDGET_USD)

    # Process each company
    print("\nGenerating forecasts with Gemini...")
    results = []
    success_count = 0
    failed_count = 0

    for idx, row in tqdm(sample.iterrows(), total=len(sample), desc="Forecasting"):
        ticker = row['ticker']

        # Check if forecast already exists (for resuming)
        output_file = config.FORECASTS_DIR / "gemini" / f"{ticker}.json"
        if output_file.exists():
            print(f"  ✓ Skipping {ticker} (forecast already exists)")
            success_count += 1
            continue

        # Check budget
        if not cost_tracker.check_budget():
            print(f"\n⚠️  Budget limit reached! Stopping at {idx+1}/{len(sample)} companies.")
            break

        # Load historical data
        historical_file = config.DATA_DIR / "historical" / f"{ticker}.json"

        if not historical_file.exists():
            print(f"  ⚠️  No historical data for {ticker}, skipping...")
            failed_count += 1
            continue

        with open(historical_file, 'r') as f:
            historical_data = json.load(f)

        # Generate forecast
        result = forecast_with_gemini(ticker, historical_data, client, rate_limiter, cost_tracker)
        results.append(result)

        # Save forecast if successful
        if result['status'] == 'success':
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            success_count += 1
        else:
            failed_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("FORECASTING SUMMARY")
    print("=" * 70)
    print(f"Total companies: {len(sample)}")
    print(f"Successful forecasts: {success_count}")
    print(f"Failed forecasts: {failed_count}")
    print(f"\n{cost_tracker.get_summary()}")
    print(f"\nForecasts saved to: {config.FORECASTS_DIR / 'gemini'}")
    print("=" * 70 + "\n")

    return success_count, failed_count


if __name__ == "__main__":
    success, failed = run_forecasting_experiment()

    if success > 0:
        print(f"✅ Forecasting complete!")
        print(f"   Generated {success} forecasts")
        if failed > 0:
            print(f"   ⚠️  {failed} forecasts failed")
    else:
        print(f"❌ Forecasting failed!")
        print(f"   No successful forecasts generated")