#!/usr/bin/env python3
"""
Verification script to ensure proper setup before running experiments.
Run this after creating your .env file to verify everything is configured correctly.
"""

import os
import subprocess
from pathlib import Path

def check_mark(condition):
    """Return checkmark or X based on condition"""
    return "✅" if condition else "❌"

def main():
    print("\n" + "=" * 70)
    print("SETUP VERIFICATION")
    print("=" * 70)

    checks_passed = []

    # Check 1: .env file exists
    env_file = Path(__file__).parent / ".env"
    env_exists = env_file.exists()
    print(f"\n{check_mark(env_exists)} .env file exists: {env_exists}")
    checks_passed.append(env_exists)

    if not env_exists:
        print("   ⚠️  Create .env file: cp .env.example .env")
        print("   Then add your Gemini API key")

    # Check 2: .env has API key
    if env_exists:
        with open(env_file) as f:
            content = f.read()
        has_key = "GEMINI_API_KEY" in content and "your_api_key_here" not in content
        print(f"{check_mark(has_key)} API key configured: {has_key}")
        checks_passed.append(has_key)

        if not has_key:
            print("   ⚠️  Add your API key to .env file")

    # Check 3: .env is ignored by Git
    try:
        result = subprocess.run(
            ["git", "check-ignore", "PART_2/.env"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True
        )
        git_ignored = result.returncode == 0
        print(f"{check_mark(git_ignored)} .env ignored by Git: {git_ignored}")
        checks_passed.append(git_ignored)

        if not git_ignored:
            print("   ⚠️  Add '.env' to .gitignore file!")
    except:
        print("⚠️  Could not check Git status (Git not available)")

    # Check 4: Can import required packages
    try:
        from dotenv import load_dotenv
        from google import genai
        packages_ok = True
        print(f"{check_mark(packages_ok)} Required packages installed: {packages_ok}")
        checks_passed.append(packages_ok)
    except ImportError as e:
        packages_ok = False
        print(f"{check_mark(packages_ok)} Required packages installed: {packages_ok}")
        print(f"   ⚠️  Missing package: {e}")
        print(f"   Install with: pip install -r requirements.txt")
        checks_passed.append(packages_ok)

    # Check 5: Can load configuration
    try:
        import config
        config_ok = True
        print(f"{check_mark(config_ok)} Configuration loads successfully: {config_ok}")
        checks_passed.append(config_ok)
    except Exception as e:
        config_ok = False
        print(f"{check_mark(config_ok)} Configuration loads successfully: {config_ok}")
        print(f"   ⚠️  Error: {e}")
        checks_passed.append(config_ok)

    # Check 6: PART_1 results exist
    part1_results = Path(__file__).parent.parent / "PART_1" / "S&P500_results_20251120_142343.csv"
    part1_ok = part1_results.exists()
    print(f"{check_mark(part1_ok)} PART_1 results available: {part1_ok}")
    checks_passed.append(part1_ok)

    if not part1_ok:
        print(f"   ⚠️  Expected file: {part1_results}")

    # Summary
    print("\n" + "=" * 70)
    total = len(checks_passed)
    passed = sum(checks_passed)

    if passed == total:
        print(f"🎉 ALL CHECKS PASSED ({passed}/{total})")
        print("\nYou're ready to run experiments!")
        print("\nNext steps:")
        print("  1. Test configuration: python config.py")
        print("  2. Run pilot test: python 00_pilot_test.py --companies 5")
        print("  3. Run full experiment: python run_experiment.py")
    else:
        print(f"⚠️  {passed}/{total} CHECKS PASSED")
        print(f"\nPlease fix the issues above before proceeding.")

    print("=" * 70 + "\n")

    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)