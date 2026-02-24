"""
Gemini Financial Ratio Analyzer
================================
Reads pre-extracted income-statement and balance-sheet PDFs from
data/extracted/ (produced by step_2_extract_pages.py) and calls
Gemini twice per run:

  Call 1 – Chain-of-thought
    Step 1 : Identify which line items are needed for each ratio.
    Step 2 : Extract the actual values and calculate every ratio.

  Call 2 – JSON output
    Takes the text from Call 1 and returns a structured JSON with
    value, formula, inputs, and one reason per metric.

Each company is run N_RUNS times (default 10) to measure consistency.
A consistency report (mean, std, cv, min, max per metric) is saved
alongside the individual run JSONs.

Output
------
  data/extracted/
    <Company>_financial_ratios_run_01.json
    ...
    <Company>_financial_ratios_run_10.json
    <Company>_consistency_report.json

Usage
-----
    python step_3_gemini_analyzer.py                          # all companies
    python step_3_gemini_analyzer.py data/extracted/EM_2024  # single company stem
"""

import io
import json
import math
import os
import sys
from pathlib import Path

import fitz  # PyMuPDF — used only for combining the two extracted PDFs
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────

load_dotenv()

GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL       = os.getenv("GEMINI_MODEL",       "gemini-2.5-flash")
GEMINI_TEMPERATURE = 0.7
GEMINI_MAX_TOKENS  = 16448
GEMINI_TOP_P       = float(os.getenv("GEMINI_TOP_P", "0.95"))

N_RUNS        = 10
EXTRACTED_DIR = Path("data/extracted")

METRICS = [
    "net_income",
    "cost_to_income_ratio",
    "quick_ratio",
    "debt_to_equity_ratio",
    "debt_to_assets_ratio",
    "debt_to_capital_ratio",
    "debt_to_ebitda_ratio",
    "interest_coverage_ratio",
]

# CV threshold below which a metric is considered consistent
CONSISTENCY_CV_THRESHOLD = 0.05   # 5 %


# ── PDF helpers ────────────────────────────────────────────────────────────

def _combine_pdfs(income_path: Path, balance_path: Path) -> tuple:
    """
    Merge the income-statement and balance-sheet PDFs into a single
    in-memory PDF using PyMuPDF.

    Returns:
        (combined_pdf_bytes, income_positions, balance_positions)
        where positions are 1-indexed page numbers within the combined PDF.
    """
    dst = fitz.open()

    income_doc = fitz.open(str(income_path))
    dst.insert_pdf(income_doc)
    income_page_count = income_doc.page_count
    income_doc.close()

    balance_doc = fitz.open(str(balance_path))
    dst.insert_pdf(balance_doc)
    balance_page_count = balance_doc.page_count
    balance_doc.close()

    buf = io.BytesIO()
    dst.save(buf)
    dst.close()

    income_positions  = list(range(1, income_page_count + 1))
    balance_positions = list(range(income_page_count + 1,
                                   income_page_count + balance_page_count + 1))

    return buf.getvalue(), income_positions, balance_positions


# ── Gemini helpers ─────────────────────────────────────────────────────────

def _gemini_config() -> types.GenerateContentConfig:
    """Shared generation config — thinking disabled, temperature 0.7."""
    return types.GenerateContentConfig(
        temperature=GEMINI_TEMPERATURE,
        top_p=GEMINI_TOP_P,
        max_output_tokens=GEMINI_MAX_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )


def _call1_chain_of_thought(
    client: genai.Client,
    pdf_bytes: bytes,
    income_pages: list,
    balance_pages: list,
) -> str:
    """
    Gemini Call 1: chain-of-thought over the combined PDF.

    Step 1 — Which line items are needed?
    Step 2 — Extract values and calculate ratios.
    """
    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

    prompt = f"""You are a financial analyst reviewing pages extracted from a company annual report.

The PDF attached contains:
  • Income statement page(s): {income_pages}
  • Balance sheet page(s):    {balance_pages}

(Page numbers refer to the position within the attached PDF, starting at 1.)

Work through the following two steps carefully.

---
STEP 1 — Required line items
List every line item you need from the income statement AND the balance sheet
to calculate each of these metrics:
  1. Net Income
  2. Cost-to-Income Ratio
  3. Quick Ratio
  4. Debt-to-Equity Ratio
  5. Debt-to-Assets Ratio
  6. Debt-to-Capital Ratio
  7. Debt-to-EBITDA Ratio
  8. Interest Coverage Ratio

---
STEP 2 — Extract values and calculate
For each line item identified in Step 1, read its value from the attached pages.
Then calculate every metric. Show the formula and the numbers used.
If a required line item is not present in the attached pages, state it explicitly.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[pdf_part, prompt],
        config=_gemini_config(),
    )
    return response.text


def _call2_json_output(client: genai.Client, cot_text: str, company_name: str) -> dict:
    """
    Gemini Call 2: convert the chain-of-thought analysis to structured JSON.
    """
    prompt = f"""Below is a financial analysis of {company_name}.

--- BEGIN ANALYSIS ---
{cot_text}
--- END ANALYSIS ---

Based on the analysis above, output ONLY a valid JSON object (no markdown, no
code fences, no extra text) with the following structure:

{{
  "<metric_key>": {{
    "value": <number or null>,
    "unit": "<e.g. millions USD, ratio, %>",
    "formula": "<formula used>",
    "inputs": {{"<line_item>": <value>, ...}},
    "reason": "<one sentence interpreting this result>"
  }},
  ...
}}

Use these exact metric keys:
  net_income, cost_to_income_ratio, quick_ratio, debt_to_equity_ratio,
  debt_to_assets_ratio, debt_to_capital_ratio, debt_to_ebitda_ratio,
  interest_coverage_ratio

Set value to null if the required data was not available in the source pages.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt],
        config=_gemini_config(),
    )

    raw = response.text.strip()
    # Strip markdown code fences if Gemini adds them despite the instruction
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    return json.loads(raw)


# ── Single-run analysis ────────────────────────────────────────────────────

def analyze_from_extracted(
    income_pdf_path: Path,
    balance_pdf_path: Path,
    pdf_bytes: bytes,
    income_positions: list,
    balance_positions: list,
    run_index: int,
) -> dict | None:
    """
    One Gemini run for a company using the pre-combined PDF bytes.

    Args:
        income_pdf_path:   Path to the income-statement PDF (used for naming).
        balance_pdf_path:  Path to the balance-sheet PDF (used for naming).
        pdf_bytes:         Combined in-memory PDF (income + balance pages).
        income_positions:  1-indexed page positions of income pages in combined PDF.
        balance_positions: 1-indexed page positions of balance pages in combined PDF.
        run_index:         1-based run number (for file naming and logging).

    Returns:
        Parsed result dict, or None if the run failed.
    """
    company_name = income_pdf_path.stem.replace("_income_statement", "")
    client = genai.Client(api_key=GEMINI_API_KEY)

    print(f"\n  ── Run {run_index:02d}/{N_RUNS} ──────────────────────────────────")

    # Call 1
    print(f"  Gemini Call 1 — chain-of-thought...")
    cot_text = _call1_chain_of_thought(
        client, pdf_bytes, income_positions, balance_positions
    )
    print(f"  Done.")

    # Call 2
    print(f"  Gemini Call 2 — JSON formatting...")
    result = _call2_json_output(client, cot_text, company_name)
    print(f"  Done.")

    # Save individual run JSON
    out_path = (income_pdf_path.parent
                / f"{company_name}_financial_ratios_run_{run_index:02d}.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved → {out_path.name}")

    return result


# ── Multi-run orchestration ────────────────────────────────────────────────

def run_multiple(
    income_pdf_path: Path,
    balance_pdf_path: Path,
    n_runs: int = N_RUNS,
) -> list:
    """
    Run the Gemini analysis n_runs times for one company.

    The combined PDF is built once and reused across all runs.

    Returns:
        List of result dicts (one per successful run).
    """
    company_name = income_pdf_path.stem.replace("_income_statement", "")

    print(f"\n{'='*70}")
    print(f"Company  : {company_name}")
    print(f"Income   : {income_pdf_path.name}")
    print(f"Balance  : {balance_pdf_path.name}")
    print(f"Runs     : {n_runs}  |  Temperature: {GEMINI_TEMPERATURE}")
    print(f"{'='*70}")

    # Combine PDFs once — reused for every run
    print("\nCombining extracted PDFs with PyMuPDF...")
    pdf_bytes, income_positions, balance_positions = _combine_pdfs(
        income_pdf_path, balance_pdf_path
    )
    print(f"  Combined PDF : {len(pdf_bytes):,} bytes  "
          f"(income pages {income_positions}, balance pages {balance_positions})")

    results = []
    for i in range(1, n_runs + 1):
        try:
            result = analyze_from_extracted(
                income_pdf_path, balance_pdf_path,
                pdf_bytes, income_positions, balance_positions,
                run_index=i,
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"  [!] Run {i:02d} failed: {e}")

    print(f"\n  Completed {len(results)}/{n_runs} runs successfully.")
    return results


# ── Consistency analysis ───────────────────────────────────────────────────

def _stats(values: list) -> dict:
    """Compute mean, std, min, max, cv for a list of numbers."""
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "cv": None}
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    cv = (std / abs(mean)) if mean != 0 else None
    return {
        "mean": round(mean, 4),
        "std":  round(std,  4),
        "min":  round(min(values), 4),
        "max":  round(max(values), 4),
        "cv":   round(cv, 6) if cv is not None else None,
    }


def check_consistency(results: list, company_name: str) -> dict:
    """
    Compute consistency statistics across all runs for each metric.

    Args:
        results:      List of result dicts from run_multiple().
        company_name: Used for labelling the report.

    Returns:
        Consistency report dict.
    """
    report = {
        "company":    company_name,
        "n_runs":     len(results),
        "temperature": GEMINI_TEMPERATURE,
        "metrics":    {},
    }

    for metric in METRICS:
        numeric_values = []
        null_count = 0

        for r in results:
            entry = r.get(metric, {})
            val = entry.get("value") if isinstance(entry, dict) else None
            if val is None:
                null_count += 1
            else:
                try:
                    numeric_values.append(float(val))
                except (TypeError, ValueError):
                    null_count += 1

        stats = _stats(numeric_values)
        consistent = (
            null_count == 0
            and stats["cv"] is not None
            and stats["cv"] < CONSISTENCY_CV_THRESHOLD
        )

        report["metrics"][metric] = {
            "values":     numeric_values,
            "null_count": null_count,
            "consistent": consistent,
            **stats,
        }

    return report


def save_consistency_report(report: dict, output_dir: Path) -> Path:
    """
    Save the consistency report as JSON and print a summary table.

    Returns the path of the saved file.
    """
    company_name = report["company"]
    out_path = output_dir / f"{company_name}_consistency_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    # ── Terminal summary table ─────────────────────────────────────────────
    n = report["n_runs"]
    print(f"\n{'='*70}")
    print(f"CONSISTENCY REPORT — {company_name}  "
          f"({n} runs, temperature={report['temperature']})")
    print(f"{'='*70}")
    print(f"  {'Metric':<28}  {'Mean':>14}  {'Std':>12}  {'CV':>8}  {'Nulls':>5}  Status")
    print(f"  {'─'*28}  {'─'*14}  {'─'*12}  {'─'*8}  {'─'*5}  {'─'*10}")

    for metric, m in report["metrics"].items():
        mean_str  = f"{m['mean']:.4f}"  if m["mean"]  is not None else "N/A"
        std_str   = f"{m['std']:.4f}"   if m["std"]   is not None else "N/A"
        cv_str    = f"{m['cv']:.4f}"    if m["cv"]    is not None else "N/A"
        status    = "✓ consistent" if m["consistent"] else "✗ variable"
        print(f"  {metric:<28}  {mean_str:>14}  {std_str:>12}  "
              f"{cv_str:>8}  {m['null_count']:>5}  {status}")

    print(f"\n  Saved → {out_path.name}")
    print(f"{'='*70}")

    return out_path


# ── Entry point ────────────────────────────────────────────────────────────

def _find_pairs(extracted_dir: Path) -> list:
    """Find all (income_pdf, balance_pdf) pairs in the extracted directory."""
    pairs = []
    for income_path in sorted(extracted_dir.glob("*_income_statement.pdf")):
        stem = income_path.stem.replace("_income_statement", "")
        balance_path = extracted_dir / f"{stem}_balance_sheet.pdf"
        if balance_path.exists():
            pairs.append((income_path, balance_path))
        else:
            print(f"[!] No balance sheet found for {stem} — skipped.")
    return pairs


if __name__ == "__main__":
    if len(sys.argv) > 1:
        stem = Path(sys.argv[1]).stem
        stem = stem.replace("_income_statement", "").replace("_balance_sheet", "")
        pairs = [
            (
                EXTRACTED_DIR / f"{stem}_income_statement.pdf",
                EXTRACTED_DIR / f"{stem}_balance_sheet.pdf",
            )
        ]
    else:
        pairs = _find_pairs(EXTRACTED_DIR)

    print(f"Found {len(pairs)} company pair(s) in {EXTRACTED_DIR}")

    for income_path, balance_path in pairs:
        if not income_path.exists():
            print(f"\n[!] Income PDF not found: {income_path} — skipped.")
            continue
        if not balance_path.exists():
            print(f"\n[!] Balance PDF not found: {balance_path} — skipped.")
            continue

        company_name = income_path.stem.replace("_income_statement", "")

        try:
            # Run N_RUNS times
            results = run_multiple(income_path, balance_path, n_runs=N_RUNS)

            if not results:
                print(f"\n[!] All runs failed for {company_name} — skipping report.")
                continue

            # Consistency analysis
            report = check_consistency(results, company_name)
            save_consistency_report(report, EXTRACTED_DIR)

        except Exception as e:
            print(f"\n✗ Error processing {company_name}: {e}")
            raise
