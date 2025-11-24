# Part 2: LLM-Based Financial Forecasting

Compares **Gemini 2.5 Flash** against the traditional policy-based model from Part 1 on 50 S&P 500 companies, using sMAPE over a 1-year ahead forecast horizon (2024).

---

## Key Results

| Model | Mean sMAPE | Median sMAPE |
|---|---|---|
| Traditional (Part 1) | 36.53% | 35.89% |
| Gemini 2.5 Flash | 43.46% | 42.12% |
| **Ensemble (simple avg)** | **24.14%** | — |

- Gemini wins head-to-head in 52% of companies (26/50)
- No statistically significant difference: p = 0.2604 (paired t-test, α = 0.05)
- Ensemble reduces mean sMAPE by **33.9%** vs the best individual model

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API key
```bash
cp .env.example .env
# Edit .env and set: GEMINI_API_KEY=your_key_here
```

### 3. Verify
```bash
python3 config.py
```

> The `.env` file is excluded from Git. Never commit API keys.

---

## Pipeline

Run scripts in order:

| Script | Purpose | Output |
|---|---|---|
| `01_select_sample.py` | Stratified sample (top/bottom/random) | `data/test_sample.csv` |
| `02_extract_data.py` | Fetch financials from Yahoo Finance | `data/historical/`, `data/actual/` |
| `03_gemini_forecast.py` | LLM forecasting with rate limiting & retry | `forecasts/gemini/` |
| `04_evaluate.py` | sMAPE vs 2024 actuals | `results/gemini_evaluation.csv` |
| `05_compare.py` | Gemini vs Traditional (t-test, Wilcoxon) | `results/model_comparison.csv` |
| `06_ensemble.py` | Simple avg, weighted, median ensembles | `results/ensemble_results.csv` |
| `07_thinking_comparison_and_roobust_test.py` | Robustness & thinking-mode analysis | `results/thinking_comparison_*.csv` |

---

## Directory Structure

```
CODE_PART_2_QUESTIONS_a_to_f/
├── config.py                    # API config & directory paths
├── requirements.txt
├── .env                         # API key (not in Git)
├── .env.example                 # Template
├── 01_select_sample.py
├── 02_extract_data.py
├── 03_gemini_forecast.py
├── 04_evaluate.py
├── 05_compare.py
├── 06_ensemble.py
├── 07_thinking_comparison_and_roobust_test.py
├── test_part2.py                # Unit & integration tests (46 tests)
├── verify_setup.py
├── data/
│   ├── test_sample.csv          # 50 selected companies
│   ├── historical/              # 2020–2023 financials (JSON)
│   └── actual/                  # 2024 actuals (JSON)
├── forecasts/
│   └── gemini/                  # Gemini predictions (JSON)
└── results/
    ├── gemini_evaluation.csv
    ├── model_comparison.csv
    ├── ensemble_results.csv
    ├── ensemble_summary.csv
    ├── thinking_comparison_summary.csv
    ├── thinking_comparison_detailed.csv
    ├── test_results.txt         # Latest test run output
    └── part_2_questions_a_to_f.pdf
```

---

## Configuration

| Setting | Default | Description |
|---|---|---|
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model identifier |
| `GEMINI_TEMPERATURE` | `0.0` | 0 = deterministic |
| `TEST_SAMPLE_SIZE` | `50` | Companies to evaluate |
| `FORECAST_YEAR` | `2024` | Target forecast year |
| `MAX_API_BUDGET_USD` | `20.00` | Cost ceiling |

---

## Testing

```bash
python3 test_part2.py
```

**46/46 tests pass.** Covers: sMAPE correctness, CV calculation, ensemble methods, forecast schema validation, statistical tests, thinking-mode experiment validation, and result file integrity.

---

## Robustness Findings

| Configuration | Success Rate | CV (revenue) |
|---|---|---|
| Token limit 4096 | 60% | — |
| Token limit 8192 | 88.3% | < 0.5% |
| Temperature 0.0 | — | 0% (deterministic) |
| Temperature 0.7 | — | up to 846% (poor-fit stocks) |
| Thinking mode (T=0.0) | — | ~23% (non-deterministic) |

**Recommended production config:** `temperature=0.0`, `max_output_tokens=8192`, thinking mode off.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `GEMINI_API_KEY not set` | Create `.env` with your key |
| `Invalid API key` | Check key at Google AI Studio |
| Rate limit exceeded | Reduce `MAX_REQUESTS_PER_MINUTE` in `.env` |