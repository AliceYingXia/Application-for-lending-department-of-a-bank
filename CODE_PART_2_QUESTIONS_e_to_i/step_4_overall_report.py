"""
Overall Consistency Report
===========================
Reads all *_consistency_report.json files from data/extracted/ and
produces a cross-company summary showing how consistently Gemini
extracts each financial metric across 9 companies x 10 runs.

Output
------
  data/extracted/overall_consistency_report.json
  data/extracted/overall_consistency_report.csv
"""

import csv
import json
from pathlib import Path

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

METRIC_LABELS = {
    "net_income":               "Net Income",
    "cost_to_income_ratio":     "Cost-to-Income",
    "quick_ratio":              "Quick Ratio",
    "debt_to_equity_ratio":     "Debt/Equity",
    "debt_to_assets_ratio":     "Debt/Assets",
    "debt_to_capital_ratio":    "Debt/Capital",
    "debt_to_ebitda_ratio":     "Debt/EBITDA",
    "interest_coverage_ratio":  "Interest Coverage",
}


def load_all_reports(extracted_dir: Path) -> list:
    reports = []
    for path in sorted(extracted_dir.glob("*_consistency_report.json")):
        reports.append(json.loads(path.read_text()))
    return reports


def cell_label(m: dict) -> str:
    """One-cell summary for a metric entry: consistent/variable/null."""
    if m["null_count"] == 10:
        return "all null"
    if m["consistent"]:
        cv = m["cv"]
        return f"✓  cv={cv:.4f}" if cv is not None else "✓  cv=0"
    cv = m["cv"]
    nulls = m["null_count"]
    if cv is not None:
        return f"✗  cv={cv:.4f}" + (f"  ({nulls} null)" if nulls else "")
    return f"✗  ({nulls}/10 null)"


def build_report(reports: list) -> dict:
    companies = [r["company"] for r in reports]

    # ── Per-cell data ──────────────────────────────────────────────────────
    grid = {}  # grid[company][metric] = cell dict
    for r in reports:
        grid[r["company"]] = {}
        for metric in METRICS:
            m = r["metrics"].get(metric, {})
            grid[r["company"]][metric] = {
                "mean":       m.get("mean"),
                "cv":         m.get("cv"),
                "null_count": m.get("null_count", 10),
                "consistent": m.get("consistent", False),
            }

    # ── Per-metric summary (across companies) ─────────────────────────────
    metric_summary = {}
    for metric in METRICS:
        consistent_companies = [
            c for c in companies if grid[c][metric]["consistent"]
        ]
        all_null_companies = [
            c for c in companies if grid[c][metric]["null_count"] == 10
        ]
        cv_values = [
            grid[c][metric]["cv"]
            for c in companies
            if grid[c][metric]["cv"] is not None
        ]
        metric_summary[metric] = {
            "consistent_count": len(consistent_companies),
            "consistent_pct":   round(len(consistent_companies) / len(companies) * 100, 1),
            "all_null_count":   len(all_null_companies),
            "median_cv":        round(sorted(cv_values)[len(cv_values)//2], 6)
                                if cv_values else None,
        }

    # ── Per-company summary ────────────────────────────────────────────────
    company_summary = {}
    for r in reports:
        c = r["company"]
        consistent_metrics = [
            metric for metric in METRICS if grid[c][metric]["consistent"]
        ]
        null_metrics = [
            metric for metric in METRICS if grid[c][metric]["null_count"] == 10
        ]
        company_summary[c] = {
            "consistent_count": len(consistent_metrics),
            "consistent_pct":   round(len(consistent_metrics) / len(METRICS) * 100, 1),
            "all_null_count":   len(null_metrics),
            "consistent_metrics": consistent_metrics,
            "all_null_metrics":   null_metrics,
        }

    return {
        "n_companies": len(companies),
        "n_runs_each": reports[0]["n_runs"] if reports else 0,
        "temperature": reports[0]["temperature"] if reports else None,
        "companies":   companies,
        "grid":        grid,
        "metric_summary":  metric_summary,
        "company_summary": company_summary,
    }


def print_report(report: dict):
    companies = report["companies"]
    n = report["n_companies"]

    SEP  = "=" * 78
    SEP2 = "─" * 78

    print(f"\n{SEP}")
    print(f"OVERALL CONSISTENCY REPORT")
    print(f"{n} companies  |  {report['n_runs_each']} runs each  |  "
          f"temperature={report['temperature']}")
    print(SEP)

    # ── 1. Company × Metric grid ───────────────────────────────────────────
    print(f"\n{'─'*78}")
    print("1. CONSISTENCY GRID  (✓ = CV < 5 %  |  ✗ = variable  |  — = all null)")
    print(f"{'─'*78}")

    col_w = 20
    label_w = 20
    header = f"  {'Metric':<{label_w}}" + "".join(f"  {c[:col_w]:<{col_w}}" for c in companies)
    print(header)
    print("  " + "─" * (label_w + (col_w + 2) * n))

    for metric in METRICS:
        row = f"  {METRIC_LABELS[metric]:<{label_w}}"
        for c in companies:
            m = report["grid"][c][metric]
            if m["null_count"] == 10:
                cell = "—"
            elif m["consistent"]:
                cell = f"✓ {m['cv']:.3f}" if m["cv"] is not None else "✓ 0"
            else:
                cell = f"✗ {m['cv']:.3f}" if m["cv"] is not None else f"✗ ({m['null_count']}null)"
            row += f"  {cell:<{col_w}}"
        print(row)

    # ── 2. Per-metric summary ──────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("2. PER-METRIC SUMMARY  (how many of 9 companies are consistent)")
    print(SEP2)
    print(f"  {'Metric':<22}  {'Consistent':>10}  {'% Companies':>11}  "
          f"{'All-null cos':>12}  {'Median CV':>10}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*11}  {'─'*12}  {'─'*10}")
    for metric in METRICS:
        ms = report["metric_summary"][metric]
        cv_s = f"{ms['median_cv']:.4f}" if ms["median_cv"] is not None else "N/A"
        print(f"  {METRIC_LABELS[metric]:<22}  {ms['consistent_count']:>10}  "
              f"{ms['consistent_pct']:>10.1f}%  {ms['all_null_count']:>12}  {cv_s:>10}")

    # ── 3. Per-company summary ─────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("3. PER-COMPANY SUMMARY  (how many of 8 metrics are consistent)")
    print(SEP2)
    print(f"  {'Company':<25}  {'Consistent':>10}  {'% Metrics':>10}  "
          f"{'All-null':>8}  Inconsistent metrics")
    print(f"  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*30}")
    for c in companies:
        cs = report["company_summary"][c]
        bad = [METRIC_LABELS[m] for m in METRICS
               if not report["grid"][c][m]["consistent"]
               and report["grid"][c][m]["null_count"] < 10]
        bad_s = ", ".join(bad) if bad else "—"
        print(f"  {c:<25}  {cs['consistent_count']:>10}  "
              f"{cs['consistent_pct']:>9.1f}%  {cs['all_null_count']:>8}  {bad_s}")

    # ── 4. Overall headline ────────────────────────────────────────────────
    total_cells = n * len(METRICS)
    consistent_cells = sum(
        report["grid"][c][m]["consistent"]
        for c in companies for m in METRICS
    )
    null_cells = sum(
        1 for c in companies for m in METRICS
        if report["grid"][c][m]["null_count"] == 10
    )
    print(f"\n{SEP}")
    print("4. HEADLINE")
    print(SEP)
    print(f"  Total metric×company cells : {total_cells}")
    print(f"  Consistently extracted     : {consistent_cells}  "
          f"({consistent_cells/total_cells*100:.1f}%)")
    print(f"  All-null (not calculable)  : {null_cells}  "
          f"({null_cells/total_cells*100:.1f}%)")
    print(f"  Variable (extracted, noisy): "
          f"{total_cells - consistent_cells - null_cells}  "
          f"({(total_cells-consistent_cells-null_cells)/total_cells*100:.1f}%)")
    print(SEP)


def save_csv(report: dict, out_path: Path):
    companies = report["companies"]
    rows = []
    for metric in METRICS:
        for c in companies:
            m = report["grid"][c][metric]
            rows.append({
                "company":    c,
                "metric":     metric,
                "mean":       m["mean"],
                "cv":         m["cv"],
                "null_count": m["null_count"],
                "consistent": m["consistent"],
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["company", "metric", "mean",
                                               "cv", "null_count", "consistent"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    reports = load_all_reports(EXTRACTED_DIR)
    if not reports:
        print("No consistency reports found in", EXTRACTED_DIR)
        raise SystemExit(1)

    print(f"Loaded {len(reports)} consistency report(s).")
    report = build_report(reports)

    print_report(report)

    # Save JSON
    json_out = EXTRACTED_DIR / "overall_consistency_report.json"
    json_out.write_text(json.dumps(report, indent=2))
    print(f"\nJSON saved → {json_out}")

    # Save CSV
    csv_out = EXTRACTED_DIR / "overall_consistency_report.csv"
    save_csv(report, csv_out)
    print(f"CSV  saved → {csv_out}")
