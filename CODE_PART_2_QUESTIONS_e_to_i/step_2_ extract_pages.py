"""
PDF Page Extractor
==================
Reads physical page numbers from a results CSV produced by
step_1_table_identifier.py and extracts the income-statement and
balance-sheet pages for every company into separate PDF files
using PyMuPDF.

Output layout
-------------
  data/extracted/
    Alibaba_2024_income_statement.pdf
    Alibaba_2024_balance_sheet.pdf
    EM_2024_income_statement.pdf
    EM_2024_balance_sheet.pdf
    ...

Usage
-----
    python step_2_ extract_pages.py                                    # uses default CSV
    python step_2_ extract_pages.py data/results_20260302_202254.csv  # explicit CSV path
"""

import csv
import io
import sys
from pathlib import Path

import fitz  # PyMuPDF


# ── Core extraction ────────────────────────────────────────────────────────

def extract_pages_as_pdf(pdf_path: Path, page_numbers: list) -> bytes:
    """
    Extract specific physical pages (1-indexed) from a PDF using PyMuPDF
    and return them as in-memory PDF bytes.

    Args:
        pdf_path:     Path to the source PDF.
        page_numbers: List of 1-indexed physical page numbers to extract.

    Returns:
        PDF bytes containing only the requested pages.
    """
    src = fitz.open(str(pdf_path))
    dst = fitz.open()

    for phys_num in sorted(page_numbers):
        page_idx = phys_num - 1
        if page_idx < 0 or page_idx >= src.page_count:
            print(f"    [!] Page {phys_num} out of range ({src.page_count} pages) — skipped.")
            continue
        dst.insert_pdf(src, from_page=page_idx, to_page=page_idx)

    buf = io.BytesIO()
    dst.save(buf)
    dst.close()
    src.close()
    return buf.getvalue()


def save_extracted_pages(
    pdf_path: Path,
    income_pages: list,
    balance_pages: list,
    output_dir: Path,
) -> tuple:
    """
    Extract and save income-statement and balance-sheet pages for one PDF.

    Args:
        pdf_path:      Path to the source PDF.
        income_pages:  List of 1-indexed income-statement page numbers.
        balance_pages: List of 1-indexed balance-sheet page numbers.
        output_dir:    Directory where extracted PDFs are saved.

    Returns:
        (income_out_path, balance_out_path) — paths of the saved files,
        or None for a type if no pages were available.
    """
    stem = pdf_path.stem
    income_out = balance_out = None

    if income_pages:
        pdf_bytes = extract_pages_as_pdf(pdf_path, income_pages)
        income_out = output_dir / f"{stem}_income_statement.pdf"
        income_out.write_bytes(pdf_bytes)
        print(f"    Income statement  ({income_pages}) → {income_out.name}  "
              f"({len(pdf_bytes):,} bytes)")
    else:
        print(f"    Income statement  — no pages found, skipped.")

    if balance_pages:
        pdf_bytes = extract_pages_as_pdf(pdf_path, balance_pages)
        balance_out = output_dir / f"{stem}_balance_sheet.pdf"
        balance_out.write_bytes(pdf_bytes)
        print(f"    Balance sheet     ({balance_pages}) → {balance_out.name}  "
              f"({len(pdf_bytes):,} bytes)")
    else:
        print(f"    Balance sheet     — no pages found, skipped.")

    return income_out, balance_out


# ── CSV reader ─────────────────────────────────────────────────────────────

def load_results_csv(csv_path: Path) -> list:
    """
    Parse a results CSV produced by table_identifier.save_results().

    Returns a list of dicts with keys:
        filename, income_pages (list[int]), balance_pages (list[int])
    Only rows with status == 'success' and at least one page are included.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "success":
                continue
            income_pages = [int(p) for p in row["income_statement_pages"].split() if p.strip()]
            balance_pages = [int(p) for p in row["balance_sheet_pages"].split() if p.strip()]
            if not income_pages and not balance_pages:
                continue
            rows.append({
                "filename":      row["filename"],
                "income_pages":  income_pages,
                "balance_pages": balance_pages,
            })
    return rows


# ── Main ───────────────────────────────────────────────────────────────────

def main(csv_path: Path, data_dir: Path):
    output_dir = data_dir / "extracted"
    output_dir.mkdir(exist_ok=True)

    entries = load_results_csv(csv_path)

    print(f"\nCSV      : {csv_path}")
    print(f"Data dir : {data_dir.resolve()}")
    print(f"Output   : {output_dir.resolve()}")
    print(f"Entries  : {len(entries)} PDF(s)\n")
    print("=" * 70)

    ok, failed = 0, 0

    for entry in entries:
        pdf_path = data_dir / entry["filename"]
        print(f"\n[{ok + failed + 1}/{len(entries)}] {entry['filename']}")

        if not pdf_path.exists():
            print(f"    [!] File not found — skipped.")
            failed += 1
            continue

        try:
            save_extracted_pages(
                pdf_path,
                entry["income_pages"],
                entry["balance_pages"],
                output_dir,
            )
            ok += 1
        except Exception as e:
            print(f"    [!] Error: {e}")
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"Done — {ok} succeeded, {failed} failed.")
    print(f"Extracted PDFs saved to: {output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    data_dir = Path("data")

    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        # Default: pick the most recent results CSV in data/
        candidates = sorted(data_dir.glob("results_*.csv"), reverse=True)
        if not candidates:
            print("No results_*.csv found in data/. Run step_1_table_identifier.py first.")
            sys.exit(1)
        csv_path = candidates[0]

    main(csv_path, data_dir)
