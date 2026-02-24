"""
Financial Statement Table Identification using pdfplumber
==========================================================
Identifies income statements and balance sheets in PDF documents.
Uses fuzzy matching to filter tables by title.

IMPORTANT: All page numbers are PHYSICAL POSITIONS (1-indexed),
NOT printed page numbers shown on the pages.
"""

import csv
import re
from datetime import datetime
import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional
from rapidfuzz import fuzz  # For fuzzy string matching


class TableIdentifier:
    """Identify tables in PDFs using multiple detection methods."""

    _artifact_dict = None  # class-level Marker model cache; loaded once per process

    @classmethod
    def _load_marker_models(cls):
        if cls._artifact_dict is None:
            print('Loading Marker models (first run: ~2 GB download to ~/.cache)...')
            from marker.models import create_model_dict
            cls._artifact_dict = create_model_dict()
            print('Models ready.')
        return cls._artifact_dict

    def __init__(self, pdf_path: Path):
        """
        Initialize table identifier.

        Args:
            pdf_path: Path to PDF file
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    def identify_with_pdfplumber(self) -> Dict[int, List[Dict]]:
        """
        Identify tables using pdfplumber.

        pdfplumber is great for:
        - Text-based PDFs (not scanned images)
        - Simple to moderate table structures
        - Fast processing

        Returns:
            Dict mapping physical page number to list of table info dicts
        """
        print("\n" + "="*70)
        print("TABLE IDENTIFICATION - pdfplumber")
        print("="*70)

        income_tables_by_page, balance_tables_by_page = {}, {}
        income_page_scores,    balance_page_scores    = {}, {}

        debug = False

        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}")

            for page_idx, page in enumerate(pdf.pages):
                #if page_idx not in [124, 125, 126, 127, 128]:
                #    continue
                physical_page_num = page_idx + 1  # 1-indexed physical position

                # Use text-based detection for all PDFs.
                # This correctly captures both border-based (e.g. EM) and
                # borderless text-layout PDFs (e.g. JPM).
                tables = page.extract_tables(table_settings={
                    'vertical_strategy': 'text',
                    'horizontal_strategy': 'text',
                })

                if not tables:
                    continue

                # Extract top 5 lines of the page for fuzzy matching
                page_text = page.extract_text()
                if not page_text:
                    continue

                lines = [line.strip() for line in page_text.split('\n') if line.strip()]
                top_5_lines = lines[:5] if len(lines) >= 5 else lines

                if debug:
                    print("page_idx: ", page_idx)
                    print(top_5_lines)

                # Score all candidate lines and pick the one with the highest
                # fuzzy match score across all keywords.
                best_match_line = None
                best_statement_type = None
                best_score = 0

                for line in top_5_lines:
                    if 'note' in line.lower():
                        continue
                    if 'comment' in line.lower():
                        continue
                    if len(line.strip()) > 50:
                        continue
                    statement_type, score = self._fuzzy_match_financial_statement(line)
                    if statement_type and score > best_score:
                        best_match_line = line
                        best_statement_type = statement_type
                        best_score = score

                # Skip if no line exceeded the threshold
                if not best_statement_type:
                    continue

                statement_type = best_statement_type

                if debug:
                    print("best_match_line: ", best_match_line)
                    print("statement_type: ", statement_type)

                # Filter tables by size and keyword matching
                # Separate lists for income statements and balance sheets
                income_statement_tables = []
                balance_sheet_tables = []

                for table_idx, table in enumerate(tables):
                    if not table:
                        continue

                    num_rows = len(table)
                    num_cols = len(table[0]) if table else 0

                    if debug:
                        print("num_rows: ", num_rows, " num_cols: ", num_cols)

                    if "continued" in best_match_line.lower():
                        min_rows = 3
                    else:
                        min_rows = 3
                    min_cols: int = 1
                    if num_rows < min_rows or num_cols < min_cols:
                        continue

                    if debug:
                        print("Passed Column and Row Test")

                    table_info = {
                        'table_index': table_idx,
                        'num_rows': num_rows,
                        'num_cols': num_cols,
                        'title': best_match_line,  # Table name/title
                        'statement_type': statement_type,  # income_statement or balance_sheet
                        'preview': table[:3] if len(table) >= 3 else table,  # First 3 rows
                    }

                    # Add to appropriate list based on type
                    if statement_type == 'income_statement':
                        income_statement_tables.append(table_info)
                    elif statement_type == 'balance_sheet':
                        balance_sheet_tables.append(table_info)

                if income_statement_tables:
                    income_tables_by_page[physical_page_num] = income_statement_tables
                    income_page_scores[physical_page_num]    = best_score

                if balance_sheet_tables:
                    balance_tables_by_page[physical_page_num] = balance_sheet_tables
                    balance_page_scores[physical_page_num]    = best_score

        # Apply top-2 / unique-top-1 filter before printing
        income_tables_by_page  = self._filter_top_pages(income_tables_by_page,  income_page_scores)
        balance_tables_by_page = self._filter_top_pages(balance_tables_by_page, balance_page_scores)

        # Print results
        self._print_pdfplumber_results(income_tables_by_page, balance_tables_by_page)

        return income_tables_by_page, balance_tables_by_page

    # ── Helper methods ────────────────────────────────────────────────────

    def _filter_top_pages(self, pages_dict: Dict, scores_dict: Dict) -> Dict:
        """
        Reduce matched pages to at most 2, using page-level fuzzy scores.

        Rules (applied after sorting pages by score descending):
          - <= 2 pages  → keep all, no change.
          - >  2 pages  → keep top 2.
          - Special case: if the top-1 score is strictly higher than top-2
            AND every page from position 2 onward shares the same score
            → top-1 is a clear unique winner; keep only top-1.
        """
        if len(pages_dict) <= 2:
            return pages_dict

        # Sort pages by score descending
        ranked = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        scores = [s for _, s in ranked]

        # Check unique top-1: top score is strictly higher, and all others are tied
        if scores[0] > scores[1] and len(set(scores[1:])) == 1:
            keep = [ranked[0][0]]
        else:
            keep = [ranked[0][0], ranked[1][0]]

        return {pg: pages_dict[pg] for pg in keep}

    def _fuzzy_match_financial_statement(self, title: str) -> tuple:
        """
        Fuzzy match table title to financial statement type.

        Iterates over every keyword for every statement type and keeps the
        highest-scoring match.  Both the normal title and a space-stripped
        variant are scored against each keyword; the higher of the two is used.

        Args:
            title: Table title text

        Returns:
            (statement_type, best_score) where statement_type is
            'income_statement', 'balance_sheet', or None if no keyword
            exceeds the threshold.
        """
        if not title or len(title.strip()) < 10:
            return None, 0

        title_lower = title.lower()
        # Some PDFs (e.g. EM_2024) extract text with no spaces between words.
        # Normalise by stripping all whitespace for a fallback comparison.
        title_no_spaces = re.sub(r'\s+', '', title_lower)

        income_keywords = [
            'consolidated statements of income',
            'consolidated income statement',
            'consolidated statement of income',
            'consolidated income statements',
            'income statement',
            'statements of income',
            'statement of operations',
            'consolidated statements of operations',
            'consolidated statements of comprehensive income',
        ]

        balance_keywords = [
            'consolidated balance sheet',
            'consolidated balance sheets',
            'balance sheet',
            'balance sheets',
            'consolidated statement of financial position',
            'statement of financial position',
        ]

        threshold = 95

        best_score = 0
        best_type = None

        all_keywords = (
            [(kw, 'income_statement') for kw in income_keywords] +
            [(kw, 'balance_sheet')    for kw in balance_keywords]
        )

        for keyword, statement_type in all_keywords:
            kw_no_spaces = re.sub(r'\s+', '', keyword)
            score = max(
                fuzz.ratio(title_lower, keyword),
                fuzz.ratio(title_no_spaces, kw_no_spaces),
            )
            if score > best_score:
                best_score = score
                best_type = statement_type

        if best_score > threshold:
            return best_type, best_score

        return None, 0

    def _print_pdfplumber_results(self, income_tables_by_page: Dict[int, List[Dict]],
                                   balance_tables_by_page: Dict[int, List[Dict]]):
        """Print pdfplumber results in readable format."""
        if not income_tables_by_page and not balance_tables_by_page:
            print("✗ No tables found")
            return

        # Get all unique pages
        all_pages = sorted(set(list(income_tables_by_page.keys()) + list(balance_tables_by_page.keys())))

        print(f"✓ Found tables on {len(all_pages)} page(s)\n")
        print(f"  Income statements: {sum(len(tables) for tables in income_tables_by_page.values())} tables")
        print(f"  Balance sheets: {sum(len(tables) for tables in balance_tables_by_page.values())} tables\n")

        # Print income statements
        if income_tables_by_page:
            print("="*70)
            print("INCOME STATEMENTS")
            print("="*70)
            for page_num, tables in sorted(income_tables_by_page.items()):
                print(f"\n{'─'*70}")
                print(f"Physical Page {page_num}: {len(tables)} table(s)")
                print(f"{'─'*70}")

                for table_info in tables:
                    print(f"  Table #{table_info['table_index']}")
                    if table_info.get('title'):
                        print(f"    Title: \"{table_info['title']}\"")
                    print(f"    Dimensions: {table_info['num_rows']} rows × {table_info['num_cols']} cols")
                    if table_info['preview']:
                        print(f"    Preview (first {len(table_info['preview'])} rows):")
                        for row in table_info['preview']:
                            truncated_row = [str(cell)[:20] + '...' if cell and len(str(cell)) > 20
                                           else str(cell) for cell in row]
                            print(f"      {truncated_row}")
                    print()

        # Print balance sheets
        if balance_tables_by_page:
            print("\n" + "="*70)
            print("BALANCE SHEETS")
            print("="*70)
            for page_num, tables in sorted(balance_tables_by_page.items()):
                print(f"\n{'─'*70}")
                print(f"Physical Page {page_num}: {len(tables)} table(s)")
                print(f"{'─'*70}")

                for table_info in tables:
                    print(f"  Table #{table_info['table_index']}")
                    if table_info.get('title'):
                        print(f"    Title: \"{table_info['title']}\"")
                    print(f"    Dimensions: {table_info['num_rows']} rows × {table_info['num_cols']} cols")
                    if table_info['preview']:
                        print(f"    Preview (first {len(table_info['preview'])} rows):")
                        for row in table_info['preview']:
                            truncated_row = [str(cell)[:20] + '...' if cell and len(str(cell)) > 20
                                           else str(cell) for cell in row]
                            print(f"      {truncated_row}")
                    print()


# ── Results persistence ───────────────────────────────────────────────────

def save_results(results: dict, output_path: Path = Path("results.csv")) -> None:
    """
    Save batch processing results to a CSV file.

    Each row contains:
      filename               - PDF file name
      total_pages            - total physical pages in the PDF
      income_statement_pages - space-separated physical page numbers (1-indexed)
      balance_sheet_pages    - space-separated physical page numbers (1-indexed)
      status                 - 'success' or 'error'
      error                  - error message if status is 'error', else empty

    Args:
        results:     dict returned by example_batch_processing()
        output_path: destination CSV file (created or overwritten)
    """
    output_path = Path(output_path)
    fieldnames = [
        'filename',
        'total_pages',
        'income_statement_pages',
        'balance_sheet_pages',
        'status',
        'error',
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for filename, result in sorted(results.items()):
            if result['status'] == 'success':
                income_pages = sorted(result.get('income_tables', {}).keys())
                balance_pages = sorted(result.get('balance_tables', {}).keys())
                writer.writerow({
                    'filename': filename,
                    'total_pages': result.get('total_pages', ''),
                    'income_statement_pages': ' '.join(str(p) for p in income_pages),
                    'balance_sheet_pages': ' '.join(str(p) for p in balance_pages),
                    'status': 'success',
                    'error': '',
                })
            else:
                writer.writerow({
                    'filename': filename,
                    'total_pages': '',
                    'income_statement_pages': '',
                    'balance_sheet_pages': '',
                    'status': 'error',
                    'error': result.get('error', ''),
                })

    print(f"\nResults saved to: {output_path.resolve()}")


# ── Usage Examples ────────────────────────────────────────────────────────

def example_basic_usage():
    """Example: Basic table identification with fuzzy matching - scans all pages."""
    pdf_path = Path("data/Tencent_2024.pdf")

    identifier = TableIdentifier(pdf_path)

    # Use pdfplumber with fuzzy matching for income statement and balance sheet
    identifier.identify_with_pdfplumber()


def example_batch_processing():
    """Example: Process ALL PDFs in data folder with comprehensive summary."""
    data_dir = Path("data")
    pdf_files = sorted(list(data_dir.glob("*.pdf")))

    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING: {len(pdf_files)} PDF FILES")
    print(f"{'='*80}\n")

    all_results = {}
    total_income_tables = 0
    total_balance_tables = 0
    total_errors = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n{'─'*80}")
        print(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        print(f"{'─'*80}")

        try:
            # Get total page count before OCR
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)

            identifier = TableIdentifier(pdf_path)
            income_tables, balance_tables = identifier.identify_with_pdfplumber()

            # Count tables
            income_count = sum(len(tables) for tables in income_tables.values())
            balance_count = sum(len(tables) for tables in balance_tables.values())

            total_income_tables += income_count
            total_balance_tables += balance_count

            all_results[pdf_path.name] = {
                'total_pages': total_pages,
                'income_tables': income_tables,
                'balance_tables': balance_tables,
                'income_count': income_count,
                'balance_count': balance_count,
                'status': 'success'
            }

            print(f"\n✓ Success: Found {income_count} income statement(s) and {balance_count} balance sheet(s)")

        except Exception as e:
            total_errors += 1
            all_results[pdf_path.name] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"\n✗ Error: {e}")

    # Print comprehensive summary
    print(f"\n\n{'='*80}")
    print("OVERALL BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")

    print(f"\n📊 Processing Statistics:")
    print(f"   Total PDFs processed: {len(pdf_files)}")
    print(f"   Successful: {len(pdf_files) - total_errors}")
    print(f"   Errors: {total_errors}")

    print(f"\n📋 Tables Found:")
    print(f"   Total Income Statements: {total_income_tables}")
    print(f"   Total Balance Sheets: {total_balance_tables}")
    print(f"   Total Tables: {total_income_tables + total_balance_tables}")

    print(f"\n📁 Detailed Results by PDF:")
    print(f"{'─'*80}")

    for filename, result in sorted(all_results.items()):
        if result['status'] == 'success':
            income_pages = sorted(result['income_tables'].keys()) if result['income_tables'] else []
            balance_pages = sorted(result['balance_tables'].keys()) if result['balance_tables'] else []

            print(f"\n✓ {filename}")
            print(f"   Income Statements: {result['income_count']} table(s) on page(s) {income_pages}")
            print(f"   Balance Sheets: {result['balance_count']} table(s) on page(s) {balance_pages}")
        else:
            print(f"\n✗ {filename}")
            print(f"   Error: {result['error']}")

    print(f"\n{'='*80}")
    print("Batch processing complete!")
    print(f"{'='*80}\n")

    # Save results to CSV
    output_path = Path("data") / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_results(all_results, output_path)

    return all_results


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run batch processing on all PDFs
    # example_basic_usage()
    example_batch_processing()
    # example_compare_methods()
