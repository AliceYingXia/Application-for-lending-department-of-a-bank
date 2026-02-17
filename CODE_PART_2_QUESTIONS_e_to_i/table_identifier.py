"""
Financial Statement Table Identification using pdfplumber
==========================================================
Identifies income statements and balance sheets in PDF documents.
Uses fuzzy matching to filter tables by title.

IMPORTANT: All page numbers are PHYSICAL POSITIONS (1-indexed),
NOT printed page numbers shown on the pages.
"""

import os
import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional
from rapidfuzz import fuzz  # For fuzzy string matching

# Disable PaddleOCR model source connectivity check (for offline/local models)
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'


class TableIdentifier:
    """Identify tables in PDFs using multiple detection methods."""

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

        debug = False

        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}")

            for page_idx, page in enumerate(pdf.pages):
                #if page_idx not in [124, 125, 126, 127, 128]:
                #    continue
                physical_page_num = page_idx + 1  # 1-indexed physical position

                # Extract tables from this page
                tables = page.extract_tables()

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

                # Apply fuzzy matching line by line to find best match
                best_match_line = None
                best_statement_type = None

                for line in top_5_lines:
                    # Skip if any line contains "note" or "notes"
                    if 'note' in line.lower():
                        continue
                    if "comment" in line.lower():
                        continue
                    if len(line.strip()) > 50:
                        continue
                    statement_type = self._fuzzy_match_financial_statement(line)
                    if statement_type:
                        best_match_line = line
                        best_statement_type = statement_type
                        break  # Use first matching line

                # Skip if no match found
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
                        min_rows = 10
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

                if balance_sheet_tables:
                    balance_tables_by_page[physical_page_num] = balance_sheet_tables

        # Print results
        self._print_pdfplumber_results( income_tables_by_page, balance_tables_by_page)

        return  income_tables_by_page, balance_tables_by_page

    def identify_pages_with_ocr(self) -> Dict[int, List[Dict]]:
        """
        Identify tables using PaddleOCR + PaddleStructure.
        Uses OCR for text extraction and PaddleStructure for table detection.

        Returns:
            Dict mapping physical page number to list of table info dicts
        """
        print("\n" + "="*70)
        print("TABLE IDENTIFICATION - PADDLEOCR METHOD")
        print("="*70)

        income_tables_by_page, balance_tables_by_page = {}, {}

        debug = False

        # Initialize PaddleOCR and PaddleStructure once
        try:
            from paddleocr import PPStructureV3, PaddleOCR
            import cv2
            import numpy as np

            # Initialize OCR for text extraction
            ocr = PaddleOCR(use_textline_orientation=True, lang='en')

            # Initialize structure analyzer for table detection
            table_engine = PPStructureV3(lang='en')

        except ImportError as e:
            print(f"✗ Error: PaddleOCR not installed.")
            print(f"  Install with: pip install paddlepaddle paddleocr")
            return {}, {}

        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}")

            for page_idx, page in enumerate(pdf.pages):
                physical_page_num = page_idx + 1

                # Convert page to image for PaddleOCR
                page_image = page.to_image(resolution=300)
                img_array = np.array(page_image.original)
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # Step 1: Extract text using PaddleOCR
                ocr_result = ocr.ocr(img_cv)

                if not ocr_result or not ocr_result[0]:
                    continue

                # Extract top 5 lines from OCR result
                all_text_lines = []
                for line in ocr_result[0]:
                    text = line[1][0]  # Extract text from OCR result
                    all_text_lines.append(text.strip())

                top_5_lines = all_text_lines[:5] if len(all_text_lines) >= 5 else all_text_lines

                if debug:
                    print(f"Page {page_idx}: Top 5 lines from PaddleOCR")
                    print(top_5_lines)

                # Apply fuzzy matching line by line
                best_match_line = None
                best_statement_type = None

                for line in top_5_lines:
                    if 'note' in line.lower() or "comment" in line.lower():
                        continue
                    if len(line.strip()) > 50:
                        continue
                    statement_type = self._fuzzy_match_financial_statement(line)
                    if statement_type:
                        best_match_line = line
                        best_statement_type = statement_type
                        break

                if not best_statement_type:
                    continue

                statement_type = best_statement_type

                if debug:
                    print(f"  Matched: {best_match_line} -> {statement_type}")

                # Step 2: Extract tables using PaddleStructure
                structure_result = table_engine(img_cv)

                # Filter for table regions
                tables_found = [item for item in structure_result if item['type'] == 'table']

                if not tables_found:
                    if debug:
                        print(f"  No tables found by PaddleStructure")
                    continue

                # Process each detected table
                income_statement_tables = []
                balance_sheet_tables = []

                for table_idx, table_item in enumerate(tables_found):
                    # Get table HTML or structure
                    table_html = table_item.get('res', {})

                    # Estimate table size from bounding box
                    bbox = table_item.get('bbox', [0, 0, 0, 0])
                    table_height = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
                    table_width = bbox[2] - bbox[0] if len(bbox) >= 4 else 0

                    # Rough estimation: assume average row height ~20 pixels
                    estimated_rows = int(table_height / 20) if table_height > 0 else 0
                    estimated_cols = len(table_html.get('cell_bbox', [])) if table_html else 0

                    if debug:
                        print(f"  Table {table_idx}: est. {estimated_rows} rows × {estimated_cols} cols")

                    # Apply size filter
                    if "continued" in best_match_line.lower():
                        min_rows = 3
                    else:
                        min_rows = 10
                    min_cols = 1

                    if estimated_rows < min_rows or estimated_cols < min_cols:
                        continue

                    table_info = {
                        'table_index': table_idx,
                        'num_rows': estimated_rows,
                        'num_cols': estimated_cols,
                        'title': best_match_line,
                        'statement_type': statement_type,
                        'bbox': bbox,
                        'extraction_method': 'paddleocr',
                        'preview': f"Table detected at bbox: {bbox}"
                    }

                    # Add to appropriate list based on type
                    if statement_type == 'income_statement':
                        income_statement_tables.append(table_info)
                    elif statement_type == 'balance_sheet':
                        balance_sheet_tables.append(table_info)

                if income_statement_tables:
                    income_tables_by_page[physical_page_num] = income_statement_tables

                if balance_sheet_tables:
                    balance_tables_by_page[physical_page_num] = balance_sheet_tables

        # Print results
        self._print_pdfplumber_results(income_tables_by_page, balance_tables_by_page)

        return income_tables_by_page, balance_tables_by_page

    def compare_methods(self) -> Dict:
        """
        Compare current method vs OCR method.
        Shows differences and performance metrics.
        """
        import time

        print("\n" + "="*80)
        print("COMPARISON: Current Method vs OCR Method")
        print("="*80 + "\n")

        # Run current method
        print("Running CURRENT METHOD (pdfplumber only)...")
        start_time = time.time()
        current_income, current_balance = self.identify_with_pdfplumber()
        current_time = time.time() - start_time

        # Extract page numbers from current method
        current_income_pages = sorted(current_income.keys())
        current_balance_pages = sorted(current_balance.keys())

        print(f"\n⏱️  Time taken: {current_time:.2f} seconds\n")

        # Run OCR method
        print("Running OCR METHOD (with OCR fallback)...")
        start_time = time.time()
        ocr_income, ocr_balance = self.identify_pages_with_ocr()
        ocr_time = time.time() - start_time

        ocr_income_pages = sorted(ocr_income.keys())
        ocr_balance_pages = sorted(ocr_balance.keys())

        print(f"\n⏱️  Time taken: {ocr_time:.2f} seconds\n")

        # Compare results
        print(f"\n{'='*80}")
        print("COMPARISON RESULTS")
        print(f"{'='*80}\n")

        print("📊 INCOME STATEMENTS:")
        print(f"   Current Method: {current_income_pages}")
        print(f"   OCR Method:     {ocr_income_pages}")

        # Find differences
        only_current_income = set(current_income_pages) - set(ocr_income_pages)
        only_ocr_income = set(ocr_income_pages) - set(current_income_pages)

        if only_current_income:
            print(f"   ⚠️  Only in Current: {sorted(only_current_income)}")
        if only_ocr_income:
            print(f"   ⚠️  Only in OCR: {sorted(only_ocr_income)}")
        if not only_current_income and not only_ocr_income:
            print(f"   ✓ Results match perfectly!")

        print("\n📊 BALANCE SHEETS:")
        print(f"   Current Method: {current_balance_pages}")
        print(f"   OCR Method:     {ocr_balance_pages}")

        only_current_balance = set(current_balance_pages) - set(ocr_balance_pages)
        only_ocr_balance = set(ocr_balance_pages) - set(current_balance_pages)

        if only_current_balance:
            print(f"   ⚠️  Only in Current: {sorted(only_current_balance)}")
        if only_ocr_balance:
            print(f"   ⚠️  Only in OCR: {sorted(only_ocr_balance)}")
        if not only_current_balance and not only_ocr_balance:
            print(f"   ✓ Results match perfectly!")

        print(f"\n⏱️  PERFORMANCE:")
        print(f"   Current Method: {current_time:.2f}s")
        print(f"   OCR Method:     {ocr_time:.2f}s")
        print(f"   Difference:     {abs(ocr_time - current_time):.2f}s ({'OCR slower' if ocr_time > current_time else 'OCR faster'})")

        print(f"\n{'='*80}\n")

        return {
            'current': {
                'income_pages': current_income_pages,
                'balance_pages': current_balance_pages,
                'income_tables': current_income,
                'balance_tables': current_balance,
                'time': current_time
            },
            'ocr': {
                'income_pages': ocr_income_pages,
                'balance_pages': ocr_balance_pages,
                'income_tables': ocr_income,
                'balance_tables': ocr_balance,
                'time': ocr_time
            },
            'differences': {
                'income_only_current': sorted(only_current_income),
                'income_only_ocr': sorted(only_ocr_income),
                'balance_only_current': sorted(only_current_balance),
                'balance_only_ocr': sorted(only_ocr_balance)
            }
        }

    # ── Helper methods ────────────────────────────────────────────────────

    def _fuzzy_match_financial_statement(self, title: str) -> Optional[str]:
        """
        Fuzzy match table title to financial statement type.

        Args:
            title: Table title text

        Returns:
            'income_statement', 'balance_sheet', or None if no match
        """
        if not title:
            return None

        title_lower = title.lower()

        # Define keywords for each statement type
        income_keywords = [
            'consolidated statements of income',
            'consolidated income statement',
            'consolidated statement of income',
            'consolidated income statements',
            'income statement',
            'statements of income',
            'statement of operations',
            'consolidated statements of operations',
            'consolidated statements of comprehensive Income'
        ]

        balance_keywords = [
            'consolidated balance sheet',
            'consolidated balance sheets',
            'balance sheet',
            'balance sheets',
            'consolidated statement of financial position',
            'statement of financial position'
        ]

        # Use partial_ratio for better substring matching
        # Threshold: 80 for good match
        threshold = 95

        # Check income statement
        # best_income_score = 0
        # best_income_keyword = None
        for keyword in income_keywords:
            score = fuzz.partial_ratio(title_lower, keyword)
            if score > threshold:
                # best_income_score = score
                # best_income_keyword = keyword
                return 'income_statement'

        # Check balance sheet
        # best_balance_score = 0
        # best_balance_keyword = None
        for keyword in balance_keywords:
            score = fuzz.partial_ratio(title_lower, keyword)
            if score > threshold:
                # best_balance_score = score
                # best_balance_keyword = keyword
                return 'balance_sheet'

        return None

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


# ── Usage Examples ────────────────────────────────────────────────────────

def example_basic_usage():
    """Example: Basic table identification with fuzzy matching - scans all pages."""
    pdf_path = Path("data/Tencent_2024.pdf")

    identifier = TableIdentifier(pdf_path)

    # Use pdfplumber with fuzzy matching for income statement and balance sheet
    identifier.identify_with_pdfplumber()


def example_compare_methods():
    """Example: Compare current method vs OCR method."""
    pdf_path = Path("data/Alibaba_2024.pdf")

    identifier = TableIdentifier(pdf_path)

    # Run comparison
    comparison_results = identifier.compare_methods()

    # Access specific results
    print("\n" + "="*80)
    print("ACCESSING COMPARISON RESULTS PROGRAMMATICALLY")
    print("="*80)

    print(f"\nCurrent method found:")
    print(f"  Income pages: {comparison_results['current']['income_pages']}")
    print(f"  Balance pages: {comparison_results['current']['balance_pages']}")
    print(f"  Time: {comparison_results['current']['time']:.2f}s")

    print(f"\nOCR method found:")
    print(f"  Income pages: {comparison_results['ocr']['income_pages']}")
    print(f"  Balance pages: {comparison_results['ocr']['balance_pages']}")
    print(f"  Time: {comparison_results['ocr']['time']:.2f}s")

    return comparison_results


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
            identifier = TableIdentifier(pdf_path)
            income_tables, balance_tables = identifier.identify_with_pdfplumber()

            # Count tables
            income_count = sum(len(tables) for tables in income_tables.values())
            balance_count = sum(len(tables) for tables in balance_tables.values())

            total_income_tables += income_count
            total_balance_tables += balance_count

            all_results[pdf_path.name] = {
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

    return all_results


def example_batch_comparison():
    """Example: Compare both methods across all PDFs in batch."""
    import time

    data_dir = Path("data")
    pdf_files = sorted(list(data_dir.glob("*.pdf")))

    print(f"\n{'='*80}")
    print(f"BATCH COMPARISON: {len(pdf_files)} PDF FILES")
    print(f"COMPARING CURRENT METHOD vs OCR METHOD")
    print(f"{'='*80}\n")

    all_results = {}
    total_current_time = 0
    total_ocr_time = 0
    ocr_helped_count = 0
    perfect_match_count = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n{'─'*80}")
        print(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        print(f"{'─'*80}")

        try:
            identifier = TableIdentifier(pdf_path)

            # Run current method
            print("\n  Running CURRENT METHOD...")
            start_time = time.time()
            current_income, current_balance = identifier.identify_with_pdfplumber()
            current_time = time.time() - start_time

            current_income_pages = sorted(current_income.keys())
            current_balance_pages = sorted(current_balance.keys())

            # Run OCR method
            print("\n  Running OCR METHOD...")
            start_time = time.time()
            ocr_income, ocr_balance = identifier.identify_pages_with_ocr()
            ocr_time = time.time() - start_time

            ocr_income_pages = sorted(ocr_income.keys())
            ocr_balance_pages = sorted(ocr_balance.keys())

            # Compare results
            income_match = set(current_income_pages) == set(ocr_income_pages)
            balance_match = set(current_balance_pages) == set(ocr_balance_pages)
            perfect_match = income_match and balance_match

            if perfect_match:
                perfect_match_count += 1

            # Check if OCR found more
            ocr_found_more = (len(ocr_income_pages) > len(current_income_pages) or
                             len(ocr_balance_pages) > len(current_balance_pages))

            if ocr_found_more:
                ocr_helped_count += 1

            total_current_time += current_time
            total_ocr_time += ocr_time

            # Store results
            all_results[pdf_path.name] = {
                'current': {
                    'income_pages': current_income_pages,
                    'balance_pages': current_balance_pages,
                    'income_count': sum(len(t) for t in current_income.values()),
                    'balance_count': sum(len(t) for t in current_balance.values()),
                    'time': current_time
                },
                'ocr': {
                    'income_pages': ocr_income_pages,
                    'balance_pages': ocr_balance_pages,
                    'income_count': sum(len(t) for t in ocr_income.values()),
                    'balance_count': sum(len(t) for t in ocr_balance.values()),
                    'time': ocr_time
                },
                'perfect_match': perfect_match,
                'ocr_helped': ocr_found_more,
                'status': 'success'
            }

            # Print quick summary for this PDF
            if perfect_match:
                print(f"\n  ✓ Results match perfectly!")
            else:
                print(f"\n  ⚠️  Results differ:")
                if not income_match:
                    print(f"     Income: Current={current_income_pages}, OCR={ocr_income_pages}")
                if not balance_match:
                    print(f"     Balance: Current={current_balance_pages}, OCR={ocr_balance_pages}")

            print(f"  ⏱️  Time: Current={current_time:.2f}s, OCR={ocr_time:.2f}s (Δ={abs(ocr_time-current_time):.2f}s)")

        except Exception as e:
            all_results[pdf_path.name] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"\n  ✗ Error: {e}")

    # Print comprehensive summary
    print(f"\n\n{'='*80}")
    print("BATCH COMPARISON SUMMARY")
    print(f"{'='*80}")

    successful_pdfs = [r for r in all_results.values() if r['status'] == 'success']

    print(f"\n📊 Overall Statistics:")
    print(f"   Total PDFs processed: {len(pdf_files)}")
    print(f"   Successful: {len(successful_pdfs)}")
    print(f"   Errors: {len(pdf_files) - len(successful_pdfs)}")
    print(f"   Perfect matches: {perfect_match_count}/{len(successful_pdfs)}")
    print(f"   OCR found more tables: {ocr_helped_count}/{len(successful_pdfs)}")

    print(f"\n⏱️  Performance Comparison:")
    print(f"   Total Current Method time: {total_current_time:.2f}s")
    print(f"   Total OCR Method time: {total_ocr_time:.2f}s")
    print(f"   Difference: {abs(total_ocr_time - total_current_time):.2f}s")
    print(f"   Average per PDF - Current: {total_current_time/len(successful_pdfs):.2f}s")
    print(f"   Average per PDF - OCR: {total_ocr_time/len(successful_pdfs):.2f}s")

    print(f"\n📁 Detailed Comparison by PDF:")
    print(f"{'─'*80}")

    for filename, result in sorted(all_results.items()):
        if result['status'] == 'success':
            print(f"\n{filename}")

            current = result['current']
            ocr = result['ocr']

            print(f"  Current Method:")
            print(f"    Income: {current['income_count']} table(s) on pages {current['income_pages']}")
            print(f"    Balance: {current['balance_count']} table(s) on pages {current['balance_pages']}")
            print(f"    Time: {current['time']:.2f}s")

            print(f"  OCR Method:")
            print(f"    Income: {ocr['income_count']} table(s) on pages {ocr['income_pages']}")
            print(f"    Balance: {ocr['balance_count']} table(s) on pages {ocr['balance_pages']}")
            print(f"    Time: {ocr['time']:.2f}s")

            if result['perfect_match']:
                print(f"  Status: ✓ Perfect match")
            else:
                print(f"  Status: ⚠️  Results differ")
                if result['ocr_helped']:
                    print(f"           OCR found more tables!")
        else:
            print(f"\n{filename}")
            print(f"  ✗ Error: {result['error']}")

    print(f"\n{'='*80}")
    print("Batch comparison complete!")
    print(f"{'='*80}\n")

    return all_results


def test_ocr_text_extraction(pdf_path: Path, page_number: int = 1):
    """
    Test PaddleOCR text extraction on a specific page.

    Args:
        pdf_path: Path to PDF file
        page_number: Physical page number (1-indexed) to test
    """
    print(f"\n{'='*70}")
    print(f"TESTING OCR TEXT EXTRACTION")
    print(f"{'='*70}")
    print(f"PDF: {pdf_path.name}")
    print(f"Page: {page_number}")

    try:
        from paddleocr import PaddleOCR
        import cv2
        import numpy as np

        # Initialize OCR
        ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        print("\n✓ PaddleOCR initialized")

        with pdfplumber.open(pdf_path) as pdf:
            if page_number < 1 or page_number > len(pdf.pages):
                print(f"\n✗ Error: Invalid page number. PDF has {len(pdf.pages)} pages.")
                return

            page = pdf.pages[page_number - 1]

            # Convert page to image
            page_image = page.to_image(resolution=300)
            img_array = np.array(page_image.original)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            print(f"✓ Converted page to image (resolution: 300 DPI)")

            # Run OCR
            print(f"\n🔍 Running PaddleOCR text extraction...")
            ocr_result = ocr.ocr(img_cv, cls=True)

            if not ocr_result or not ocr_result[0]:
                print("\n✗ No text detected on this page")
                return

            # Display all detected text
            print(f"\n📄 Detected Text (Total lines: {len(ocr_result[0])}):")
            print(f"{'─'*70}")

            for i, line in enumerate(ocr_result[0], 1):
                bbox = line[0]  # Bounding box coordinates
                text = line[1][0]  # Detected text
                confidence = line[1][1]  # Confidence score

                print(f"{i:3d}. {text}")
                print(f"      Confidence: {confidence:.4f}, BBox: {bbox[0]}")

            # Extract top 5 lines
            print(f"\n🔝 Top 5 Lines:")
            print(f"{'─'*70}")
            for i, line in enumerate(ocr_result[0][:5], 1):
                text = line[1][0]
                print(f"{i}. {text}")

    except ImportError:
        print(f"\n✗ Error: PaddleOCR not installed.")
        print(f"  Install with: pip install paddlepaddle paddleocr opencv-python numpy")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def test_ocr_table_detection(pdf_path: Path, page_number: int = 1):
    """
    Test PaddleStructure table detection on a specific page.

    Args:
        pdf_path: Path to PDF file
        page_number: Physical page number (1-indexed) to test
    """
    print(f"\n{'='*70}")
    print(f"TESTING OCR TABLE DETECTION")
    print(f"{'='*70}")
    print(f"PDF: {pdf_path.name}")
    print(f"Page: {page_number}")

    try:
        from paddleocr import PPStructureV3
        import cv2
        import numpy as np

        # Initialize table detection engine
        table_engine = PPStructureV3(lang='en')
        print("\n✓ PaddleStructure V3 initialized")

        with pdfplumber.open(pdf_path) as pdf:
            if page_number < 1 or page_number > len(pdf.pages):
                print(f"\n✗ Error: Invalid page number. PDF has {len(pdf.pages)} pages.")
                return

            page = pdf.pages[page_number - 1]

            # Convert page to image
            page_image = page.to_image(resolution=300)
            img_array = np.array(page_image.original)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            print(f"✓ Converted page to image (resolution: 300 DPI)")

            # Run table detection
            print(f"\n🔍 Running PaddleStructure table detection...")
            structure_result = table_engine(img_cv)

            # Find all detected elements
            tables_found = [item for item in structure_result if item['type'] == 'table']

            print(f"\n📊 Detection Results:")
            print(f"{'─'*70}")
            print(f"Total elements detected: {len(structure_result)}")
            print(f"Tables detected: {len(tables_found)}")

            # Display all detected elements
            if structure_result:
                print(f"\n🔍 All Detected Elements:")
                for i, item in enumerate(structure_result, 1):
                    elem_type = item.get('type', 'unknown')
                    bbox = item.get('bbox', [])
                    print(f"{i}. Type: {elem_type}, BBox: {bbox}")

            # Display table details
            if tables_found:
                print(f"\n📋 Table Details:")
                print(f"{'─'*70}")

                for i, table in enumerate(tables_found, 1):
                    bbox = table.get('bbox', [])

                    # Estimate dimensions
                    if bbox and len(bbox) == 4:
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]

                        print(f"\nTable {i}:")
                        print(f"  BBox: {bbox}")
                        print(f"  Width: {width:.1f}px, Height: {height:.1f}px")
                        print(f"  Position: ({bbox[0]:.1f}, {bbox[1]:.1f})")
            else:
                print("\n✗ No tables detected on this page")

    except ImportError:
        print(f"\n✗ Error: PaddleOCR/PaddleStructure not installed.")
        print(f"  Install with: pip install paddlepaddle paddleocr opencv-python numpy")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def example_ocr_only():
    """Example: Run OCR method only on a single PDF."""
    pdf_path = Path("data/JPM_2024.pdf")

    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        return

    print(f"\n{'='*70}")
    print(f"TESTING OCR METHOD ONLY")
    print(f"{'='*70}")

    identifier = TableIdentifier(pdf_path)

    # Run OCR method
    income_tables, balance_tables = identifier.identify_pages_with_ocr()

    # Print results summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    income_pages = sorted(income_tables.keys())
    balance_pages = sorted(balance_tables.keys())

    print(f"\nIncome Statements found on pages: {income_pages}")
    print(f"Balance Sheets found on pages: {balance_pages}")

    print(f"\nTotal Income Statements: {sum(len(tables) for tables in income_tables.values())}")
    print(f"Total Balance Sheets: {sum(len(tables) for tables in balance_tables.values())}")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run batch processing on all PDFs
    # example_basic_usage()
    # example_batch_processing()
    # example_compare_methods()

    # Test OCR on a specific page
    # test_ocr_text_extraction(Path("data/JPM_2024.pdf"), page_number=1)
    # test_ocr_table_detection(Path("data/JPM_2024.pdf"), page_number=90)

    # Test OCR method only
    example_ocr_only()
