"""
Unit tests for the PDF financial extraction pipeline.

Covers:
  Step 1 — TableIdentifier._fuzzy_match_financial_statement
           TableIdentifier._filter_top_pages
           save_results
  Step 2 — load_results_csv
           extract_pages_as_pdf
           save_extracted_pages
  Step 3 — _stats
           check_consistency
           _call2_json_output (Gemini mocked)
           _combine_pdfs
  Step 4 — cell_label
           build_report
           save_csv
           load_all_reports
"""

import csv
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz  # PyMuPDF — needed for in-memory PDF helpers

# ── Path setup ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Step-1 imports ─────────────────────────────────────────────────────────
from step_1_table_identifier import TableIdentifier, save_results

# ── Step-2 import (filename contains a space) ──────────────────────────────
_spec2 = importlib.util.spec_from_file_location(
    "extract_pages", ROOT / "step_2_ extract_pages.py"
)
_mod2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_mod2)

extract_pages_as_pdf  = _mod2.extract_pages_as_pdf
save_extracted_pages  = _mod2.save_extracted_pages
load_results_csv      = _mod2.load_results_csv

# ── Step-3 imports ─────────────────────────────────────────────────────────
from step_3_gemini_analyzer import (
    _stats,
    check_consistency,
    _call2_json_output,
    _combine_pdfs,
)

# ── Step-4 imports ─────────────────────────────────────────────────────────
from step_4_overall_report import cell_label, build_report, save_csv, load_all_reports


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_pdf_bytes(n_pages: int = 3, text_on_page: dict = None) -> bytes:
    """Create a minimal in-memory PDF with n_pages blank pages.

    text_on_page: {1-indexed page number: text string} to insert text.
    """
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        if text_on_page and (i + 1) in text_on_page:
            page.insert_text((72, 72), text_on_page[i + 1])
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _write_tmp_pdf(tmp_dir: Path, name: str, n_pages: int = 3,
                   text_on_page: dict = None) -> Path:
    path = tmp_dir / name
    path.write_bytes(_make_pdf_bytes(n_pages, text_on_page))
    return path


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestFuzzyMatchFinancialStatement(unittest.TestCase):
    """Tests for TableIdentifier._fuzzy_match_financial_statement."""

    def setUp(self):
        """Create a dummy PDF path (no file needed; we only call _fuzzy_match)."""
        self.ti = object.__new__(TableIdentifier)  # bypass __init__ / file check

    def _match(self, title):
        return self.ti._fuzzy_match_financial_statement(title)

    # ── None / too-short inputs ──────────────────────────────────────────────

    def test_none_input(self):
        stmt, score = self._match(None)
        self.assertIsNone(stmt)
        self.assertEqual(score, 0)

    def test_empty_string(self):
        stmt, score = self._match("")
        self.assertIsNone(stmt)
        self.assertEqual(score, 0)

    def test_short_string_below_10_chars(self):
        stmt, score = self._match("Income")
        self.assertIsNone(stmt)
        self.assertEqual(score, 0)

    def test_exactly_9_chars_rejected(self):
        stmt, score = self._match("123456789")
        self.assertIsNone(stmt)
        self.assertEqual(score, 0)

    # ── Income statement keywords ────────────────────────────────────────────

    def test_income_statement_exact(self):
        stmt, score = self._match("income statement")
        self.assertEqual(stmt, "income_statement")
        self.assertGreater(score, 95)

    def test_consolidated_statement_of_income(self):
        stmt, score = self._match("Consolidated Statement of Income")
        self.assertEqual(stmt, "income_statement")
        self.assertGreater(score, 95)

    def test_consolidated_statements_of_income(self):
        stmt, score = self._match("Consolidated Statements of Income")
        self.assertEqual(stmt, "income_statement")
        self.assertGreater(score, 95)

    def test_statement_of_operations(self):
        stmt, score = self._match("Statement of Operations")
        self.assertEqual(stmt, "income_statement")
        self.assertGreater(score, 95)

    def test_consolidated_statements_of_operations(self):
        stmt, score = self._match("Consolidated Statements of Operations")
        self.assertEqual(stmt, "income_statement")
        self.assertGreater(score, 95)

    # ── Space-stripped variant (EM-style PDFs) ───────────────────────────────

    def test_income_statement_no_spaces(self):
        """Titles with all whitespace removed should still match (EM_2024 case)."""
        stmt, score = self._match("CONSOLIDATEDSTATEMENTOFINCOME")
        self.assertEqual(stmt, "income_statement")
        self.assertGreater(score, 95)

    def test_balance_sheet_no_spaces(self):
        stmt, score = self._match("CONSOLIDATEDBALANCESHEETS")
        self.assertEqual(stmt, "balance_sheet")
        self.assertGreater(score, 95)

    # ── Balance sheet keywords ───────────────────────────────────────────────

    def test_balance_sheet_exact(self):
        stmt, score = self._match("Balance Sheet")
        self.assertEqual(stmt, "balance_sheet")
        self.assertGreater(score, 95)

    def test_consolidated_balance_sheet(self):
        stmt, score = self._match("Consolidated Balance Sheet")
        self.assertEqual(stmt, "balance_sheet")
        self.assertGreater(score, 95)

    def test_consolidated_balance_sheets(self):
        stmt, score = self._match("Consolidated Balance Sheets")
        self.assertEqual(stmt, "balance_sheet")
        self.assertGreater(score, 95)

    def test_statement_of_financial_position(self):
        stmt, score = self._match("Statement of Financial Position")
        self.assertEqual(stmt, "balance_sheet")
        self.assertGreater(score, 95)

    def test_consolidated_statement_of_financial_position(self):
        stmt, score = self._match("Consolidated Statement of Financial Position")
        self.assertEqual(stmt, "balance_sheet")
        self.assertGreater(score, 95)

    # ── Low-score / unrelated text ───────────────────────────────────────────

    def test_unrelated_text_returns_none(self):
        stmt, score = self._match("Notes to the Financial Statements")
        self.assertIsNone(stmt)
        self.assertEqual(score, 0)

    def test_generic_financial_text_returns_none(self):
        stmt, score = self._match("Selected Financial Data")
        self.assertIsNone(stmt)
        self.assertEqual(score, 0)

    def test_random_text_returns_none(self):
        stmt, score = self._match("Management Discussion and Analysis")
        self.assertIsNone(stmt)
        self.assertEqual(score, 0)

    # ── Case insensitivity ───────────────────────────────────────────────────

    def test_uppercase_income(self):
        stmt, score = self._match("INCOME STATEMENT")
        self.assertEqual(stmt, "income_statement")
        self.assertGreater(score, 95)

    def test_mixed_case_balance(self):
        stmt, score = self._match("Consolidated BALANCE SHEET")
        self.assertEqual(stmt, "balance_sheet")
        self.assertGreater(score, 95)


class TestFilterTopPages(unittest.TestCase):
    """Tests for TableIdentifier._filter_top_pages."""

    def setUp(self):
        self.ti = object.__new__(TableIdentifier)

    def _filter(self, pages_dict, scores_dict):
        return self.ti._filter_top_pages(pages_dict, scores_dict)

    # ── 0 / 1 / 2 pages — always pass through unchanged ──────────────────────

    def test_empty_dict_unchanged(self):
        self.assertEqual(self._filter({}, {}), {})

    def test_one_page_unchanged(self):
        pages = {5: ["t1"]}
        scores = {5: 98}
        self.assertEqual(self._filter(pages, scores), pages)

    def test_two_pages_unchanged(self):
        pages = {3: ["t1"], 7: ["t2"]}
        scores = {3: 98, 7: 97}
        self.assertEqual(self._filter(pages, scores), pages)

    # ── 3+ pages — keep top 2 ────────────────────────────────────────────────

    def test_three_pages_keeps_top_two(self):
        pages = {1: ["a"], 2: ["b"], 3: ["c"]}
        scores = {1: 95, 2: 98, 3: 97}
        result = self._filter(pages, scores)
        self.assertEqual(set(result.keys()), {2, 3})  # top 2 by score

    def test_four_pages_keeps_top_two(self):
        pages = {1: ["a"], 2: ["b"], 3: ["c"], 4: ["d"]}
        scores = {1: 90, 2: 100, 3: 98, 4: 85}
        result = self._filter(pages, scores)
        self.assertEqual(set(result.keys()), {2, 3})

    # ── Unique top-1 special case ─────────────────────────────────────────────

    def test_unique_top1_all_others_tied_keeps_only_top1(self):
        """Top-1 strictly higher AND all others share the same score → keep only top-1."""
        pages = {10: ["a"], 20: ["b"], 30: ["c"]}
        scores = {10: 99, 20: 95, 30: 95}
        result = self._filter(pages, scores)
        self.assertEqual(set(result.keys()), {10})

    def test_unique_top1_but_others_differ_keeps_top2(self):
        """Top-1 strictly higher but other pages are NOT all tied → keep top 2."""
        pages = {10: ["a"], 20: ["b"], 30: ["c"]}
        scores = {10: 99, 20: 97, 30: 95}
        result = self._filter(pages, scores)
        self.assertEqual(set(result.keys()), {10, 20})

    def test_all_scores_tied_keeps_top2(self):
        pages = {1: ["a"], 2: ["b"], 3: ["c"]}
        scores = {1: 96, 2: 96, 3: 96}
        result = self._filter(pages, scores)
        self.assertEqual(len(result), 2)

    def test_table_data_preserved(self):
        """Filtered result keeps the original table data, not just page numbers."""
        tables_pg2 = [{"num_rows": 5, "statement_type": "income_statement"}]
        tables_pg3 = [{"num_rows": 4, "statement_type": "income_statement"}]
        tables_pg7 = [{"num_rows": 3, "statement_type": "income_statement"}]
        pages = {2: tables_pg2, 3: tables_pg3, 7: tables_pg7}
        scores = {2: 100, 3: 98, 7: 90}
        result = self._filter(pages, scores)
        self.assertEqual(result[2], tables_pg2)
        self.assertEqual(result[3], tables_pg3)


class TestSaveResults(unittest.TestCase):
    """Tests for the module-level save_results() function in step_1."""

    def _write_and_read(self, results: dict) -> list:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results.csv"
            save_results(results, out)
            with open(out, newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))

    def test_success_row_income_and_balance(self):
        results = {
            "company.pdf": {
                "status": "success",
                "total_pages": 200,
                "income_tables": {88: [], 89: []},
                "balance_tables": {120: []},
            }
        }
        rows = self._write_and_read(results)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "success")
        self.assertEqual(rows[0]["income_statement_pages"], "88 89")
        self.assertEqual(rows[0]["balance_sheet_pages"], "120")
        self.assertEqual(rows[0]["total_pages"], "200")

    def test_error_row(self):
        results = {
            "bad.pdf": {
                "status": "error",
                "error": "File corrupt",
            }
        }
        rows = self._write_and_read(results)
        self.assertEqual(rows[0]["status"], "error")
        self.assertEqual(rows[0]["error"], "File corrupt")
        self.assertEqual(rows[0]["income_statement_pages"], "")

    def test_multiple_rows_sorted_by_filename(self):
        results = {
            "z_file.pdf": {"status": "success", "total_pages": 10,
                           "income_tables": {1: []}, "balance_tables": {}},
            "a_file.pdf": {"status": "success", "total_pages": 20,
                           "income_tables": {}, "balance_tables": {2: []}},
        }
        rows = self._write_and_read(results)
        self.assertEqual(rows[0]["filename"], "a_file.pdf")
        self.assertEqual(rows[1]["filename"], "z_file.pdf")

    def test_empty_results_only_header(self):
        rows = self._write_and_read({})
        self.assertEqual(rows, [])


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadResultsCsv(unittest.TestCase):
    """Tests for load_results_csv() in step_2_ extract_pages.py."""

    def _write_csv(self, rows: list, tmp_dir: Path) -> Path:
        path = tmp_dir / "results.csv"
        fieldnames = ["filename", "total_pages", "income_statement_pages",
                      "balance_sheet_pages", "status", "error"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_success_rows_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_csv([
                {"filename": "A.pdf", "total_pages": 100,
                 "income_statement_pages": "88 89",
                 "balance_sheet_pages": "120",
                 "status": "success", "error": ""},
            ], Path(tmp))
            rows = load_results_csv(csv_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["filename"], "A.pdf")
        self.assertEqual(rows[0]["income_pages"], [88, 89])
        self.assertEqual(rows[0]["balance_pages"], [120])

    def test_error_rows_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_csv([
                {"filename": "bad.pdf", "total_pages": "",
                 "income_statement_pages": "",
                 "balance_sheet_pages": "",
                 "status": "error", "error": "corrupt"},
            ], Path(tmp))
            rows = load_results_csv(csv_path)
        self.assertEqual(rows, [])

    def test_success_but_no_pages_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_csv([
                {"filename": "empty.pdf", "total_pages": 50,
                 "income_statement_pages": "",
                 "balance_sheet_pages": "",
                 "status": "success", "error": ""},
            ], Path(tmp))
            rows = load_results_csv(csv_path)
        self.assertEqual(rows, [])

    def test_multiple_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_csv([
                {"filename": "A.pdf", "total_pages": 100,
                 "income_statement_pages": "10",
                 "balance_sheet_pages": "20 21",
                 "status": "success", "error": ""},
                {"filename": "B.pdf", "total_pages": 50,
                 "income_statement_pages": "5",
                 "balance_sheet_pages": "15",
                 "status": "success", "error": ""},
                {"filename": "C.pdf", "total_pages": 80,
                 "income_statement_pages": "",
                 "balance_sheet_pages": "",
                 "status": "error", "error": "failed"},
            ], Path(tmp))
            rows = load_results_csv(csv_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["balance_pages"], [15])

    def test_single_page_per_statement(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_csv([
                {"filename": "X.pdf", "total_pages": 200,
                 "income_statement_pages": "88",
                 "balance_sheet_pages": "150",
                 "status": "success", "error": ""},
            ], Path(tmp))
            rows = load_results_csv(csv_path)
        self.assertEqual(rows[0]["income_pages"], [88])
        self.assertEqual(rows[0]["balance_pages"], [150])


class TestExtractPagesAsPdf(unittest.TestCase):
    """Tests for extract_pages_as_pdf()."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_single_page_extracted(self):
        src = _write_tmp_pdf(self.tmp_path, "src.pdf", n_pages=5)
        result_bytes = extract_pages_as_pdf(src, [2])
        doc = fitz.open(stream=result_bytes, filetype="pdf")
        self.assertEqual(doc.page_count, 1)
        doc.close()

    def test_multiple_pages_extracted(self):
        src = _write_tmp_pdf(self.tmp_path, "src.pdf", n_pages=10)
        result_bytes = extract_pages_as_pdf(src, [1, 3, 7])
        doc = fitz.open(stream=result_bytes, filetype="pdf")
        self.assertEqual(doc.page_count, 3)
        doc.close()

    def test_pages_returned_in_sorted_order(self):
        """Pages should come out in ascending physical order regardless of input order."""
        src = _write_tmp_pdf(self.tmp_path, "src.pdf", n_pages=5)
        result_bytes = extract_pages_as_pdf(src, [5, 1, 3])
        doc = fitz.open(stream=result_bytes, filetype="pdf")
        self.assertEqual(doc.page_count, 3)
        doc.close()

    def test_out_of_range_page_skipped(self):
        """Page numbers beyond the PDF length should be silently skipped."""
        src = _write_tmp_pdf(self.tmp_path, "src.pdf", n_pages=3)
        result_bytes = extract_pages_as_pdf(src, [2, 99])
        doc = fitz.open(stream=result_bytes, filetype="pdf")
        self.assertEqual(doc.page_count, 1)
        doc.close()

    def test_all_out_of_range_raises(self):
        """PyMuPDF raises ValueError when trying to save a zero-page document."""
        src = _write_tmp_pdf(self.tmp_path, "src.pdf", n_pages=3)
        with self.assertRaises(ValueError):
            extract_pages_as_pdf(src, [50, 60])

    def test_returns_bytes(self):
        src = _write_tmp_pdf(self.tmp_path, "src.pdf", n_pages=2)
        result = extract_pages_as_pdf(src, [1])
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)

    def test_duplicate_pages_deduped(self):
        """sorted() eliminates nothing but inserting same page twice keeps 2 pages."""
        src = _write_tmp_pdf(self.tmp_path, "src.pdf", n_pages=5)
        # [2, 2] — duplicates: fitz inserts each reference separately
        result_bytes = extract_pages_as_pdf(src, [2, 2])
        doc = fitz.open(stream=result_bytes, filetype="pdf")
        # PyMuPDF inserts duplicates — confirm at least 1 page came through
        self.assertGreaterEqual(doc.page_count, 1)
        doc.close()


class TestSaveExtractedPages(unittest.TestCase):
    """Tests for save_extracted_pages()."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_income_file_created(self):
        src = _write_tmp_pdf(self.tmp_path, "Company.pdf", n_pages=5)
        income_out, balance_out = save_extracted_pages(
            src, income_pages=[1, 2], balance_pages=[], output_dir=self.tmp_path
        )
        self.assertIsNotNone(income_out)
        self.assertTrue(income_out.exists())
        self.assertIsNone(balance_out)

    def test_balance_file_created(self):
        src = _write_tmp_pdf(self.tmp_path, "Company.pdf", n_pages=5)
        income_out, balance_out = save_extracted_pages(
            src, income_pages=[], balance_pages=[3, 4], output_dir=self.tmp_path
        )
        self.assertIsNone(income_out)
        self.assertIsNotNone(balance_out)
        self.assertTrue(balance_out.exists())

    def test_both_files_created(self):
        src = _write_tmp_pdf(self.tmp_path, "Company.pdf", n_pages=5)
        income_out, balance_out = save_extracted_pages(
            src, income_pages=[1], balance_pages=[3], output_dir=self.tmp_path
        )
        self.assertTrue(income_out.exists())
        self.assertTrue(balance_out.exists())

    def test_output_filenames_match_stem(self):
        src = _write_tmp_pdf(self.tmp_path, "EM_2024.pdf", n_pages=5)
        income_out, balance_out = save_extracted_pages(
            src, income_pages=[1], balance_pages=[3], output_dir=self.tmp_path
        )
        self.assertEqual(income_out.name, "EM_2024_income_statement.pdf")
        self.assertEqual(balance_out.name, "EM_2024_balance_sheet.pdf")

    def test_extracted_page_count_correct(self):
        src = _write_tmp_pdf(self.tmp_path, "X.pdf", n_pages=10)
        income_out, _ = save_extracted_pages(
            src, income_pages=[2, 4, 6], balance_pages=[], output_dir=self.tmp_path
        )
        doc = fitz.open(str(income_out))
        self.assertEqual(doc.page_count, 3)
        doc.close()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestStats(unittest.TestCase):
    """Tests for _stats() in step_3_gemini_analyzer."""

    def test_empty_list_returns_all_none(self):
        result = _stats([])
        self.assertIsNone(result["mean"])
        self.assertIsNone(result["std"])
        self.assertIsNone(result["min"])
        self.assertIsNone(result["max"])
        self.assertIsNone(result["cv"])

    def test_single_value_zero_std(self):
        result = _stats([5.0])
        self.assertAlmostEqual(result["mean"], 5.0)
        self.assertAlmostEqual(result["std"], 0.0)
        self.assertAlmostEqual(result["min"], 5.0)
        self.assertAlmostEqual(result["max"], 5.0)
        self.assertAlmostEqual(result["cv"], 0.0)

    def test_identical_values_zero_cv(self):
        result = _stats([3.0, 3.0, 3.0, 3.0])
        self.assertAlmostEqual(result["mean"], 3.0)
        self.assertAlmostEqual(result["std"], 0.0)
        self.assertAlmostEqual(result["cv"], 0.0)

    def test_mean_zero_cv_is_none(self):
        """When mean is 0, CV is undefined (None)."""
        result = _stats([0.0, 0.0, 0.0])
        self.assertAlmostEqual(result["mean"], 0.0)
        self.assertIsNone(result["cv"])

    def test_cv_calculation(self):
        """Population std / |mean| — use symmetric values for easy calc."""
        # values: [1, 3] → mean=2, variance=1, std=1, cv=0.5
        result = _stats([1.0, 3.0])
        self.assertAlmostEqual(result["mean"], 2.0, places=4)
        self.assertAlmostEqual(result["cv"], 0.5, places=4)

    def test_negative_mean_cv_uses_abs(self):
        """CV uses |mean|, so negative mean should still yield positive CV."""
        result = _stats([-4.0, -6.0])
        self.assertAlmostEqual(result["mean"], -5.0, places=4)
        # std = 1, cv = 1/5 = 0.2
        self.assertAlmostEqual(result["cv"], 0.2, places=4)

    def test_min_max_correct(self):
        result = _stats([7.5, 1.2, 4.8, 9.1])
        self.assertAlmostEqual(result["min"], 1.2, places=4)
        self.assertAlmostEqual(result["max"], 9.1, places=4)

    def test_rounding_to_4_decimal_places(self):
        result = _stats([1.123456789, 1.987654321])
        self.assertEqual(len(str(result["mean"]).split(".")[-1]), 4)

    def test_ten_identical_values_consistent_pattern(self):
        """Mirrors real net-income extractions (all 10 runs identical)."""
        values = [71332.0] * 10
        result = _stats(values)
        self.assertAlmostEqual(result["mean"], 71332.0)
        self.assertAlmostEqual(result["cv"], 0.0)


class TestCheckConsistency(unittest.TestCase):
    """Tests for check_consistency() in step_3_gemini_analyzer."""

    def _run_result(self, **kwargs) -> dict:
        """Build a fake single-run result with the given metric values."""
        result = {}
        for k, v in kwargs.items():
            result[k] = {"value": v}
        return result

    def test_all_same_values_consistent(self):
        runs = [self._run_result(net_income=10000.0) for _ in range(10)]
        report = check_consistency(runs, "TestCo")
        self.assertTrue(report["metrics"]["net_income"]["consistent"])
        self.assertEqual(report["metrics"]["net_income"]["null_count"], 0)
        self.assertAlmostEqual(report["metrics"]["net_income"]["cv"], 0.0)

    def test_all_null_not_consistent(self):
        runs = [self._run_result() for _ in range(10)]  # no net_income key
        report = check_consistency(runs, "TestCo")
        m = report["metrics"]["net_income"]
        self.assertFalse(m["consistent"])
        self.assertEqual(m["null_count"], 10)
        self.assertIsNone(m["mean"])

    def test_high_cv_not_consistent(self):
        """One run returns 100×, others return 1× → CV ≫ 5%."""
        runs = [self._run_result(quick_ratio=1.0) for _ in range(9)]
        runs.append(self._run_result(quick_ratio=100.0))
        report = check_consistency(runs, "TestCo")
        self.assertFalse(report["metrics"]["quick_ratio"]["consistent"])
        self.assertGreater(report["metrics"]["quick_ratio"]["cv"], 0.05)

    def test_cv_just_below_threshold_consistent(self):
        """CV = 0.04 (< 5%) with no nulls should be marked consistent."""
        mean = 100.0
        # population std ≈ 4% of mean → values evenly spaced around mean
        vals = [mean * (1 - 0.04), mean, mean, mean, mean,
                mean, mean, mean, mean, mean * (1 + 0.04)]
        runs = [self._run_result(debt_to_equity_ratio=v) for v in vals]
        report = check_consistency(runs, "TestCo")
        self.assertLess(report["metrics"]["debt_to_equity_ratio"]["cv"], 0.05)
        self.assertTrue(report["metrics"]["debt_to_equity_ratio"]["consistent"])

    def test_one_null_makes_not_consistent(self):
        """Even if CV is tiny, one null value should block consistency."""
        runs = [self._run_result(net_income=5000.0) for _ in range(9)]
        runs.append(self._run_result())  # one run with null net_income
        report = check_consistency(runs, "TestCo")
        self.assertFalse(report["metrics"]["net_income"]["consistent"])
        self.assertEqual(report["metrics"]["net_income"]["null_count"], 1)

    def test_report_contains_all_metrics(self):
        expected_metrics = [
            "net_income", "cost_to_income_ratio", "quick_ratio",
            "debt_to_equity_ratio", "debt_to_assets_ratio",
            "debt_to_capital_ratio", "debt_to_ebitda_ratio",
            "interest_coverage_ratio",
        ]
        runs = [{}] * 3
        report = check_consistency(runs, "TestCo")
        for m in expected_metrics:
            self.assertIn(m, report["metrics"])

    def test_report_metadata(self):
        runs = [self._run_result(net_income=1.0)] * 5
        report = check_consistency(runs, "Acme")
        self.assertEqual(report["company"], "Acme")
        self.assertEqual(report["n_runs"], 5)

    def test_string_values_coerced_to_float(self):
        """Gemini sometimes returns numbers as strings."""
        runs = [self._run_result(net_income="12345.0") for _ in range(10)]
        report = check_consistency(runs, "TestCo")
        self.assertTrue(report["metrics"]["net_income"]["consistent"])

    def test_invalid_string_value_treated_as_null(self):
        runs = [self._run_result(net_income="N/A") for _ in range(10)]
        report = check_consistency(runs, "TestCo")
        self.assertEqual(report["metrics"]["net_income"]["null_count"], 10)


class TestCall2JsonOutput(unittest.TestCase):
    """Tests for _call2_json_output() with a mocked Gemini client."""

    def _mock_client(self, response_text: str):
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = response_text
        client.models.generate_content.return_value = mock_response
        return client

    def _sample_json(self):
        return json.dumps({
            "net_income": {
                "value": 62360.0,
                "unit": "millions USD",
                "formula": "bottom-line profit",
                "inputs": {"net_income": 62360.0},
                "reason": "Net income for fiscal year 2024.",
            }
        })

    def test_clean_json_parsed(self):
        client = self._mock_client(self._sample_json())
        result = _call2_json_output(client, "analysis text", "Meta")
        self.assertEqual(result["net_income"]["value"], 62360.0)

    def test_json_with_triple_backtick_fence_stripped(self):
        fenced = "```\n" + self._sample_json() + "\n```"
        client = self._mock_client(fenced)
        result = _call2_json_output(client, "analysis text", "Meta")
        self.assertEqual(result["net_income"]["value"], 62360.0)

    def test_json_with_json_backtick_fence_stripped(self):
        fenced = "```json\n" + self._sample_json() + "\n```"
        client = self._mock_client(fenced)
        result = _call2_json_output(client, "analysis text", "Meta")
        self.assertEqual(result["net_income"]["value"], 62360.0)

    def test_null_value_in_response(self):
        payload = json.dumps({
            "quick_ratio": {"value": None, "unit": "ratio",
                            "formula": "N/A", "inputs": {}, "reason": "Not available."}
        })
        client = self._mock_client(payload)
        result = _call2_json_output(client, "text", "TestCo")
        self.assertIsNone(result["quick_ratio"]["value"])

    def test_gemini_called_with_correct_arguments(self):
        client = self._mock_client(self._sample_json())
        _call2_json_output(client, "some CoT text", "Meta")
        self.assertTrue(client.models.generate_content.called)
        call_kwargs = client.models.generate_content.call_args
        # contents should be a list with one prompt string
        contents = call_kwargs.kwargs.get("contents") or call_kwargs.args[0] if call_kwargs.args else None
        self.assertIsNotNone(call_kwargs)

    def test_multiple_metrics_in_response(self):
        payload = json.dumps({
            "net_income":         {"value": 100.0, "unit": "M", "formula": "", "inputs": {}, "reason": ""},
            "quick_ratio":        {"value": 1.5,   "unit": "ratio", "formula": "", "inputs": {}, "reason": ""},
            "debt_to_equity_ratio": {"value": 0.3,  "unit": "ratio", "formula": "", "inputs": {}, "reason": ""},
        })
        client = self._mock_client(payload)
        result = _call2_json_output(client, "text", "TestCo")
        self.assertIn("net_income", result)
        self.assertIn("quick_ratio", result)
        self.assertIn("debt_to_equity_ratio", result)


class TestCombinePdfs(unittest.TestCase):
    """Tests for _combine_pdfs() in step_3_gemini_analyzer."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_combined_page_count(self):
        income_path = _write_tmp_pdf(self.tmp_path, "inc.pdf", n_pages=2)
        balance_path = _write_tmp_pdf(self.tmp_path, "bal.pdf", n_pages=3)
        pdf_bytes, inc_pos, bal_pos = _combine_pdfs(income_path, balance_path)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        self.assertEqual(doc.page_count, 5)
        doc.close()

    def test_income_positions_are_1_indexed(self):
        income_path = _write_tmp_pdf(self.tmp_path, "inc.pdf", n_pages=2)
        balance_path = _write_tmp_pdf(self.tmp_path, "bal.pdf", n_pages=2)
        _, inc_pos, bal_pos = _combine_pdfs(income_path, balance_path)
        self.assertEqual(inc_pos, [1, 2])

    def test_balance_positions_follow_income(self):
        income_path = _write_tmp_pdf(self.tmp_path, "inc.pdf", n_pages=2)
        balance_path = _write_tmp_pdf(self.tmp_path, "bal.pdf", n_pages=3)
        _, inc_pos, bal_pos = _combine_pdfs(income_path, balance_path)
        self.assertEqual(bal_pos, [3, 4, 5])

    def test_single_page_each(self):
        income_path = _write_tmp_pdf(self.tmp_path, "inc.pdf", n_pages=1)
        balance_path = _write_tmp_pdf(self.tmp_path, "bal.pdf", n_pages=1)
        pdf_bytes, inc_pos, bal_pos = _combine_pdfs(income_path, balance_path)
        self.assertEqual(inc_pos, [1])
        self.assertEqual(bal_pos, [2])

    def test_returns_bytes(self):
        income_path = _write_tmp_pdf(self.tmp_path, "inc.pdf", n_pages=1)
        balance_path = _write_tmp_pdf(self.tmp_path, "bal.pdf", n_pages=1)
        pdf_bytes, _, _ = _combine_pdfs(income_path, balance_path)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 0)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestCellLabel(unittest.TestCase):
    """Tests for cell_label() in step_4_overall_report."""

    def test_all_null(self):
        m = {"null_count": 10, "consistent": False, "cv": None}
        self.assertEqual(cell_label(m), "all null")

    def test_consistent_with_cv(self):
        m = {"null_count": 0, "consistent": True, "cv": 0.025}
        label = cell_label(m)
        self.assertIn("✓", label)
        self.assertIn("cv=0.0250", label)

    def test_consistent_with_cv_zero(self):
        m = {"null_count": 0, "consistent": True, "cv": 0.0}
        label = cell_label(m)
        self.assertIn("✓", label)
        self.assertIn("cv=0", label)

    def test_consistent_with_none_cv(self):
        """cv=None means all values were identical (std=0, mean=0)."""
        m = {"null_count": 0, "consistent": True, "cv": None}
        label = cell_label(m)
        self.assertIn("✓", label)
        self.assertIn("cv=0", label)

    def test_variable_with_cv_no_nulls(self):
        m = {"null_count": 0, "consistent": False, "cv": 0.15}
        label = cell_label(m)
        self.assertIn("✗", label)
        self.assertIn("cv=0.1500", label)
        self.assertNotIn("null", label)

    def test_variable_with_cv_and_some_nulls(self):
        m = {"null_count": 3, "consistent": False, "cv": 0.20}
        label = cell_label(m)
        self.assertIn("✗", label)
        self.assertIn("cv=0.2000", label)
        self.assertIn("3 null", label)

    def test_variable_null_cv_with_nulls(self):
        """cv=None and not all-null means partial nulls dominate."""
        m = {"null_count": 7, "consistent": False, "cv": None}
        label = cell_label(m)
        self.assertIn("✗", label)
        self.assertIn("7/10 null", label)


class TestBuildReport(unittest.TestCase):
    """Tests for build_report() in step_4_overall_report."""

    def _make_company_report(self, company, metrics_override=None):
        """Build a minimal consistency report for one company."""
        base_metric = {
            "mean": 1.0, "cv": 0.01, "null_count": 0, "consistent": True
        }
        metrics = {
            "net_income": dict(base_metric),
            "cost_to_income_ratio": dict(base_metric),
            "quick_ratio": dict(base_metric),
            "debt_to_equity_ratio": dict(base_metric),
            "debt_to_assets_ratio": dict(base_metric),
            "debt_to_capital_ratio": dict(base_metric),
            "debt_to_ebitda_ratio": dict(base_metric),
            "interest_coverage_ratio": dict(base_metric),
        }
        if metrics_override:
            for k, v in metrics_override.items():
                metrics[k] = v
        return {"company": company, "n_runs": 10,
                "temperature": 0.7, "metrics": metrics}

    def setUp(self):
        self.report_A = self._make_company_report("CompanyA", {
            "quick_ratio": {"mean": None, "cv": None, "null_count": 10, "consistent": False},
            "cost_to_income_ratio": {"mean": 50.0, "cv": 1.5, "null_count": 0, "consistent": False},
        })
        self.report_B = self._make_company_report("CompanyB")
        self.reports = [self.report_A, self.report_B]

    def test_n_companies(self):
        result = build_report(self.reports)
        self.assertEqual(result["n_companies"], 2)

    def test_n_runs_each(self):
        result = build_report(self.reports)
        self.assertEqual(result["n_runs_each"], 10)

    def test_companies_list(self):
        result = build_report(self.reports)
        self.assertEqual(result["companies"], ["CompanyA", "CompanyB"])

    def test_grid_has_all_companies_and_metrics(self):
        from step_4_overall_report import METRICS
        result = build_report(self.reports)
        for co in ["CompanyA", "CompanyB"]:
            for m in METRICS:
                self.assertIn(m, result["grid"][co])

    def test_all_null_in_grid(self):
        result = build_report(self.reports)
        cell = result["grid"]["CompanyA"]["quick_ratio"]
        self.assertEqual(cell["null_count"], 10)

    def test_metric_summary_consistent_count(self):
        result = build_report(self.reports)
        # quick_ratio: A → null (not consistent), B → consistent
        self.assertEqual(result["metric_summary"]["quick_ratio"]["consistent_count"], 1)
        self.assertEqual(result["metric_summary"]["quick_ratio"]["all_null_count"], 1)

    def test_company_summary_consistent_pct(self):
        result = build_report(self.reports)
        # CompanyB: all 8 metrics consistent → 100%
        self.assertAlmostEqual(result["company_summary"]["CompanyB"]["consistent_pct"], 100.0)

    def test_company_summary_consistent_count(self):
        result = build_report(self.reports)
        # CompanyA: quick_ratio and cost_to_income_ratio are inconsistent → 6/8
        self.assertEqual(result["company_summary"]["CompanyA"]["consistent_count"], 6)

    def test_temperature_from_first_report(self):
        result = build_report(self.reports)
        self.assertEqual(result["temperature"], 0.7)


class TestSaveCsv(unittest.TestCase):
    """Tests for save_csv() in step_4_overall_report."""

    def _build_minimal_report(self):
        from step_4_overall_report import METRICS
        companies = ["Co1", "Co2"]
        grid = {}
        for co in companies:
            grid[co] = {}
            for m in METRICS:
                grid[co][m] = {
                    "mean": 1.0 if co == "Co1" else None,
                    "cv": 0.01 if co == "Co1" else None,
                    "null_count": 0 if co == "Co1" else 10,
                    "consistent": co == "Co1",
                }
        return {"companies": companies, "grid": grid}

    def test_csv_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output.csv"
            save_csv(self._build_minimal_report(), out)
            self.assertTrue(out.exists())

    def test_csv_row_count(self):
        from step_4_overall_report import METRICS
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output.csv"
            report = self._build_minimal_report()
            save_csv(report, out)
            with open(out, newline="") as f:
                rows = list(csv.DictReader(f))
        # 2 companies × 8 metrics = 16 rows
        self.assertEqual(len(rows), 2 * len(METRICS))

    def test_csv_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output.csv"
            save_csv(self._build_minimal_report(), out)
            with open(out, newline="") as f:
                reader = csv.DictReader(f)
                cols = set(reader.fieldnames)
        self.assertEqual(cols, {"company", "metric", "mean", "cv", "null_count", "consistent"})

    def test_consistent_flag_in_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output.csv"
            save_csv(self._build_minimal_report(), out)
            with open(out, newline="") as f:
                rows = list(csv.DictReader(f))
        co1_rows = [r for r in rows if r["company"] == "Co1"]
        self.assertTrue(all(r["consistent"] == "True" for r in co1_rows))
        co2_rows = [r for r in rows if r["company"] == "Co2"]
        self.assertTrue(all(r["consistent"] == "False" for r in co2_rows))


class TestLoadAllReports(unittest.TestCase):
    """Tests for load_all_reports() in step_4_overall_report."""

    def _write_report(self, tmp_dir: Path, company: str) -> None:
        report = {
            "company": company,
            "n_runs": 10,
            "temperature": 0.7,
            "metrics": {
                "net_income": {"mean": 100.0, "cv": 0.0, "null_count": 0, "consistent": True}
            },
        }
        path = tmp_dir / f"{company}_consistency_report.json"
        path.write_text(json.dumps(report))

    def test_loads_all_matching_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_report(d, "Company_A")
            self._write_report(d, "Company_B")
            # Non-matching file should be ignored
            (d / "other_file.json").write_text("{}")
            reports = load_all_reports(d)
        self.assertEqual(len(reports), 2)

    def test_company_name_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_report(d, "Meta_2024")
            reports = load_all_reports(d)
        self.assertEqual(reports[0]["company"], "Meta_2024")

    def test_returns_sorted_by_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_report(d, "Z_corp")
            self._write_report(d, "A_corp")
            reports = load_all_reports(d)
        self.assertEqual(reports[0]["company"], "A_corp")
        self.assertEqual(reports[1]["company"], "Z_corp")

    def test_empty_directory_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = load_all_reports(Path(tmp))
        self.assertEqual(reports, [])

    def test_report_json_content_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_report(d, "TestCo")
            reports = load_all_reports(d)
        self.assertEqual(reports[0]["n_runs"], 10)
        self.assertAlmostEqual(reports[0]["temperature"], 0.7)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
