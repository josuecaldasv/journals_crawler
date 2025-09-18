#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import csv
import argparse

def detect_delimiter(path: Path, sample_bytes: int = 4096) -> str:
    """
    Detect the most probable delimiter (comma, tab, semicolon).
    """
    with path.open('r', encoding='utf-8-sig', errors='replace') as f:
        sample = f.read(sample_bytes)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[',', '\t', ';', '|'])
        return dialect.delimiter
    except Exception:
        if sample.count('\t') > sample.count(','):
            return '\t'
        return ','

def read_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    """
    Read a CSV/TSV file and return headers and rows.
    """
    delimiter = detect_delimiter(path)
    with path.open('r', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]

def normalize_header(header: str) -> str:
    """
    Normalize header names to consistent snake_case.
    """
    header = (header or "").strip()
    mapping = {
        "Item Title": "item_title",
        "Publication Title": "publication_title",
        "Book Series Title": "book_series_title",
        "Journal Volume": "journal_volume",
        "Journal Issue": "journal_issue",
        "Item DOI": "doi",
        "Authors": "authors",
        "Publication Year": "publication_year",
        "URL": "url",
        "Content Type": "content_type",
    }
    if header in mapping:
        return mapping[header]
    header = re.sub(r"\s+", "_", header)
    header = re.sub(r"[^\w_]", "", header)
    return header.lower()

def clean_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value

def normalize_doi(doi: str) -> str:
    """
    Clean and normalize DOI (remove prefixes like https://doi.org/).
    """
    if not doi:
        return ""
    doi = doi.strip().replace("\\", "/")
    doi = re.sub(r"(?i)^https?://(dx\.)?doi\.org/", "", doi)
    return doi.lower()

def build_record(headers: List[str], row: List[str]) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    for i, h in enumerate(headers):
        key = normalize_header(h)
        value = row[i].strip() if i < len(row) else ""
        record[key] = clean_value(value)
    record["doi"] = normalize_doi(record.get("doi", ""))
    pub_year = record.get("publication_year", "")
    if isinstance(pub_year, str) and pub_year.isdigit():
        record["publication_year"] = int(pub_year)
    return record

def fallback_identifier(record: Dict[str, Any]) -> str:
    """
    Use title+year as a fallback unique key if DOI is missing.
    """
    title = (record.get("item_title") or "").strip().lower()
    year = str(record.get("publication_year") or "").strip()
    return f"{title}::{year}"

def consolidate_csvs(input_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Consolidate all CSV files into one dictionary grouped by filename (without extension).
    Deduplicate globally by DOI or fallback key.
    """
    consolidated: Dict[str, List[Dict[str, Any]]] = {}
    seen: set = set()

    for file_path in sorted(input_dir.glob("*.csv")):
        group_key = file_path.stem
        headers, rows = read_csv(file_path)
        if not headers:
            continue
        group_items: List[Dict[str, Any]] = []
        for row in rows:
            rec = build_record(headers, row)
            unique_id = rec.get("doi") or fallback_identifier(rec)
            if unique_id in seen:
                continue
            seen.add(unique_id)
            group_items.append(rec)
        if group_items:
            consolidated.setdefault(group_key, []).extend(group_items)
    return consolidated


def count_items(json_file: Path):
    data = json.loads(json_file.read_text(encoding="utf-8"))
    counts = {key: len(items) for key, items in data.items()}
    counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    print(f"{'Keyword':50} | {'Items'}")
    print("-"*65)
    for key, count in counts.items():
        print(f"{key:50} | {count}")
    total = sum(counts.values())
    print("-"*65)
    print(f"{'TOTAL':50} | {total}")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple CSV files into one JSON file for NLP preprocessing.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing the CSV files.")
    parser.add_argument("--output", required=True, type=Path, help="Path for the output JSON file.")
    parser.add_argument("--pretty", action="store_true", help="Format JSON with indentation.")
    args = parser.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise SystemExit(f"Folder {args.input_dir} does not exist or is not a directory.")

    result = consolidate_csvs(args.input_dir)

    ordered: Dict[str, List[Dict[str, Any]]] = {}
    for k in sorted(result.keys()):
        ordered[k] = sorted(
            result[k],
            key=lambda r: (-(r.get("publication_year") or 0), (r.get("item_title") or "").lower())
        )

    if args.pretty:
        args.output.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        args.output.write_text(json.dumps(ordered, ensure_ascii=False), encoding="utf-8")

    print(f"Completed -> {args.output} ({sum(len(v) for v in ordered.values())} unique records)")

    count_items(args.output)

if __name__ == "__main__":
    main()
