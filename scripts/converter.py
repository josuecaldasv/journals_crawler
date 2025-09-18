# -*- coding: utf-8 -*-
"""
Transform out_bigrams.csv / out_trigrams.csv -> single-sheet Excel files sorted by G² (desc),
keeping only ["keyword", "G2", "freq"]. Reads CSVs in chunks to stay memory-friendly.

Outputs:
- ../keywords/parabras_chave_bigramas.xlsx
- ../keywords/paravras_chave_trigramas.xlsx
"""

import os
import pandas as pd
from typing import List

# ------------------ Config ------------------

INPUT_BIGRAMS_CSV  = "../metrics/out_bigrams.csv"
INPUT_TRIGRAMS_CSV = "../metrics/out_trigrams.csv"

OUTPUT_BIGRAMS_XLSX  = "../keywords/parabras_chave_bigramas.xlsx"
OUTPUT_TRIGRAMS_XLSX = "../keywords/paravras_chave_trigramas.xlsx"

# Candidate columns for the token (some exports mislabel)
BIGRAM_TOKEN_CANDIDATES  = ["bigram", "term"]
TRIGRAM_TOKEN_CANDIDATES = ["trigram", "bigram", "term"]

G2_COL   = "G2"
FREQ_COL = "freq"

CHUNKSIZE = 100_000  # rows per chunk

# ------------------ Helpers ------------------

def detect_token_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of {candidates} found in columns: {list(df.columns)}")

def read_all_sorted_by_g2(csv_path: str, token_candidates: List[str]) -> pd.DataFrame:
    """
    Stream the CSV in chunks, keep only [token, G2, freq], concat all,
    drop duplicates on token, and sort by G2 desc.
    """
    parts = []
    token_col = None

    for chunk in pd.read_csv(csv_path, chunksize=CHUNKSIZE):
        # detect token column once
        if token_col is None:
            token_col = detect_token_col(chunk, token_candidates)

        # keep minimal columns
        cols = [c for c in [token_col, G2_COL, FREQ_COL] if c in chunk.columns]
        chunk = chunk[cols].copy()

        # coerce numeric
        if G2_COL in chunk.columns:
            chunk[G2_COL] = pd.to_numeric(chunk[G2_COL], errors="coerce")
        if FREQ_COL in chunk.columns:
            chunk[FREQ_COL] = pd.to_numeric(chunk[FREQ_COL], errors="coerce")

        # clean rows
        chunk[token_col] = chunk[token_col].astype(str).str.strip()
        chunk = chunk.dropna(subset=[token_col, G2_COL, FREQ_COL])

        parts.append(chunk)

    if not parts:
        return pd.DataFrame(columns=["keyword", G2_COL, FREQ_COL])

    df = pd.concat(parts, ignore_index=True)

    # rename token -> keyword
    if token_col != "keyword":
        df = df.rename(columns={token_col: "keyword"})

    # drop duplicates on keyword (keep highest G2 by sorting later)
    df = df.dropna(subset=["keyword"]).drop_duplicates(subset=["keyword"], keep="first")

    # sort by G2 desc (stable mergesort)
    df = df.sort_values(G2_COL, ascending=False, kind="mergesort")

    # ensure final column order
    df = df[["keyword", G2_COL, FREQ_COL]].reset_index(drop=True)
    return df

def write_single_sheet_xlsx(df: pd.DataFrame, out_path: str, sheet_name: str = "Top"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

# ------------------ Main ------------------

def main():
    # BIGRAMS
    print("[INFO] Reading bigrams...")
    df_bi = read_all_sorted_by_g2(INPUT_BIGRAMS_CSV, BIGRAM_TOKEN_CANDIDATES)
    print(f"[INFO] Bigrams rows: {len(df_bi)}")
    write_single_sheet_xlsx(df_bi, OUTPUT_BIGRAMS_XLSX, sheet_name="Bigrams")
    print(f"[OK] Wrote: {OUTPUT_BIGRAMS_XLSX}")

    # TRIGRAMS
    print("[INFO] Reading trigrams...")
    df_tri = read_all_sorted_by_g2(INPUT_TRIGRAMS_CSV, TRIGRAM_TOKEN_CANDIDATES)
    print(f"[INFO] Trigrams rows: {len(df_tri)}")
    write_single_sheet_xlsx(df_tri, OUTPUT_TRIGRAMS_XLSX, sheet_name="Trigrams")
    print(f"[OK] Wrote: {OUTPUT_TRIGRAMS_XLSX}")

if __name__ == "__main__":
    main()