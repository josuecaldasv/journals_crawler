# -*- coding: utf-8 -*-
"""
Plot top-N terms for multiple CSVs (unigrams/bigrams/bigrams_pos/trigrams/trigrams_pos),
with pre-filtering, per-metric pastel colors, descending order, right-end labels,
and dynamic xlim to avoid clipping.

Input files & columns:
- out_unigrams.csv:      term, df, freq, burstiness, tf_rel, tfidf_mean
- out_bigrams.csv:       bigram, freq, G2, PMI
- out_bigrams_pos.csv:   bigram, freq, G2, PMI
- out_trigrams.csv:      trigram(or bigram*), freq, G2, PMI
- out_trigrams_pos.csv:  trigram(or bigram*), freq, G2, PMI

(*) Some pipelines export trigrams with column name "bigram" by mistake.
    This script will auto-detect/rename to a generic 'token_col'.

Author: you :)
"""

import os
import re
import math
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ Global Configuration ------------------

INPUT_DIR  = "../metrics"
OUTPUT_DIR = "../plots"
TOP_N = 50

# Stop-lists per file-type (case-insensitive). Add domain-specific noise here.
STOP_TERMS = {
    "unigrams":      set({"study", "source", "low", "analysis"}),
    "bigrams":       set({"case study"}),  
    "bigrams_pos":   set({"case study"}),
    "trigrams":      set(),
    "trigrams_pos":  set(),
}

# Regex stop-patterns applied to the token (term/bigram/trigram)
STOP_PATTERNS = [
    r"^\d+$",         # numbers only
    r"^[a-z]$",       # single letter
]

# Minimum thresholds (to reduce noise)
MIN_DF_UNI   = 1
MIN_FREQ_ALL = 1

FIGSIZE = (10, 12)
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICKS = 10
BAR_EDGEWIDTH = 0.6
LABEL_FONTSIZE = 9
XLIM_MARGIN = 0.10     # initial 10% margin to the right
LABEL_OFFSET_FRAC = 0.01  # label offset as fraction of (max-min)

# Pastel palette per metric (consistent across all files)
METRIC_COLORS = {
    "tfidf_mean": "#a6cee3",  # pastel blue
    "df":         "#b2df8a",  # pastel green
    "freq":       "#fdbf6f",  # pastel orange
    "burstiness": "#cab2d6",  # pastel purple
    "tf_rel":     "#ccebc5",  # pastel mint
    "G2":         "#fb9a99",  # pastel salmon
    "PMI":        "#fddbc7",  # pastel peach
}

# File specifications (what to load, how to interpret)
FILE_SPECS = [
    {
        "key": "unigrams",
        "path": os.path.join(INPUT_DIR, "out_unigrams.csv"),
        "token_candidates": ["term"],       # name of the token column
        "metrics": [
            ("tfidf_mean", "TF-IDF (mean per document)"),
            ("df",         "Document Frequency (DF)"),
            ("freq",       "Total Frequency (TF)"),
            ("burstiness", "Burstiness (Var/Mean per document)"),
            ("tf_rel",     "Relative Frequency (TF / total_tokens)"),
        ],
        "min_df": MIN_DF_UNI,
        "min_freq": MIN_FREQ_ALL,
    },
    {
        "key": "bigrams",
        "path": os.path.join(INPUT_DIR, "out_bigrams.csv"),
        "token_candidates": ["bigram", "term"],
        "metrics": [
            ("G2",   "Log-Likelihood (G²)"),
            ("freq", "Total Frequency"),
            ("PMI",  "PMI"),
        ],
        "min_df": None,
        "min_freq": MIN_FREQ_ALL,
    },
    {
        "key": "bigrams_pos",
        "path": os.path.join(INPUT_DIR, "out_bigrams_pos.csv"),
        "token_candidates": ["bigram", "term"],
        "metrics": [
            ("G2",   "Log-Likelihood (G²)"),
            ("freq", "Total Frequency"),
            ("PMI",  "PMI"),
        ],
        "min_df": None,
        "min_freq": MIN_FREQ_ALL,
    },
    {
        "key": "trigrams",
        "path": os.path.join(INPUT_DIR, "out_trigrams.csv"),
        # Some exports mislabel trigrams as 'bigram'; we will catch and rename.
        "token_candidates": ["trigram", "bigram", "term"],
        "metrics": [
            ("G2",   "Log-Likelihood (G²)"),
            ("freq", "Total Frequency"),
            ("PMI",  "PMI"),
        ],
        "min_df": None,
        "min_freq": MIN_FREQ_ALL,
    },
    {
        "key": "trigrams_pos",
        "path": os.path.join(INPUT_DIR, "out_trigrams_pos.csv"),
        "token_candidates": ["trigram", "bigram", "term"],
        "metrics": [
            ("G2",   "Log-Likelihood (G²)"),
            ("freq", "Total Frequency"),
            ("PMI",  "PMI"),
        ],
        "min_df": None,
        "min_freq": MIN_FREQ_ALL,
    },
]

# ------------------ Utilities ------------------

def ensure_output_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def passes_regex_filters(token: str, patterns) -> bool:
    """Return True if token does NOT match any exclusion pattern."""
    for pat in patterns:
        if re.search(pat, token):
            return False
    return True

def detect_token_column(df: pd.DataFrame, candidates) -> str:
    """
    Find the first present column among candidates and return it.
    Raise if none found.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of the token candidates {candidates} are present in the CSV columns: {list(df.columns)}")

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def normalize_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def filter_rows(df: pd.DataFrame, token_col: str, key: str, min_df: int | None, min_freq: int | None) -> pd.DataFrame:
    # Basic NA handling
    df = df.copy()
    df = df.dropna(subset=[token_col]).fillna(0)

    # Thresholds
    if min_df is not None and "df" in df.columns:
        df = df[df["df"] >= min_df]
    if min_freq is not None and "freq" in df.columns:
        df = df[df["freq"] >= min_freq]

    # Manual stop-list (case-insensitive)
    if STOP_TERMS.get(key):
        stop = {t.lower() for t in STOP_TERMS[key]}
        df = df[~df[token_col].str.lower().isin(stop)]

    # Regex filters
    if STOP_PATTERNS:
        mask = df[token_col].astype(str).apply(lambda t: passes_regex_filters(t, STOP_PATTERNS))
        df = df[mask]

    # Normalize token
    df[token_col] = df[token_col].astype(str).str.strip()
    df = df.drop_duplicates(subset=[token_col])

    return df

def top_by_metric(df: pd.DataFrame, token_col: str, metric: str, top_n: int) -> pd.DataFrame:
    # Sort descending by metric, tie-breaker by freq (if exists)
    by = [metric] + (["freq"] if "freq" in df.columns and metric != "freq" else [])
    asc = [False] + ([False] if len(by) > 1 else [])
    df_sorted = df.sort_values(by, ascending=asc)
    return df_sorted.head(top_n)

def format_value(val) -> str:
    if isinstance(val, (int,)) and not isinstance(val, bool):
        return f"{int(val)}"
    # floats or others:
    try:
        v = float(val)
        return f"{v:.3f}" if abs(v) < 100 else f"{int(v)}"
    except Exception:
        return str(val)

def plot_barh_metric(df_top: pd.DataFrame, token_col: str, metric: str, metric_label: str,
                     title_prefix: str, output_dir: str):
    """
    Horizontal bars, descending (top at top), right-side numeric labels,
    dynamic xlim enlargement to avoid clipping, per-metric pastel color.
    """
    if df_top.empty:
        print(f"[Info] No data to plot for '{title_prefix}' - metric '{metric}'.")
        return

    dplot = df_top.copy().sort_values(metric, ascending=False)
    tokens = dplot[token_col].values
    vals   = dplot[metric].values

    color = METRIC_COLORS.get(metric, "#bdbdbd")  # fallback gray

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.barh(tokens, vals, edgecolor="black", linewidth=BAR_EDGEWIDTH, color=color)

    # Initial right margin so labels don't collide
    vmin, vmax = float(min(vals)), float(max(vals))
    right = vmax * (1 + XLIM_MARGIN) if vmax > 0 else 1.0
    ax.set_xlim(0, right)

    ax.set_xlabel(metric_label, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Token", fontsize=FONT_SIZE_LABEL)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=FONT_SIZE_TICKS)

    ax.set_title(f"{title_prefix} • Top {len(dplot)} by {metric_label}", fontsize=FONT_SIZE_TITLE)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax.yaxis.grid(False)
    ax.invert_yaxis()  # largest at top

    # Label offset
    offset = (vmax - vmin) * LABEL_OFFSET_FRAC if vmax > vmin else (abs(vmax) + 1e-9) * LABEL_OFFSET_FRAC

    texts = []
    for bar, val in zip(bars, vals):
        label = format_value(val)
        t = ax.text(
            float(val) + offset,
            bar.get_y() + bar.get_height()/2.0,
            label,
            va="center", ha="left", fontsize=LABEL_FONTSIZE, color="black",
            clip_on=False
        )
        texts.append(t)

    # Force draw to measure text extents, then extend xlim if needed
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_x_display = max(t.get_window_extent(renderer=renderer).x1 for t in texts)
    inv = ax.transData.inverted()
    max_x_data = inv.transform((max_x_display, 0))[0]
    cur_left, cur_right = ax.get_xlim()
    if max_x_data > cur_right:
        ax.set_xlim(cur_left, max_x_data * 1.02)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    safe_metric = metric.replace("/", "_").replace(" ", "_")
    fname = f"{title_prefix.lower().replace(' ', '_')}_top_{safe_metric}.png"
    fpath = os.path.join(output_dir, fname)
    plt.savefig(fpath, dpi=200)
    print(f"[OK] Saved: {fpath}")
    plt.close(fig)

# ------------------ Main ------------------

def main():
    ensure_output_dir(OUTPUT_DIR)

    for spec in FILE_SPECS:
        key   = spec["key"]
        path  = spec["path"]
        mets  = spec["metrics"]
        cand  = spec["token_candidates"]
        min_df   = spec.get("min_df")
        min_freq = spec.get("min_freq")

        try:
            df = load_csv(path)
        except FileNotFoundError as e:
            print(f"[Warn] {e}")
            continue

        # Detect/normalize token column
        token_col = detect_token_column(df, cand)
        if token_col != "token":
            df = df.rename(columns={token_col: "token"})
            token_col = "token"

        # Normalize numerics (all potential numeric columns)
        num_cols = set(["df", "freq", "burstiness", "tf_rel", "tfidf_mean", "G2", "PMI"])
        df = normalize_numeric(df, num_cols)

        # Filter rows
        df = filter_rows(df, token_col, key, min_df, min_freq)

        print(f"[Info] {key}: {len(df)} rows after filtering.")

        # For each metric defined for this file, plot top-N
        for metric, label in mets:
            if metric not in df.columns:
                print(f"[Info] '{metric}' not in columns for {key}. Skipped.")
                continue
            df_top = top_by_metric(df, token_col, metric, TOP_N)
            title_prefix = {
                "unigrams":     "Unigrams",
                "bigrams":      "Bigrams",
                "bigrams_pos":  "Bigrams (POS-filtered)",
                "trigrams":     "Trigrams",
                "trigrams_pos": "Trigrams (POS-filtered)",
            }.get(key, key.title())
            plot_barh_metric(df_top, token_col, metric, label, title_prefix, OUTPUT_DIR)

if __name__ == "__main__":
    main()
