#!/usr/bin/env python3
"""
Map journal coverage by seed link and compare against a downloaded articles JSON.

What it does
------------
1) Counts how many journals (revistas) you have per base domain from JOURNAL_SEEDS
   (e.g., link.springer.com, nature.com, biomedcentral.com, springeropen.com).
2) Loads your downloaded articles JSON and counts articles per journal.
3) Compares seeds vs downloaded to find missing journals (0 articles) and totals per domain.
4) Writes a JSON report to 'coverage_report.json' and prints a compact console summary.

Input
-----
- JOURNAL_SEEDS: your dictionary of journals -> seed URL
- DOWNLOADED_JSON_PATH: path to your downloaded articles JSON (list of dicts with keys:
    journal, source, article_url, title, abstract)

Output
------
- coverage_report.json with:
  {
    "by_journal": { "<journal>": { "seed_url": "...", "domain": "...",
                                   "downloaded_articles": N } },
    "missing_journals": ["..."],
    "by_domain": {
        "<domain>": {
          "journals_in_seeds": M,
          "journals_with_any_articles": K,
          "total_downloaded_articles": T,
          "journals": ["..."]
        }
    },
    "json_summary": {
        "total_articles_in_json": ...,
        "journals_in_json_only_not_in_seeds": [...]
    }
  }
"""

import json
import os
from collections import Counter, defaultdict
from urllib.parse import urlparse
from typing import Dict, List, Any, Tuple

# ----------------- CONFIG ----------------- #
DOWNLOADED_JSON_PATH = "geojournals_articles.json"   # your downloaded JSON file
REPORT_JSON_PATH = "coverage_report.json"

JOURNAL_SEEDS = {
    # Springer journals (some with explicit /articles or volumes-and-issues pages)
    "Acta Geochemica": "https://link.springer.com/journal/11631",
    "Aquatic Geochemistry": "https://link.springer.com/journal/10498",
    "Arabian Journal of Geociences": "https://link.springer.com/journal/12517",
    "Biogeochemistry": "https://link.springer.com/journal/10533",
    "Computational Geosciences": "https://link.springer.com/journal/10596",
    "Discover Geoscience": "https://link.springer.com/journal/44288",
    "Frontiers of Earth Science": "https://link.springer.com/journal/11707",
    "Geo-Marine Letters": "https://link.springer.com/journal/367",
    "Geochemistry International": "https://link.springer.com/journal/11476",
    "Geosciences Journal": "https://link.springer.com/journal/12303",
    "Journal of Earth Science": "https://link.springer.com/journal/12583",
    "Journal of Earth System Science": "https://link.springer.com/journal/12040",
    "Journal of Iberian Geology": "https://link.springer.com/journal/41513",
    "Journal of Petroleum Exploration and Production Technology": "https://link.springer.com/journal/13202/articles",
    "Mediterranean Geoscience Reviews": "https://link.springer.com/journal/42990",
    "Moscow University Geology Bulletin": "https://link.springer.com/journal/11969",
    "Paleontological Journal": "https://link.springer.com/journal/11492",
    "Russian Journal of Pacific Geology": "https://link.springer.com/journal/11720",
    "Science China Earth Sciences": "https://link.springer.com/journal/11430/volumes-and-issues",

    # Nature portfolio
    "Nature Geoscience": "https://www.nature.com/ngeo/",

    # BMC & SpringerOpen
    "Geochemical Transactions": "https://geochemicaltransactions.biomedcentral.com/",
    "Progress in Earth and Planetary Science": "https://progearthplanetsci.springeropen.com/",
}
# --------------- END CONFIG --------------- #


def domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower().strip()
        # normalize common www
        return netloc.replace("www.", "")
    except Exception:
        return ""


def load_downloaded_json(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"[WARN] Downloaded JSON not found: {path}. Proceeding with empty dataset.")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("[WARN] JSON is not a list. Expecting a list of article objects.")
            return []
        return data
    except Exception as e:
        print(f"[WARN] Could not read JSON: {e}. Proceeding with empty dataset.")
        return []


def summarize_seeds(seeds: Dict[str, str]) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      - seeds_by_domain: {domain: [journal, ...]}
      - per_journal_seed: {journal: {"seed_url": ..., "domain": ...}}
    """
    seeds_by_domain: Dict[str, List[str]] = defaultdict(list)
    per_journal_seed: Dict[str, Dict[str, Any]] = {}
    for journal, url in seeds.items():
        d = domain_of(url)
        seeds_by_domain[d].append(journal)
        per_journal_seed[journal] = {"seed_url": url, "domain": d}
    return seeds_by_domain, per_journal_seed


def count_articles_per_journal(downloaded: List[Dict[str, Any]]) -> Counter:
    """
    Expects each item to have a 'journal' field (string).
    """
    c = Counter()
    for item in downloaded:
        jname = (item.get("journal") or "").strip()
        if jname:
            c[jname] += 1
    return c


def build_coverage_report(
    seeds: Dict[str, str],
    downloaded: List[Dict[str, Any]]
) -> Dict[str, Any]:
    seeds_by_domain, per_journal_seed = summarize_seeds(seeds)
    articles_per_journal = count_articles_per_journal(downloaded)

    # Map journal -> details + downloaded count
    by_journal: Dict[str, Dict[str, Any]] = {}
    for jname, meta in per_journal_seed.items():
        by_journal[jname] = {
            "seed_url": meta["seed_url"],
            "domain": meta["domain"],
            "downloaded_articles": int(articles_per_journal.get(jname, 0)),
        }

    # Journals from JSON that are not in seeds (maybe typos or extra sources)
    journals_only_in_json = sorted(
        set(articles_per_journal.keys()) - set(per_journal_seed.keys())
    )

    # Missing coverage (zero articles)
    missing = sorted([j for j, rec in by_journal.items() if rec["downloaded_articles"] == 0])

    # Aggregate by domain
    by_domain: Dict[str, Dict[str, Any]] = {}
    for dom, jlist in seeds_by_domain.items():
        total_articles = sum(by_journal[j]["downloaded_articles"] for j in jlist)
        with_any = sum(1 for j in jlist if by_journal[j]["downloaded_articles"] > 0)
        by_domain[dom] = {
            "journals_in_seeds": len(jlist),
            "journals_with_any_articles": with_any,
            "total_downloaded_articles": total_articles,
            "journals": sorted(jlist),
        }

    report = {
        "by_journal": by_journal,
        "missing_journals": missing,
        "by_domain": by_domain,
        "json_summary": {
            "total_articles_in_json": int(sum(articles_per_journal.values())),
            "journals_in_json_only_not_in_seeds": journals_only_in_json
        }
    }
    return report


def print_console_summary(report: Dict[str, Any]) -> None:
    try:
        from prettytable import PrettyTable
    except ImportError:
        PrettyTable = None

    print("\n=== Coverage by domain (journals in seeds vs with any articles) ===")
    if PrettyTable:
        t = PrettyTable(["Domain", "Journals (seeds)", "Journals with articles", "Total articles"])
        for dom, info in sorted(report["by_domain"].items()):
            t.add_row([
                dom,
                info["journals_in_seeds"],
                info["journals_with_any_articles"],
                info["total_downloaded_articles"],
            ])
        print(t)
    else:
        for dom, info in sorted(report["by_domain"].items()):
            print(f"- {dom}: seeds={info['journals_in_seeds']}, "
                  f"with_articles={info['journals_with_any_articles']}, "
                  f"total_articles={info['total_downloaded_articles']}")

    print("\n=== Missing journals (0 articles) ===")
    if report["missing_journals"]:
        for j in report["missing_journals"]:
            print(f"- {j}")
    else:
        print("(none)")

    extra = report.get("json_summary", {}).get("journals_in_json_only_not_in_seeds", [])
    if extra:
        print("\n=== Journals present in JSON but NOT in seeds (check names / sources) ===")
        for j in extra:
            print(f"- {j}")


def main() -> None:
    downloaded = load_downloaded_json(DOWNLOADED_JSON_PATH)
    report = build_coverage_report(JOURNAL_SEEDS, downloaded)

    # Write JSON report
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nReport written to: {REPORT_JSON_PATH}")
    print_console_summary(report)


if __name__ == "__main__":
    main()