"""
Geo-journals crawler: collects (title, abstract/description) from selected journals.

Usage:
    python crawl_geo_journals.py

Output:
    geojournals_articles.json  (columns: journal, source, article_url, title, abstract)
"""

import time
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
import json


# --------------------------- Config --------------------------- #

OUTPUT_JSON = "geojournals_articles.json"
MAX_ARTICLES_PER_JOURNAL = 120          # safety cap per journal; adjust as needed
MAX_WORKERS = 8                         # parallel fetchers (be polite!)
REQ_TIMEOUT = 15                        # seconds
PAUSE_BETWEEN_REQUESTS = 0.7            # polite delay between HTTP calls per thread

HEADERS = {
    "User-Agent": "geo-crawler/1.0 (+research; contact: you@example.com)"
}

# Journal seeds (you can add more or adjust)
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

# ------------------------ HTTP utilities --------------------- #

class FetchError(Exception):
    pass

def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS,
        max_retries=0,  # we do retries via tenacity
    )
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=0.7, min=0.7, max=6),
    retry=retry_if_exception_type(FetchError),
)
def fetch_html(session: requests.Session, url: str) -> str:
    try:
        resp = session.get(url, timeout=REQ_TIMEOUT)
        if resp.status_code >= 400:
            raise FetchError(f"HTTP {resp.status_code} for {url}")
        # polite pause
        time.sleep(PAUSE_BETWEEN_REQUESTS)
        return resp.text
    except (requests.RequestException, requests.Timeout) as e:
        raise FetchError(str(e)) from e

def make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")

# ----------------------- Domain helpers ---------------------- #

def is_springer(url: str) -> bool:
    return "link.springer.com" in urlparse(url).netloc

def is_nature(url: str) -> bool:
    host = urlparse(url).netloc
    return "nature.com" in host and "www.nature.com" in host

def is_bmc(url: str) -> bool:
    return "biomedcentral.com" in urlparse(url).netloc

def is_springeropen(url: str) -> bool:
    return "springeropen.com" in urlparse(url).netloc

def normalize_list_pages(seed_url: str) -> list[str]:
    """
    Given a journal seed, produce candidate listing URLs to discover article links.
    We try several common patterns per platform.
    """
    urls = [seed_url]

    if is_springer(seed_url):
        # common list pages on Springer
        for suffix in ("/articles", "/latest", "/volumes-and-issues"):
            if not seed_url.rstrip("/").endswith(suffix.strip("/")):
                urls.append(seed_url.rstrip("/") + suffix)

    if is_nature(seed_url):
        # Nature has issue archive & subject feeds
        urls.append(urljoin(seed_url, "research"))
        urls.append(urljoin(seed_url, "news-and-comment"))
        urls.append(urljoin(seed_url, "articles"))

    if is_bmc(seed_url) or is_springeropen(seed_url):
        # BMC/SpringerOpen usually have "/articles" as list
        if not seed_url.rstrip("/").endswith("articles"):
            urls.append(seed_url.rstrip("/") + "/articles")

    # Deduplicate preserving order
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out

def extract_article_links_from_listing(base_url: str, soup: BeautifulSoup) -> list[str]:
    """
    Extract candidate article links from listing pages across platforms.
    We filter to plausible article-detail URLs.
    """
    links = set()
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if not href:
            continue
        url = urljoin(base_url, href)

        # Heuristics for article detail pages
        if is_springer(base_url):
            # Springer articles often /article/10.xxxx/...
            if re.search(r"/article/10\.\d{4,9}/", url):
                links.add(url)
        elif is_nature(base_url):
            # Nature articles often /articles/xxxxx
            if re.search(r"/articles/[a-zA-Z0-9\-\._]+", url) and "/collections/" not in url:
                links.add(url)
        elif is_bmc(base_url) or is_springeropen(base_url):
            # BMC/SpringerOpen: /articles/10.xxxx/...
            if re.search(r"/articles/10\.\d{4,9}/", url):
                links.add(url)

    return list(links)

def text_or_none(el) -> str | None:
    if el:
        txt = el.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", txt) if txt else None
    return None

def first_meta(soup: BeautifulSoup, *names) -> str | None:
    for n in names:
        m = soup.find("meta", attrs={"name": n})
        if m and m.get("content"):
            return m["content"].strip()
    return None

def first_meta_property(soup: BeautifulSoup, *props) -> str | None:
    for p in props:
        m = soup.find("meta", attrs={"property": p})
        if m and m.get("content"):
            return m["content"].strip()
    return None

def extract_title_generic(soup: BeautifulSoup) -> str | None:
    # Try common scholarly meta first
    meta_title = first_meta(soup, "citation_title", "dc.title")
    if meta_title:
        return meta_title
    ogt = first_meta_property(soup, "og:title", "twitter:title")
    if ogt:
        return ogt
    # Try typical h1
    h1 = soup.find("h1")
    return text_or_none(h1)

def extract_abstract_springer(soup: BeautifulSoup) -> str | None:
    # Springer variants: Abstract sections often with id="Abs1" or class "Abstract"
    # Common container: section[data-title='Abstract'] or section#Abs1
    candidates = []
    candidates += soup.select("section#Abs1, section#Abs2, section.Abstract")
    candidates += soup.select("div#Abs1-content, div#Abs2-content")
    candidates += soup.select("section[data-title='Abstract']")
    for c in candidates:
        txt = text_or_none(c)
        if txt and len(txt) > 40:
            return txt

    # Fallback meta
    meta = first_meta(soup, "dc.description", "description")
    if meta and len(meta) > 40:
        return meta

    ogd = first_meta_property(soup, "og:description", "twitter:description")
    if ogd and len(ogd) > 40:
        return ogd
    return None

def extract_abstract_nature(soup: BeautifulSoup) -> str | None:
    sel = [
        "section#Abs1 .c-article-section__content",
        "section#abstract .c-article-section__content",
        "section[data-title='Abstract'] .c-article-section__content",
        "div#Abs1",  # fallback
        "section#abstract",  # fallback
    ]
    for s in sel:
        el = soup.select_one(s)
        txt = text_or_none(el)
        if txt and len(txt) > 40:
            return txt

    meta = first_meta(soup, "dc.description", "description")
    if meta and len(meta) > 40:
        return meta

    ogd = first_meta_property(soup, "og:description", "twitter:description")
    if ogd and len(ogd) > 40:
        return ogd
    return None

def extract_abstract_bmc(soup: BeautifulSoup) -> str | None:
    sel = [
        "section#Abs1",
        "div#Abs1-content",
        "section[data-title='Abstract']",
        "section.Abstract",
        "div.Abstract",
    ]
    for s in sel:
        el = soup.select_one(s)
        txt = text_or_none(el)
        if txt and len(txt) > 40:
            return txt

    meta = first_meta(soup, "dc.description", "description")
    if meta and len(meta) > 40:
        return meta

    ogd = first_meta_property(soup, "og:description", "twitter:description")
    if ogd and len(ogd) > 40:
        return ogd
    return None

def extract_title_and_abstract(article_url: str, soup: BeautifulSoup) -> tuple[str | None, str | None]:
    title = extract_title_generic(soup)

    if "link.springer.com" in article_url:
        abstract = extract_abstract_springer(soup)
    elif "nature.com" in article_url:
        abstract = extract_abstract_nature(soup)
    elif "biomedcentral.com" in article_url or "springeropen.com" in article_url:
        abstract = extract_abstract_bmc(soup)
    else:
        # generic fallback
        abstract = (
            first_meta(soup, "dc.description", "description")
            or first_meta_property(soup, "og:description", "twitter:description")
        )
    return title, abstract

# --------------------------- Pipeline ------------------------ #

@dataclass
class ArticleRecord:
    journal: str
    source: str
    article_url: str
    title: str | None
    abstract: str | None

def discover_articles(session: requests.Session, journal: str, seed_url: str) -> list[str]:
    discovered = set()
    for listing in normalize_list_pages(seed_url):
        try:
            html = fetch_html(session, listing)
            soup = make_soup(html)
            links = extract_article_links_from_listing(listing, soup)
            for u in links:
                discovered.add(u)
        except Exception:
            # Keep going—list pages vary a lot
            continue
        if len(discovered) >= MAX_ARTICLES_PER_JOURNAL:
            break
    return list(discovered)[:MAX_ARTICLES_PER_JOURNAL]

def fetch_article_record(session: requests.Session, journal: str, article_url: str) -> ArticleRecord | None:
    try:
        html = fetch_html(session, article_url)
        soup = make_soup(html)
        title, abstract = extract_title_and_abstract(article_url, soup)

        # Basic validation: require title; abstract may be None
        if not title:
            # sometimes title is inside <meta property="og:title"> only
            title = first_meta_property(soup, "og:title", "twitter:title")

        return ArticleRecord(
            journal=journal,
            source=urlparse(article_url).netloc,
            article_url=article_url,
            title=title.strip() if title else None,
            abstract=abstract.strip() if abstract else None,
        )
    except Exception:
        return None

def crawl_all(journal_seeds: dict[str, str]) -> list[ArticleRecord]:
    session = build_session()
    all_records: list[ArticleRecord] = []

    # 1) Discover article URLs per journal
    discoveries: dict[str, list[str]] = {}
    for journal, seed in tqdm(journal_seeds.items(), desc="Discovering article links"):
        urls = discover_articles(session, journal, seed)
        discoveries[journal] = urls

    # 2) Fetch article pages in parallel (bounded)
    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for journal, urls in discoveries.items():
            for u in urls:
                tasks.append(ex.submit(fetch_article_record, session, journal, u))

        for fut in tqdm(as_completed(tasks), total=len(tasks), desc="Fetching articles"):
            rec = fut.result()
            if rec:
                all_records.append(rec)

    return all_records


def write_json(records: list[ArticleRecord], path: str) -> None:
    data = []
    for r in records:
        data.append({
            "journal": r.journal,
            "source": r.source,
            "article_url": r.article_url,
            "title": r.title or "",
            "abstract": r.abstract or ""
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("Starting crawl…")
    records = crawl_all(JOURNAL_SEEDS)
    print(f"Fetched {len(records)} articles. Writing JSON...")
    write_json(records, OUTPUT_JSON)
    print(f"Done: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
