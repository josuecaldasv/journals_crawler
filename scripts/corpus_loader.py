"""
Cargador/normalizador de 2 datasets JSON:
- JSON 1: lista de objetos con 'title' y opcional 'abstract'
- JSON 2: dict { category: [ items ] } con 'item_title' (sin abstract)

Devuelve:
- docs: List[str] -> "title\n\nabstract" o solo "title"
- meta_df: pd.DataFrame con columnas: id, source_file, category, title, abstract_available,
           doi, url, year, journal, content_type, hash_key
Opciones:
- title_weight: duplica o pondera el título en el texto para dar más peso en TF-IDF/estadísticos
- drop_non_english: si deseas filtrar por idioma (requiere 'langdetect' opcional)
- dedup: exacta por DOI/URL/título y near-duplicate básico por hashing normalizado
"""

from __future__ import annotations
import json, re, hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict

import pandas as pd

SAFE_SPACE = "\n\n"

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _normalize_title(s: str) -> str:
    s = _norm_ws(s)
    # normalización leve para hashing/dedup
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-']", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _compose_text(title: str, abstract: Optional[str], title_weight: int = 1) -> str:
    """Concatena título y abstract. Si title_weight>1, repite el título para darle mayor peso."""
    title = _norm_ws(title)
    if not title:
        return _norm_ws(abstract or "")
    parts = [title] * max(1, title_weight)
    if abstract:
        parts.append(_norm_ws(abstract))
    return SAFE_SPACE.join(parts)

def _lang_is_english(text: str) -> bool:
    """Filtro opcional. Requiere langdetect (pip install langdetect)."""
    try:
        from langdetect import detect
        lang = detect(text[:1000])  # mirar solo el inicio por velocidad
        return lang == "en"
    except Exception:
        # Si no está instalado o falla, no filtra.
        return True

def load_json1(path: str | Path) -> List[Dict]:
    data = json.load(open(path, "r", encoding="utf-8"))
    out = []
    for obj in data:
        out.append({
            "source_file": Path(path).name,
            "category": None,
            "title": obj.get("title") or "",
            "abstract": obj.get("abstract") or "",
            "doi": obj.get("doi") or "",
            "url": obj.get("article_url") or "",
            "year": obj.get("publication_year") or "",
            "journal": obj.get("journal") or "",
            "content_type": obj.get("content_type") or ( "Article" if obj.get("abstract") else "" ),
        })
    return out

def load_json2(path: str | Path) -> List[Dict]:
    data = json.load(open(path, "r", encoding="utf-8"))
    out = []
    for category, items in data.items():
        for it in items:
            out.append({
                "source_file": Path(path).name,
                "category": category,
                "title": it.get("item_title") or "",
                "abstract": "",  # no hay abstract en dataset 2
                "doi": it.get("doi") or "",
                "url": it.get("url") or "",
                "year": it.get("publication_year") or "",
                "journal": it.get("publication_title") or "",
                "content_type": it.get("content_type") or "",
            })
    return out

def unify_corpus(
    json1_path: str | Path,
    json2_path: str | Path,
    title_weight: int = 1,
    drop_non_english: bool = False,
    dedup: bool = True
) -> Tuple[List[str], pd.DataFrame]:
    rows = []
    rows.extend(load_json1(json1_path))
    rows.extend(load_json2(json2_path))

    # Deduplicación exacta por DOI/URL/título normalizado
    seen = set()
    unique_rows = []
    for r in rows:
        key = r.get("doi") or r.get("url") or _normalize_title(r.get("title",""))
        if not key:
            key = _hash(r.get("title",""))
        if key not in seen:
            seen.add(key)
            r["hash_key"] = key
            unique_rows.append(r)

    # Dedup aproximado simple por título normalizado (para colisiones sin DOI/URL)
    if dedup:
        bucket = OrderedDict()
        for r in unique_rows:
            k = _normalize_title(r.get("title",""))
            k = k if k else _hash(r.get("title",""))
            if k not in bucket:
                bucket[k] = r
        unique_rows = list(bucket.values())

    docs = []
    meta = []
    for i, r in enumerate(unique_rows):
        text = _compose_text(r["title"], r.get("abstract",""), title_weight=title_weight)
        if drop_non_english and not _lang_is_english(text):
            continue
        docs.append(text)
        meta.append({
            "id": i,
            "source_file": r["source_file"],
            "category": r["category"],
            "title": _norm_ws(r["title"]),
            "abstract_available": bool(r.get("abstract")),
            "doi": r.get("doi",""),
            "url": r.get("url",""),
            "year": r.get("year",""),
            "journal": r.get("journal",""),
            "content_type": r.get("content_type",""),
            "hash_key": r["hash_key"],
            "text_len": len(text),
        })

    meta_df = pd.DataFrame(meta)
    return docs, meta_df
