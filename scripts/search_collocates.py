# -*- coding: utf-8 -*-
from __future__ import annotations
import re
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from corpus_loader import unify_corpus

# NLTK stopwords
import nltk
nltk.data.find("corpora/stopwords")
from nltk.corpus import stopwords

# spaCy
import spacy
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    raise SystemExit("Missing spaCy model en_core_web_sm. Run: python -m spacy download en_core_web_sm")

EN_STOPS = set(stopwords.words("english"))

# ---------- Configuración ----------
@dataclass
class PreprocConfig:
    lowercase: bool = True
    lemmatize: bool = True
    keep_pos_tags: Tuple[str, ...] = ("NOUN", "PROPN", "ADJ")  # filtrar candidatos por POS (para unigrams)
    remove_stopwords: bool = True
    min_token_len: int = 2
    max_token_len: int = 30
    allowed_pattern: re.Pattern = re.compile(r"^[A-Za-z][A-Za-z\-']+$")
    filter_pos_for_unigrams: bool = True

@dataclass
class NgramConfig:
    min_count: int = 3          # frecuencia mínima para considerar n-gramas
    window_size: int = 4        # para co-ocurrencias de ventana (no contiguas)
    use_contiguous: bool = True # bigramas/trigramas contiguos
    use_windowed: bool = False  # co-ocurrencias en ventana (pares)
    max_vocab: Optional[int] = None

# ---------- Utilidades ----------
def clean_token(t: str, cfg: PreprocConfig) -> Optional[str]:
    if cfg.lowercase:
        t = t.lower()
    if len(t) < cfg.min_token_len or len(t) > cfg.max_token_len:
        return None
    if not cfg.allowed_pattern.match(t):
        return None
    if cfg.remove_stopwords and t in EN_STOPS:
        return None
    return t

def spacy_tokenize_docs(texts: List[str], cfg: PreprocConfig) -> List[List[dict]]:
    docs = []
    for doc in tqdm(nlp.pipe(texts, batch_size=64), total=len(texts), desc="spaCy"):
        toks = []
        for tok in doc:
            if not tok.is_space and not tok.is_punct:
                txt = tok.text
                lemma = tok.lemma_ if cfg.lemmatize else tok.text
                pos = tok.pos_
                toks.append({"text": txt, "lemma": lemma, "pos": pos, "is_alpha": tok.is_alpha})
        docs.append(toks)
    return docs

def tokens_from_doc(spacy_doc: List[dict], cfg: PreprocConfig) -> List[str]:
    out = []
    for d in spacy_doc:
        base = d["lemma"] if cfg.lemmatize else d["text"]
        if cfg.filter_pos_for_unigrams and d["pos"] not in cfg.keep_pos_tags:
            continue
        tok = clean_token(base, cfg)
        if tok:
            out.append(tok)
    return out

def tokens_with_pos(spacy_doc: List[dict], cfg: PreprocConfig) -> List[Tuple[str, str]]:
    out = []
    for d in spacy_doc:
        base = d["lemma"] if cfg.lemmatize else d["text"]
        tok = clean_token(base, cfg)
        if tok:
            out.append((tok, d["pos"]))
    return out

# ---------- Candidatos ----------
def contiguous_ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i+n])

def pos_pattern_filter(seqs: List[Tuple[Tuple[str, ...], Tuple[str, ...]]],
                       allowed_patterns: Tuple[Tuple[str, ...], ...]) -> List[Tuple[str, ...]]:
    keep = []
    for words, poses in seqs:
        if poses in allowed_patterns:
            keep.append(words)
    return keep

# ---------- Estadísticos base ----------
def compute_tf_df(unigram_docs: List[List[str]]) -> Tuple[Counter, Counter]:
    tf = Counter()
    df = Counter()
    for doc in unigram_docs:
        tf.update(doc)
        df.update(set(doc))
    return tf, df

def compute_tfidf(unigram_docs: List[List[str]], max_features: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str,int]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs_strings = [" ".join(doc) for doc in unigram_docs]
    vec = TfidfVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(docs_strings)
    vocab = vec.vocabulary_
    avg_scores = np.asarray(X.mean(axis=0)).ravel()
    inv_vocab = {i: t for t, i in vocab.items()}
    df = pd.DataFrame({
        "term": [inv_vocab[i] for i in range(len(inv_vocab))],
        "tfidf_mean": avg_scores
    }).sort_values("tfidf_mean", ascending=False, ignore_index=True)
    return df, vocab

def ngram_freqs(ngram_docs: List[List[Tuple[str, ...]]], min_count: int = 1) -> Counter:
    c = Counter()
    for doc in ngram_docs:
        c.update(doc)
    if min_count > 1:
        c = Counter({k:v for k,v in c.items() if v >= min_count})
    return c

# ---------- Métricas de asociación ----------
def pmi_pairs(pair_counts: Counter, unigram_counts: Counter, total_tokens: int) -> Dict[Tuple[str,str], float]:
    total_pairs = sum(pair_counts.values())
    Pw = {w: unigram_counts[w] / total_tokens for w in unigram_counts}
    res = {}
    for (w1, w2), c12 in pair_counts.items():
        p12 = c12 / total_pairs if total_pairs > 0 else 0.0
        p1 = Pw.get(w1, 0.0); p2 = Pw.get(w2, 0.0)
        if p1 > 0 and p2 > 0 and p12 > 0:
            res[(w1, w2)] = math.log(p12 / (p1 * p2), 2)
    return res

def log_likelihood_pairs(pair_counts: Counter,
                         left_marginal_counts: Counter,
                         right_marginal_counts: Counter,
                         universe_total: int) -> Dict[Tuple[str,str], float]:
    """
    G^2 para pares (A,B).
    - pair_counts: conteos conjuntos c(A,B)
    - left_marginal_counts: conteos de A
    - right_marginal_counts: conteos de B
    - universe_total: total de observaciones conjuntas posibles (p.ej., suma de pares)
    """
    res = {}
    N = max(universe_total, 1)
    for (a, b), k11 in pair_counts.items():
        k1_ = left_marginal_counts.get(a, 0)
        k_1 = right_marginal_counts.get(b, 0)
        k12 = max(k1_ - k11, 0)
        k21 = max(k_1 - k11, 0)
        k22 = max(N - (k11 + k12 + k21), 0)

        row1 = k11 + k12
        row2 = k21 + k22
        col1 = k11 + k21
        col2 = k12 + k22
        total = row1 + row2

        def E(r, c): return (r * c) / total if total > 0 else 0.0

        cells = [(k11, E(row1, col1)),
                 (k12, E(row1, col2)),
                 (k21, E(row2, col1)),
                 (k22, E(row2, col2))]
        g2 = 0.0
        for obs, exp in cells:
            if obs > 0 and exp > 0:
                g2 += 2.0 * obs * math.log(obs/exp)
        res[(a, b)] = g2
    return res

# ---- PMI y G2 para trigramas ----
def pmi_trigrams(tri_counts: Counter, unigram_counts: Counter, total_tokens: int) -> Dict[Tuple[str,str,str], float]:
    total_tris = sum(tri_counts.values())
    Pw = {w: unigram_counts[w] / max(total_tokens,1) for w in unigram_counts}
    res = {}
    for (w1, w2, w3), c123 in tri_counts.items():
        p123 = c123 / total_tris if total_tris > 0 else 0.0
        p1 = Pw.get(w1, 0.0); p2 = Pw.get(w2, 0.0); p3 = Pw.get(w3, 0.0)
        if p1 > 0 and p2 > 0 and p3 > 0 and p123 > 0:
            res[(w1, w2, w3)] = math.log(p123 / (p1 * p2 * p3), 2)
    return res

def g2_trigrams_via_prefix(tri_counts: Counter,
                           bi_prefix_counts: Counter,
                           unigram_counts: Counter) -> Dict[Tuple[str,str,str], float]:
    """
    G^2 aproximado para trigramas:
    Trata el trigram (w1,w2,w3) como asociación entre A=(w1,w2) y B=w3,
    usando:
      - c(A,B) = tri_counts[(w1,w2,w3)]
      - c(A)   = bi_prefix_counts[(w1,w2)]
      - c(B)   = unigram_counts[w3]
      - N      = total de trigramas (sum(tri_counts.values()))
    """
    N = max(sum(tri_counts.values()), 1)
    # Construimos un pseudo-par de claves (A,B)
    pair_counts = Counter()
    left_marg = Counter()
    right_marg = Counter()
    for (w1, w2, w3), c in tri_counts.items():
        A = (w1, w2)
        B = w3
        pair_counts[(A, B)] += c
        left_marg[A] += bi_prefix_counts.get((w1, w2), 0)
        right_marg[B] += unigram_counts.get(w3, 0)

    # Ahora aplicamos el mismo G^2 de pares, pero con claves compuestas
    res = {}
    for (A, B), k11 in pair_counts.items():
        k1_ = left_marg.get(A, 0)
        k_1 = right_marg.get(B, 0)
        k12 = max(k1_ - k11, 0)
        k21 = max(k_1 - k11, 0)
        k22 = max(N - (k11 + k12 + k21), 0)

        row1 = k11 + k12
        row2 = k21 + k22
        col1 = k11 + k21
        col2 = k12 + k22
        total = row1 + row2

        def E(r, c): return (r * c) / total if total > 0 else 0.0

        g2 = 0.0
        for obs, exp in [(k11, E(row1, col1)),
                         (k12, E(row1, col2)),
                         (k21, E(row2, col1)),
                         (k22, E(row2, col2))]:
            if obs > 0 and exp > 0:
                g2 += 2.0 * obs * math.log(obs/exp)
        # Mapear de vuelta a la tupla del trigram
        w1, w2 = A
        res[(w1, w2, B)] = g2
    return res


def burstiness_per_term(unigram_docs: List[List[str]]) -> Dict[str, float]:
    """
    Índice simple de burstiness por término: varianza / media de cuentas por documento (Index of Dispersion).
    Alto = término concentrado en pocos documentos; Bajo = distribuido de forma homogénea.
    """
    per_doc_counts: Dict[str, List[int]] = defaultdict(list)
    for doc in unigram_docs:
        c = Counter(doc)
        for term, cnt in c.items():
            per_doc_counts[term].append(cnt)
    scores = {}
    for term, counts in per_doc_counts.items():
        arr = np.array(counts, dtype=float)
        mu = arr.mean()
        var = arr.var(ddof=0)
        if mu > 0:
            scores[term] = var / mu
    return scores
    

# ---------- Orquestación ----------
def build_pipeline(
    raw_texts: List[str],
    pre_cfg: PreprocConfig = PreprocConfig(),
    ng_cfg: NgramConfig = NgramConfig(),
    pos_patterns_bi: Tuple[Tuple[str, str], ...] = (("ADJ","NOUN"), ("NOUN","NOUN")),
    pos_patterns_tri: Tuple[Tuple[str, str, str], ...] = (("ADJ","ADJ","NOUN"), ("NOUN","NOUN","NOUN"))
):
    """
    Devuelve DataFrames:
      - unigrams: df, freq, burstiness, tf_rel, tfidf_mean
      - bigrams:  bigram, freq, G2, PMI
      - bigrams_POS: bigram, freq, G2, PMI
      - trigrams: trigram, freq, G2, PMI
      - trigrams_POS: trigram, freq, G2, PMI
    """
    # 1) spaCy parse
    spacy_docs = spacy_tokenize_docs(raw_texts, pre_cfg)

    # 2) Unigrams (con o sin filtro POS)
    if pre_cfg.filter_pos_for_unigrams:
        unigram_docs = [tokens_from_doc(d, pre_cfg) for d in spacy_docs]
    else:
        unigram_docs = []
        for d in spacy_docs:
            toks = []
            for x in d:
                base = x["lemma"] if pre_cfg.lemmatize else x["text"]
                tok = clean_token(base, pre_cfg)
                if tok:
                    toks.append(tok)
            unigram_docs.append(toks)

    # 3) Tokens + POS para n-gramas con patrón
    tokpos_docs = [tokens_with_pos(d, pre_cfg) for d in spacy_docs]

    # 4) Frecuencias y DF
    tf, df = compute_tf_df(unigram_docs)
    total_tokens = sum(tf.values())
    n_docs = len(unigram_docs)

    # 5) TF-IDF (promedio por término)
    tfidf_df, _ = compute_tfidf(unigram_docs)

    # 6) N-gramas contiguos
    bigram_docs_contig, trigram_docs_contig = [], []
    if ng_cfg.use_contiguous:
        for toks in unigram_docs:
            bigram_docs_contig.append(list(contiguous_ngrams(toks, 2)))
            trigram_docs_contig.append(list(contiguous_ngrams(toks, 3)))

    # 7) N-gramas por patrón POS
    bigram_docs_pos, trigram_docs_pos = [], []
    for tp in tokpos_docs:
        words = [w for w,_ in tp]
        poses = [p for _,p in tp]
        if ng_cfg.use_contiguous:
            bi_seqs  = [ (tuple(words[i:i+2]), tuple(poses[i:i+2])) for i in range(len(words)-1) ]
            tri_seqs = [ (tuple(words[i:i+3]), tuple(poses[i:i+3])) for i in range(len(words)-2) ]
            bigram_docs_pos.append(pos_pattern_filter(bi_seqs, pos_patterns_bi))
            trigram_docs_pos.append(pos_pattern_filter(tri_seqs, pos_patterns_tri))

    # 8) Conteos
    bi_counts        = ngram_freqs(bigram_docs_contig, ng_cfg.min_count) if ng_cfg.use_contiguous else Counter()
    tri_counts       = ngram_freqs(trigram_docs_contig, ng_cfg.min_count) if ng_cfg.use_contiguous else Counter()
    bi_pos_counts    = ngram_freqs(bigram_docs_pos,    ng_cfg.min_count) if ng_cfg.use_contiguous else Counter()
    tri_pos_counts   = ngram_freqs(trigram_docs_pos,   ng_cfg.min_count) if ng_cfg.use_contiguous else Counter()

    # 9) Métricas para BIGRAMAS (pares)
    total_bigrams = sum(bi_counts.values())
    pmi_bi  = pmi_pairs(bi_counts, tf, total_tokens) if bi_counts else {}
    g2_bi   = log_likelihood_pairs(bi_counts, left_marginal_counts=tf, right_marginal_counts=tf,
                                   universe_total=max(total_bigrams,1)) if bi_counts else {}

    # 10) Métricas para TRIGRAMAS
    # PMI_3 directo con unigrams como márgenes
    pmi_tri = pmi_trigrams(tri_counts, tf, total_tokens) if tri_counts else {}
    # G2 via prefijo bigrama (w1 w2) -> w3
    g2_tri  = g2_trigrams_via_prefix(tri_counts, bi_counts, tf) if tri_counts else {}

    # 11) Burstiness unigrams
    burst_scores = burstiness_per_term(unigram_docs)

    # ---------- Empaquetar resultados ----------
    # Unigrams
    uni_df = pd.DataFrame({
        "term": list(tf.keys()),
        "freq": [tf[t] for t in tf],                # freq total
        "df":   [df.get(t,0) for t in tf],
        "burstiness": [burst_scores.get(t, 0.0) for t in tf],
    })
    uni_df["tf_rel"] = uni_df["freq"] / max(1, total_tokens)
    uni_df = uni_df.merge(tfidf_df, on="term", how="left").fillna({"tfidf_mean": 0.0})
    uni_df = uni_df[["term","df","freq","burstiness","tf_rel","tfidf_mean"]] \
                 .sort_values(["tfidf_mean","freq"], ascending=[False, False], ignore_index=True)

    # Bigrams contiguos
    def pack_bigram_df(bc: Counter) -> pd.DataFrame:
        if not bc:
            return pd.DataFrame(columns=["bigram","freq","G2","PMI"])
        rows = []
        for k,v in bc.items():
            rows.append({
                "bigram": " ".join(k),
                "freq": v,
                "G2": g2_bi.get(k, np.nan),
                "PMI": pmi_bi.get(k, np.nan),
            })
        df_ = pd.DataFrame(rows)
        return df_.sort_values(["G2","freq"], ascending=[False, False], ignore_index=True)

    bigram_df = pack_bigram_df(bi_counts)

    # Bigrams POS (usar métricas calculadas sobre el conjunto completo; subset por POS)
    def pack_bigram_pos_df(bpc: Counter) -> pd.DataFrame:
        if not bpc:
            return pd.DataFrame(columns=["bigram","freq","G2","PMI"])
        rows = []
        for k,v in bpc.items():
            rows.append({
                "bigram": " ".join(k),
                "freq": v,
                "G2": g2_bi.get(k, np.nan),
                "PMI": pmi_bi.get(k, np.nan),
            })
        df_ = pd.DataFrame(rows)
        return df_.sort_values(["G2","freq"], ascending=[False, False], ignore_index=True)

    bigram_pos_df = pack_bigram_pos_df(bi_pos_counts)

    # Trigrams contiguos
    def pack_trigram_df(tc: Counter) -> pd.DataFrame:
        if not tc:
            return pd.DataFrame(columns=["trigram","freq","G2","PMI"])
        rows = []
        for k,v in tc.items():
            rows.append({
                "trigram": " ".join(k),
                "freq": v,
                "G2": g2_tri.get(k, np.nan),
                "PMI": pmi_tri.get(k, np.nan),
            })
        df_ = pd.DataFrame(rows)
        # Orden sugerido: por G2 y luego frecuencia
        return df_.sort_values(["G2","freq"], ascending=[False, False], ignore_index=True)

    trigram_df = pack_trigram_df(tri_counts)

    # Trigrams POS (subset por patrón)
    def pack_trigram_pos_df(tpc: Counter) -> pd.DataFrame:
        if not tpc:
            return pd.DataFrame(columns=["trigram","freq","G2","PMI"])
        rows = []
        for k,v in tpc.items():
            rows.append({
                "trigram": " ".join(k),
                "freq": v,
                "G2": g2_tri.get(k, np.nan),
                "PMI": pmi_tri.get(k, np.nan),
            })
        df_ = pd.DataFrame(rows)
        return df_.sort_values(["G2","freq"], ascending=[False, False], ignore_index=True)

    trigram_pos_df = pack_trigram_pos_df(tri_pos_counts)

    results = {
        "unigrams": uni_df,
        "bigrams": bigram_df,
        "bigrams_POS": bigram_pos_df,
        "trigrams": trigram_df,
        "trigrams_POS": trigram_pos_df,
    }
    meta = {"n_docs": n_docs, "total_tokens": total_tokens}
    return results, meta


# ---------- Main ----------
if __name__ == "__main__":
    JSON1 = "../corpus/geojournals_articles.json"
    JSON2 = "../corpus/springerlink_data.json"

    docs, meta_df = unify_corpus(
        JSON1, JSON2,
        title_weight=2,
        drop_non_english=False,
        dedup=True
    )

    pre_cfg = PreprocConfig(
        lowercase=True,
        lemmatize=True,
        keep_pos_tags=("NOUN","PROPN","ADJ"),
        remove_stopwords=True,
        min_token_len=2,
    )

    ng_cfg = NgramConfig(
        min_count=1,
        window_size=3,
        use_contiguous=True,
        use_windowed=False
    )

    results, meta = build_pipeline(docs, pre_cfg, ng_cfg)

    print("Meta info:", meta)
    meta_df.to_csv("../metrics/out_meta.csv", index=False)
    results["unigrams"].to_csv("../metrics/out_unigrams.csv", index=False)
    results["bigrams"].to_csv("../metrics/out_bigrams.csv", index=False)
    results["bigrams_POS"].to_csv("../metrics/out_bigrams_pos.csv", index=False)
    results["trigrams"].to_csv("../metrics/out_trigrams.csv", index=False)
    results["trigrams_POS"].to_csv("../metrics/out_trigrams_pos.csv", index=False)