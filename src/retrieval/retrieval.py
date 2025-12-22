from __future__ import annotations

import math
from collections import Counter, defaultdict
from heapq import nlargest
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import nltk
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def tokenize(text: str) -> List[str]:
    try:
        return nltk.word_tokenize((text or "").lower())
    except LookupError:
        raise RuntimeError(
            "NLTK 'punkt' resource not found. Please run 'python -m src.ingest prepare' from the project root to download it."
        )


def _iter_doc_tokens(corpus: Mapping[str, Mapping[str, object]]) -> Tuple[List[str], List[List[str]]]:
    doc_ids = list(corpus.keys())
    tokenized: List[List[str]] = []
    for doc_id in doc_ids:
        payload = corpus[doc_id]
        tokens = payload.get("tokens") if isinstance(payload, Mapping) else None
        if isinstance(tokens, list) and tokens:
            tokenized.append([str(t) for t in tokens if str(t)])
            continue
        title = payload.get("title", "") if isinstance(payload, Mapping) else ""
        text = payload.get("text", "") if isinstance(payload, Mapping) else ""
        tokenized.append(tokenize(f"{title} {text}".strip()))
    return doc_ids, tokenized


def build_bm25(corpus: Mapping[str, Mapping[str, object]]) -> Tuple[BM25Okapi, List[str]]:
    doc_ids, tokenized_corpus = _iter_doc_tokens(corpus)
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, doc_ids


def retrieve_bm25_results(
    bm25: BM25Okapi,
    doc_ids: Sequence[str],
    queries: Mapping[str, str],
    *,
    top_k: int = 10,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for qid, query_text in tqdm(queries.items(), total=len(queries), desc="Processing queries"):
        tokenized_query = tokenize(query_text)
        scores = bm25.get_scores(tokenized_query)
        scored_docs = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        top_n = scored_docs[:top_k]
        results[str(qid)] = {doc_id: float(score) for doc_id, score in top_n}
    return results


def run_bm25_baseline(
    corpus: Mapping[str, Mapping[str, object]],
    queries: Mapping[str, str],
    *,
    top_k: int = 10,
) -> Dict[str, Dict[str, float]]:
    print("Building BM25 index...")
    bm25, doc_ids = build_bm25(corpus)

    print("Retrieving results...")
    return retrieve_bm25_results(bm25, doc_ids, queries, top_k=top_k)


TfidfIndex = Tuple[Dict[str, float], Dict[str, float], Dict[str, List[Tuple[str, float]]]]


def build_tfidf(corpus: Mapping[str, Mapping[str, object]]) -> Tuple[TfidfIndex, List[str]]:
    doc_ids, tokenized_corpus = _iter_doc_tokens(corpus)
    n_docs = len(doc_ids)

    doc_freq: Counter[str] = Counter()
    doc_term_counts: List[Counter[str]] = []

    for tokens in tqdm(tokenized_corpus, desc="Computing TF/DF", unit="doc"):
        tf = Counter(tokens)
        doc_term_counts.append(tf)
        doc_freq.update(tf.keys())

    idf: Dict[str, float] = {term: math.log((n_docs + 1.0) / (df + 1.0)) + 1.0 for term, df in doc_freq.items()}

    postings: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    doc_norms: Dict[str, float] = {}

    for doc_id, tf in tqdm(zip(doc_ids, doc_term_counts), total=n_docs, desc="Building TF-IDF index", unit="doc"):
        norm_sq = 0.0
        for term, freq in tf.items():
            weight = (1.0 + math.log(freq)) * idf.get(term, 0.0)
            if weight == 0.0:
                continue
            postings[term].append((doc_id, weight))
            norm_sq += weight * weight
        doc_norms[doc_id] = math.sqrt(norm_sq) if norm_sq > 0.0 else 0.0

    return (idf, doc_norms, dict(postings)), doc_ids


def retrieve_tfidf_results(
    index: TfidfIndex,
    queries: Mapping[str, str],
    *,
    top_k: int = 10,
) -> Dict[str, Dict[str, float]]:
    idf, doc_norms, postings = index

    results: Dict[str, Dict[str, float]] = {}
    for qid, query_text in tqdm(queries.items(), total=len(queries), desc="Processing queries"):
        tokens = tokenize(query_text)
        tf = Counter(tokens)

        q_weights: Dict[str, float] = {}
        q_norm_sq = 0.0
        for term, freq in tf.items():
            weight = (1.0 + math.log(freq)) * idf.get(term, 0.0)
            if weight == 0.0:
                continue
            q_weights[term] = weight
            q_norm_sq += weight * weight

        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0.0 else 0.0
        if q_norm == 0.0:
            results[str(qid)] = {}
            continue

        scores: MutableMapping[str, float] = defaultdict(float)
        for term, q_weight in q_weights.items():
            for doc_id, d_weight in postings.get(term, []):
                scores[doc_id] += q_weight * d_weight

        scored: List[Tuple[str, float]] = []
        for doc_id, numerator in scores.items():
            denom = q_norm * (doc_norms.get(doc_id, 0.0) or 0.0)
            if denom == 0.0:
                continue
            scored.append((doc_id, float(numerator / denom)))

        top = nlargest(top_k, scored, key=lambda x: x[1])
        results[str(qid)] = {doc_id: score for doc_id, score in top}

    return results


def run_tfidf_baseline(
    corpus: Mapping[str, Mapping[str, object]],
    queries: Mapping[str, str],
    *,
    top_k: int = 10,
) -> Dict[str, Dict[str, float]]:
    print("Building TF-IDF index...")
    index, _ = build_tfidf(corpus)

    print("Retrieving results...")
    return retrieve_tfidf_results(index, queries, top_k=top_k)


def run_baseline(
    corpus: Mapping[str, Mapping[str, object]],
    queries: Mapping[str, str],
    retrieval: str = "bm25",
    *,
    top_k: int = 10,
) -> Dict[str, Dict[str, float]]:
    retrieval_norm = (retrieval or "").lower()
    if retrieval_norm == "bm25":
        return run_bm25_baseline(corpus, queries, top_k=top_k)
    if retrieval_norm in {"tfidf", "tf-idf"}:
        return run_tfidf_baseline(corpus, queries, top_k=top_k)
    raise ValueError(f"Unsupported retrieval backend '{retrieval}'.")