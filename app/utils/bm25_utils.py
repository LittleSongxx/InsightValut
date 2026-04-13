import math
from collections import Counter
from typing import Any, Callable, Dict, List, Sequence, Tuple

import regex

_TOKEN_PATTERN = regex.compile(r"\p{Han}+|[\p{Letter}\p{Number}]+", regex.VERSION1)
_ALPHA_NUM_PATTERN = regex.compile(r"\p{Letter}+|\p{Number}+", regex.VERSION1)
_HAN_PATTERN = regex.compile(r"\p{Han}+", regex.VERSION1)


def tokenize_text(text: str) -> List[str]:
    normalized = (text or "").strip().lower()
    if not normalized:
        return []

    tokens: List[str] = []
    for raw in _TOKEN_PATTERN.findall(normalized):
        if _HAN_PATTERN.fullmatch(raw):
            chars = [ch for ch in raw if ch.strip()]
            if not chars:
                continue
            tokens.append(raw)
            tokens.extend(chars)
            if len(chars) > 1:
                tokens.extend("".join(chars[i : i + 2]) for i in range(len(chars) - 1))
            continue

        parts = _ALPHA_NUM_PATTERN.findall(raw)
        if not parts:
            continue
        tokens.extend(parts)
        if len(parts) > 1:
            tokens.append("".join(parts))

    return tokens


def score_corpus_bm25(
    query_tokens: Sequence[str],
    corpus_tokens: Sequence[Sequence[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    if not query_tokens or not corpus_tokens:
        return []

    doc_term_freqs = [Counter(tokens) for tokens in corpus_tokens]
    doc_lengths = [len(tokens) for tokens in corpus_tokens]
    if not any(doc_lengths):
        return [0.0 for _ in corpus_tokens]

    avgdl = sum(doc_lengths) / max(len(doc_lengths), 1)
    document_frequency: Counter[str] = Counter()
    for term_freq in doc_term_freqs:
        document_frequency.update(term_freq.keys())

    unique_query_terms = list(dict.fromkeys(query_tokens))
    total_docs = len(corpus_tokens)
    scores: List[float] = []
    for doc_length, term_freq in zip(doc_lengths, doc_term_freqs):
        score = 0.0
        length_norm = k1 * (1 - b + b * (doc_length / avgdl if avgdl > 0 else 0.0))
        for term in unique_query_terms:
            tf = term_freq.get(term, 0)
            if tf <= 0:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1.0 + ((total_docs - df + 0.5) / (df + 0.5)))
            score += idf * ((tf * (k1 + 1.0)) / (tf + length_norm))
        scores.append(score)
    return scores


def rank_documents_bm25(
    query_text: str,
    documents: Sequence[Dict[str, Any]],
    *,
    text_getter: Callable[[Dict[str, Any]], str],
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Tuple[Dict[str, Any], float]]:
    if not query_text or not documents:
        return []

    query_tokens = tokenize_text(query_text)
    if not query_tokens:
        return []

    corpus_tokens = [tokenize_text(text_getter(doc)) for doc in documents]
    scores = score_corpus_bm25(query_tokens, corpus_tokens, k1=k1, b=b)

    ranked = [
        (doc, float(score))
        for doc, score in zip(documents, scores)
        if score and score > 0
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    if top_k is not None and top_k >= 0:
        return ranked[:top_k]
    return ranked
