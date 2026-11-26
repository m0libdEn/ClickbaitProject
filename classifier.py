import numpy as np
from clickbait import analyze_headline, embedder

CLICKBAIT_ANCHORS = [
    "You won't believe what happened next",
    "This one simple trick will change your life forever",
    "The shocking reason why nobody talks about this",
    "Things you didn't know until today"
]
ANCHOR_EMBEDDINGS = embedder.encode(CLICKBAIT_ANCHORS)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def predict_clickbait(headline: str, threshold: float = 0.5) -> dict:
    features = analyze_headline(headline)
    lex = features["lexical"]
    syn = features["syntactic"]
    sent = features["sentiment"]

    headline_vec = embedder.encode(headline)
    max_semantic_sim = max(_cosine_similarity(headline_vec, anchor) for anchor in ANCHOR_EMBEDDINGS)

    raw_score = 0.0
    raw_score += min(lex["forward_references"] * 0.22, 0.44)
    raw_score += min(lex["superlatives"] * 0.25, 0.50)
    raw_score += 0.20 if (lex["exclamation_marks"] > 0 or lex["question_marks"] > 0) else 0.0
    raw_score += min(lex["caps_ratio"] * 0.50, 0.25)
    raw_score += min(syn["pronoun_density"] * 0.60, 0.25)
    raw_score += min(syn["modal_verb_density"] * 0.60, 0.15)
    raw_score += sent["intensity"] * 0.20
    raw_score += max(0.0, (max_semantic_sim - 0.35)) * 0.65

    probability = round(min(max(raw_score, 0.0), 1.0), 3)
    verdict_int = 1 if probability >= threshold else 0

    return {
        "headline": headline,
        "verdict": verdict_int,
        "is_clickbait": bool(verdict_int),
        "probability": probability,
        "threshold": threshold
    }