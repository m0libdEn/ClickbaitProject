import numpy as np
import joblib
from clickbait import analyze_headline, embedder

model = joblib.load("clickbait_model.joblib")


def predict_clickbait(headline: str, threshold: float = 0.5) -> dict:
    feats = analyze_headline(headline)
    lex = feats["lexical"]
    syn = feats["syntactic"]
    sent = feats["sentiment"]

    numeric_row = [
        lex["word_count"],
        lex["superlatives"],
        lex["forward_references"],
        lex["caps_ratio"],
        lex["exclamation_marks"],
        lex["question_marks"],
        syn["pronoun_density"],
        syn["modal_verb_density"],
        sent["intensity"]
    ]

    embed_text = headline.capitalize() if lex["caps_ratio"] > 0.5 else headline
    headline_vec = embedder.encode(headline)
    full_row = np.concatenate([numeric_row, headline_vec]).reshape(1, -1)

    probability = round(float(model.predict_proba(full_row)[0][1]), 3)
    verdict_int = 1 if probability >= threshold else 0

    return {
        "headline": headline,
        "verdict": verdict_int,
        "is_clickbait": bool(verdict_int),
        "probability": probability,
        "threshold": threshold
    }