import json
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer

# 1. Load NLP models into memory once at startup
nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Seed lexicons from Section 4.1
SUPERLATIVES = {"amazing", "shocking", "incredible", "unbelievable", "insane", "best", "worst"}
FORWARD_REFS = {"this", "these", "here", "here's", "what", "why", "who", "secret", "trick"}


def analyze_headline(headline: str) -> dict:
    doc = nlp(headline)
    words = [token.text for token in doc if not token.is_punct]
    num_words = max(len(words), 1)

    # --- A. Lexical Features (v_lex) ---
    lower_words = [w.lower() for w in words]
    superlative_count = sum(1 for w in lower_words if w in SUPERLATIVES)
    forward_ref_count = sum(1 for w in lower_words if w in FORWARD_REFS)
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / num_words
    template_tokens = ["<NUM>" if token.like_num else token.text.lower() for token in doc]

    # --- B. Syntactic & POS Features (v_pos) ---
    pos_tags = [token.pos_ for token in doc]
    pronoun_density = pos_tags.count("PRON") / num_words
    modal_density = sum(1 for token in doc if token.tag_ == "MD") / num_words
    entities = [ent.text for ent in doc.ents]

    # --- C. Sentiment & Emotion Intensity (v_sent) ---
    sentiment = vader.polarity_scores(headline)
    emotional_intensity = abs(sentiment["compound"])

    # --- D. Dense Semantic Embedding (v_dense) ---
    embedding = embedder.encode(headline)

    return {
        "original_headline": headline,
        "template": " ".join(template_tokens),
        "lexical": {
            "word_count": num_words,
            "superlatives": superlative_count,
            "forward_references": forward_ref_count,
            "caps_ratio": round(caps_ratio, 2),
            "exclamation_marks": headline.count("!"),
            "question_marks": headline.count("?")
        },
        "syntactic": {
            "pos_sequence": pos_tags,
            "pronoun_density": round(pronoun_density, 2),
            "modal_verb_density": round(modal_density, 2),
            "named_entities": entities
        },
        "sentiment": {
            "intensity": round(emotional_intensity, 3),
            "raw_compound": sentiment["compound"]
        },
        "embedding_dimensions": len(embedding)
    }


if __name__ == "__main__":
    test_title = "You Won't Believe What This City Did With Its Old Subway Cars"
    result = analyze_headline(test_title)
    print(json.dumps(result, indent=2))