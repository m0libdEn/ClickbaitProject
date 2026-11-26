import ollama
import numpy as np
from clickbait import analyze_headline, embedder
from classifier import predict_clickbait


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def rewrite_headline(headline: str, min_similarity: float = 0.62, max_attempts: int = 3) -> dict:
    classification = predict_clickbait(headline)
    if classification["verdict"] == 0:
        return {
            "original": headline,
            "is_clickbait": False,
            "verdict": 0,
            "probability": classification["probability"],
            "rewritten": headline,
            "similarity": 1.0,
            "entities_preserved": True
        }

    features = analyze_headline(headline)
    entities = [ent["text"] for ent in features["syntactic"]["named_entities"]]
    orig_vec = embedder.encode(headline)

    best_rewrite = headline
    best_sim = 0.0
    entities_ok = False

    for attempt in range(max_attempts):
        prompt = f"Original clickbait headline: {headline}\n"
        if entities:
            prompt += f"Mandatory entities to preserve: {', '.join(entities)}\n"
        prompt += "Rewrite into one objective news headline:"

        response = ollama.chat(
            model="neutral-editor",
            messages=[{"role": "user", "content": prompt}]
        )
        candidate = response["message"]["content"].strip().strip('"').strip("'")

        cand_vec = embedder.encode(candidate)
        sim = round(_cosine_similarity(orig_vec, cand_vec), 3)
        all_entities_present = all(ent.lower() in candidate.lower() for ent in entities)

        if sim > best_sim:
            best_rewrite = candidate
            best_sim = sim
            entities_ok = all_entities_present

        if sim >= min_similarity and all_entities_present:
            break

    return {
        "original": headline,
        "is_clickbait": True,
        "verdict": 1,
        "probability": classification["probability"],
        "rewritten": best_rewrite,
        "similarity": best_sim,
        "entities_preserved": entities_ok
    }


if __name__ == "__main__":
    test_headlines = [
        "You Won't Believe What Elon Musk Just Announced About Tesla Batteries!",
        "15 Shocking Secrets Miami Doctors Never Tell You About Sleep",
        "Federal Reserve Holds Interest Rates Steady at 5.25 Percent"
    ]

    for h in test_headlines:
        result = rewrite_headline(h)
        print("-" * 60)
        print(f"Original  : {result['original']}")
        print(f"Verdict   : {result['verdict']} (p={result['probability']})")
        print(f"Rewritten : {result['rewritten']}")
        print(f"Similarity: {result['similarity']} | Entities kept: {result['entities_preserved']}")