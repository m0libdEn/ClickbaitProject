import time
import pandas as pd
from clickbait import analyze_headline, nlp
from classifier import predict_clickbait
from rewrite import rewrite_headline


def inspect_headline(headline: str):
    t0 = time.perf_counter()

    # 1. Linguistic features
    feats = analyze_headline(headline)
    lex = feats["lexical"]
    syn = feats["syntactic"]
    sent = feats["sentiment"]

    # 2. Classification
    clf_res = predict_clickbait(headline)
    t_clf = time.perf_counter() - t0

    # 3. Entities extracted by spaCy
    doc = nlp(headline)
    raw_ents = [f"{ent.text} ({ent.label_})" for ent in doc.ents]

    print("=" * 75)
    print(f"INPUT      : {headline}")
    print(f"VERDICT    : {'CLICKBAIT (1)' if clf_res['verdict'] == 1 else 'NORMAL (0)'} | "
          f"Prob: {clf_res['probability']} | Clf time: {t_clf*1000:.1f} ms")
    print(f"TRIGGERS   : words={lex['word_count']}, superlatives={lex['superlatives']}, "
          f"fwd_ref={lex['forward_references']}, caps={lex['caps_ratio']}, "
          f"excl={lex['exclamation_marks']}, quest={lex['question_marks']}, "
          f"pronouns={syn['pronoun_density']}, intensity={sent['intensity']}")
    print(f"SPACY ENTS : {raw_ents if raw_ents else 'None'}")

    # 4. Rewriting (if clickbait)
    if clf_res["verdict"] == 1:
        t1 = time.perf_counter()
        rw_res = rewrite_headline(headline)
        t_rw = time.perf_counter() - t1
        print(f"REWRITTEN  : {rw_res['rewritten']}")
        print(f"QUALITY    : Sim={rw_res['similarity']} | "
              f"Entities kept={rw_res['entities_preserved']} | LLM time={t_rw:.2f} s")
    else:
        print("REWRITTEN  : [Skipped - Not Clickbait]")


def run_dataset_sample(n_samples: int = 10):
    df = pd.read_csv("notClickbait.csv", header=None)

    # Если первая строка — это название колонки (например, "headline" или "title"), пропускаем её
    first_val = str(df.iloc[0, 1]).strip().lower()
    if first_val in {"headline", "title", "text"}:
        df = df.iloc[1:]

    headlines = df[1].dropna().astype(str).sample(n=min(n_samples, len(df))).tolist()

    for h in headlines:
        inspect_headline(h.strip())

if __name__ == "__main__":
    run_dataset_sample(n_samples=10)