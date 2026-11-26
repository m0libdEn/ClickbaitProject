# Clickbait Detection & Neutralization Pipeline

Hybrid NLP pipeline combining linguistic feature extraction (393 features: lexical, syntactic, VADER sentiment, and `all-MiniLM-L6-v2` embeddings) with a Logistic Regression classifier (96.4% accuracy) and a local `Llama 3.2 3B` LLM via Ollama for factual headline rewriting.

## Project Structure

- `clickbait.py` — Feature extraction module (lexical, syntactic, sentiment, and dense embeddings).
- `train.py` — Model training script on the Kaggle clickbait dataset (`clickbait_data.csv`).
- `classifier.py` — Inference wrapper (`predict_clickbait`) using the trained `clickbait_model.joblib`.
- `rewrite.py` — End-to-end detection and factual rewriting pipeline with entity preservation and cosine similarity validation.
- `debug.py` — Diagnostic script for inspecting linguistic triggers, NER entities, and latency on sample datasets.
- `Modelfile` — Custom Ollama configuration for the `neutral-editor` model.
- `clickbait_model.joblib` — Pre-trained Logistic Regression model weights.
- `requirements.txt` — Python package dependencies.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally

## Setup & Installation

1. Clone the repository and create a virtual environment:

   git clone <your-repo-url>
   cd ClickbaitProject
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

2. Install Python dependencies and the spaCy English model:

   pip install -r requirements.txt
   python -m spacy download en_core_web_sm

3. Pull the base LLM and build the custom Ollama editor model:

   ollama pull llama3.2:3b
   ollama create neutral-editor -f Modelfile

## How to Verify Everything Works

Run the end-to-end rewrite test on sample headlines:

   python rewrite.py

Expected output:
- Clickbait headlines receive `Verdict: 1`, a neutralized rewrite, cosine similarity `>= 0.62`, and `Entities kept: True`.
- Normal news headlines receive `Verdict: 0` and are skipped without modification.

## Optional: Running Diagnostics & Retraining

- To inspect extracted features, spaCy entities, and model latency on custom headlines or CSV files:

   python debug.py

- To retrain the classifier from scratch (requires `clickbait_data.csv` in the root folder):

   python train.py