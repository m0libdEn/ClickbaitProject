Clone the repository from GitHub and navigate into the project folder in your terminal.
Create a virtual environment:

python -m venv .venv

Activate the virtual environment:
Windows (PowerShell / CMD): .venv\Scripts\activate
macOS / Linux: source .venv/bin/activate
Install required Python libraries (this downloads PyTorch and Sentence-Transformers, requiring approximately 2.5 GB of disk space):

pip install -r requirements.txt

Download the spaCy English language model:

python -m spacy download en_core_web_sm

Run the test script:

python clickbait.py
