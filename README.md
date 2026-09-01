# Customer Support NLP Pipeline — Streamlit Dashboard

Streamlit app showcasing a dual-model customer support pipeline: a TF-IDF +
Logistic Regression intent classifier (99.67% accuracy) and a LLaMA-2 QLoRA
fine-tuned response generator (ROUGE-L 0.6840), trained on the Bitext
Customer Service dataset (26,872 samples).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files the app needs

- `app.py` — entry point
- `requirements.txt` — dependencies
- `eda_dashboard.png`, `baseline_confusion_matrix.png`
- `model_comparison_results.csv`
- `bitext_preprocessed_prompts.csv`

## Large file note

`bitext_preprocessed_prompts.csv` (~68MB) is under GitHub's 100MB per-file
push limit, so it's pushed as-is via `git push` from the terminal — no
compression needed. (GitHub's web upload UI caps at 25MB, but that limit
doesn't apply to command-line pushes.)

## Deployment

Free hosting via [Streamlit Community Cloud](https://share.streamlit.io):
push this folder to a GitHub repo, then connect the repo and set `app.py`
as the entry point.
