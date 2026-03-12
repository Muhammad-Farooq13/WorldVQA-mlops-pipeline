"""
train_demo.py — Build a demo bundle for the WorldVQA Streamlit dashboard.

Uses entirely synthetic VQA-style question/answer pairs so no network
access (HuggingFace dataset download) is required.  The bundle is
saved to  models/demo_bundle.pkl  and loaded by streamlit_app.py.
"""

from __future__ import annotations

import pathlib
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Synthetic VQA data ─────────────────────────────────────────────────────────
CATEGORIES = ["geography", "science", "history", "culture", "sports"]

TEMPLATES: dict[str, list[str]] = {
    "geography": [
        "What is the capital of {}?",
        "Which country is {} located in?",
        "What river flows through {}?",
        "What ocean borders {}?",
        "What is the largest city in {}?",
        "What mountain range is found in {}?",
        "Which continent does {} belong to?",
        "What is the population of {}?",
        "How many countries share a border with {}?",
        "What language is spoken most in {}?",
    ],
    "science": [
        "What is the chemical symbol for {}?",
        "What is the atomic number of {}?",
        "Which planet is known for {}?",
        "What organ in the human body is responsible for {}?",
        "What is the speed of {} in a vacuum?",
        "Who discovered {}?",
        "What unit measures {}?",
        "What is the formula for {}?",
        "Which scientist first described {}?",
        "How does {} affect the human body?",
    ],
    "history": [
        "When did the {} war begin?",
        "Who was the leader of {} during the revolution?",
        "What year was {} founded?",
        "Which empire controlled {} in ancient times?",
        "What caused the downfall of {}?",
        "Who signed the {} treaty?",
        "Where was the {} battle fought?",
        "What was the significance of {} in world history?",
        "Which dynasty ruled {} for centuries?",
        "How did {} change the course of history?",
    ],
    "culture": [
        "What traditional food is {} famous for?",
        "Which festival celebrates {} every year?",
        "What instrument is associated with {} music?",
        "Who wrote the famous {} novel?",
        "What art movement influenced {} painters?",
        "What clothes are traditionally worn during {}?",
        "Which religion is most widely practised in {}?",
        "What language gave rise to {}?",
        "Who is considered the national hero of {}?",
        "What is the most popular sport in {}?",
    ],
    "sports": [
        "Who holds the record for {} in the Olympics?",
        "Which team has won the most {} championships?",
        "When was {} officially recognised as an Olympic sport?",
        "How many players are on a {} team?",
        "What is the maximum score in {}?",
        "Who invented the game of {}?",
        "Which country dominates {} at the world level?",
        "What equipment is required to play {}?",
        "How long does a standard {} match last?",
        "What is the governing body for {} worldwide?",
    ],
}

FILLERS: dict[str, list[str]] = {
    "geography": [
        "France", "Brazil", "Egypt", "India", "Japan", "Argentina",
        "Nigeria", "Canada", "Germany", "Australia", "Mexico", "China",
        "Russia", "South Africa", "Italy", "Spain", "Peru", "Thailand",
        "Turkey", "Sweden",
    ],
    "science": [
        "oxygen", "carbon", "gravity", "photosynthesis", "electricity",
        "hydrogen", "DNA", "mitosis", "thermodynamics", "light",
        "sound", "magnetism", "evolution", "relativity", "fusion",
        "entropy", "osmosis", "radioactivity", "momentum", "acceleration",
    ],
    "history": [
        "World War II", "the Roman Empire", "the French Revolution",
        "the Ottoman Empire", "the Cold War", "the Renaissance",
        "the Mongol Empire", "the American Civil War", "the Silk Road",
        "the Industrial Revolution", "the British Empire", "the Ming Dynasty",
        "the Greek city states", "the Spanish Armada", "the Berlin Wall",
        "the Mughal Empire", "the Viking Age", "the Age of Exploration",
        "the Crusades", "the Black Death",
    ],
    "culture": [
        "Japan", "Mexico", "India", "Italy", "Brazil", "France",
        "China", "Greece", "Turkey", "Egypt", "Korea", "Spain",
        "Ghana", "Iran", "Peru", "Russia", "Ethiopia", "Argentina",
        "Vietnam", "Nigeria",
    ],
    "sports": [
        "football", "basketball", "cricket", "tennis", "swimming",
        "athletics", "boxing", "cycling", "rowing", "gymnastics",
        "volleyball", "baseball", "golf", "rugby", "table tennis",
        "badminton", "wrestling", "judo", "archery", "fencing",
    ],
}

ROWS_PER_CAT = 120  # 600 total rows


def _make_rows() -> pd.DataFrame:
    rows: list[dict] = []
    for cat in CATEGORIES:
        templates = TEMPLATES[cat]
        fillers = FILLERS[cat]
        for i in range(ROWS_PER_CAT):
            tmpl = templates[i % len(templates)]
            filler = fillers[i % len(fillers)]
            question = tmpl.format(filler)
            rows.append({"question": question, "category": cat})
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


def main() -> None:
    print("Building synthetic WorldVQA demo dataset …")
    df = _make_rows()
    print(f"  {len(df)} rows across {df['category'].nunique()} categories")

    X, y = df["question"], df["category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print("Training TF-IDF + LogisticRegression …")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=SEED)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=CATEGORIES)

    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  F1 macro : {f1_macro:.4f}")

    # ── Per-class metrics ────────────────────────────────────────────────────
    per_class: list[dict] = []
    for cat in CATEGORIES:
        r = report[cat]
        per_class.append({
            "category": cat,
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1-score"], 4),
            "support": int(r["support"]),
        })

    # ── Top TF-IDF terms per class ───────────────────────────────────────────
    feature_names = vectorizer.get_feature_names_out()
    top_terms: dict[str, list[str]] = {}
    for i, cat in enumerate(model.classes_):
        coef = model.coef_[i]
        top_idx = coef.argsort()[-15:][::-1]
        top_terms[cat] = [feature_names[j] for j in top_idx]

    # ── Sample predictions ───────────────────────────────────────────────────
    sample_qs = [
        "What is the capital of France?",
        "Who discovered gravity?",
        "When did World War II begin?",
        "What traditional food is Japan famous for?",
        "Who holds the record for swimming in the Olympics?",
    ]
    sample_vec = vectorizer.transform(sample_qs)
    sample_preds = model.predict(sample_vec).tolist()
    sample_probs = model.predict_proba(sample_vec).max(axis=1).tolist()

    # ── Bundle ───────────────────────────────────────────────────────────────
    bundle = {
        "vectorizer": vectorizer,
        "model": model,
        "label_names": CATEGORIES,
        "dataset_stats": {
            "total_rows": len(df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "num_classes": len(CATEGORIES),
            "vocabulary_size": len(vectorizer.vocabulary_),
        },
        "category_counts": df["category"].value_counts().to_dict(),
        "results": {
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1_macro, 4),
            "per_class": per_class,
        },
        "confusion_matrix_data": {
            "matrix": cm.tolist(),
            "labels": CATEGORIES,
        },
        "top_tfidf_terms": top_terms,
        "sample_questions": [
            {"question": q, "predicted": p, "confidence": round(c, 4)}
            for q, p, c in zip(sample_qs, sample_preds, sample_probs)
        ],
        "train_questions": X_train.tolist(),
        "train_labels": y_train.tolist(),
    }

    out_path = pathlib.Path("models") / "demo_bundle.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"  Bundle saved → {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
