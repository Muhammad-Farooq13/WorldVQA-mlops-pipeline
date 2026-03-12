"""
streamlit_app.py — WorldVQA MLOps Dashboard
5-tab Streamlit dashboard powered by models/demo_bundle.pkl
"""

from __future__ import annotations

import pathlib
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WorldVQA Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BUNDLE_PATH = pathlib.Path("models/demo_bundle.pkl")

CATEGORY_ICONS = {
    "geography": "🌍",
    "science": "🔬",
    "history": "📜",
    "culture": "🎭",
    "sports": "🏆",
}

PLACEHOLDER_QUESTIONS = {
    "geography": "What is the capital of France?",
    "science": "What is the chemical symbol for oxygen?",
    "history": "When did the World War II war begin?",
    "culture": "What traditional food is Japan famous for?",
    "sports": "Who holds the record for swimming in the Olympics?",
}


@st.cache_resource(show_spinner="Loading model bundle …")
def load_bundle(path: pathlib.Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _color_metric(val: float, metric: str = "accuracy") -> str:
    if val >= 0.9:
        return "green"
    if val >= 0.75:
        return "orange"
    return "red"


# ── Load bundle ────────────────────────────────────────────────────────────────
if not BUNDLE_PATH.exists():
    st.error(
        "Demo bundle not found. Run `python train_demo.py` first to generate "
        "`models/demo_bundle.pkl`."
    )
    st.stop()

bundle = load_bundle(BUNDLE_PATH)
vectorizer = bundle["vectorizer"]
model = bundle["model"]
label_names: list[str] = bundle["label_names"]
stats = bundle["dataset_stats"]
cat_counts = bundle["category_counts"]
results = bundle["results"]
per_class_df = pd.DataFrame(results["per_class"])
cm_data = bundle["confusion_matrix_data"]
top_terms = bundle["top_tfidf_terms"]
sample_qs = bundle["sample_questions"]

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 Overview",
    "📊 Model Results",
    "📈 Analytics",
    "⚙️ Pipeline & API",
    "🔮 Classify",
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("🌍 WorldVQA — World Knowledge Question Answering")
    st.markdown(
        "A **TF-IDF + Logistic Regression** baseline for classifying "
        "world-knowledge questions into five semantic categories."
    )

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Questions", f"{stats['total_rows']:,}")
    c2.metric("Training Set", f"{stats['train_rows']:,}")
    c3.metric("Test Set", f"{stats['test_rows']:,}")
    c4.metric("Categories", stats["num_classes"])
    c5.metric("Vocabulary Size", f"{stats['vocabulary_size']:,}")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Category Distribution")
        cat_df = pd.DataFrame(
            {"Category": list(cat_counts.keys()),
             "Count": list(cat_counts.values())}
        ).sort_values("Count", ascending=False)
        fig_bar = px.bar(
            cat_df, x="Category", y="Count", color="Category",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text="Count",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(showlegend=False, height=350,
                              plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.subheader("Sample Questions per Category")
        rows = []
        for cat in label_names:
            icon = CATEGORY_ICONS.get(cat, "")
            rows.append({"Category": f"{icon} {cat}", "Sample": PLACEHOLDER_QUESTIONS[cat]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Sample Predictions from Demo Bundle")
    sample_df = pd.DataFrame(sample_qs)
    sample_df["confidence"] = (sample_df["confidence"] * 100).round(1).astype(str) + "%"
    sample_df["predicted"] = sample_df["predicted"].apply(
        lambda c: f"{CATEGORY_ICONS.get(c, '')} {c}"
    )
    sample_df.columns = ["Question", "Predicted Category", "Confidence"]
    st.dataframe(sample_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Model Results
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("📊 Model Performance")

    acc = results["accuracy"]
    f1 = results["f1_macro"]

    ka, kb, kc = st.columns(3)
    ka.metric("Test Accuracy", f"{acc:.2%}", delta=f"{acc - 0.5:.2%} vs random")
    kb.metric("Macro F1", f"{f1:.4f}")
    kc.metric("Classes", str(stats["num_classes"]))

    st.markdown("---")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Per-Class Metrics")
        styled_df = per_class_df.copy()
        styled_df["category"] = styled_df["category"].apply(
            lambda c: f"{CATEGORY_ICONS.get(c, '')} {c}"
        )
        styled_df.columns = ["Category", "Precision", "Recall", "F1", "Support"]
        st.dataframe(
            styled_df.style.background_gradient(subset=["Precision", "Recall", "F1"],
                                                cmap="RdYlGn", vmin=0, vmax=1),
            use_container_width=True, hide_index=True,
        )

    with col_r:
        st.subheader("Precision / Recall / F1 by Category")
        melted = per_class_df.melt(
            id_vars="category", value_vars=["precision", "recall", "f1"],
            var_name="Metric", value_name="Score"
        )
        fig_grp = px.bar(
            melted, x="category", y="Score", color="Metric", barmode="group",
            color_discrete_sequence=["#4FC3F7", "#81C784", "#FFB74D"],
        )
        fig_grp.update_layout(height=380, plot_bgcolor="rgba(0,0,0,0)",
                              yaxis_range=[0, 1.05])
        st.plotly_chart(fig_grp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Analytics
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("📈 Analytics")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm_matrix = np.array(cm_data["matrix"])
    cm_labels = cm_data["labels"]
    cm_norm = cm_matrix.astype(float) / cm_matrix.sum(axis=1, keepdims=True)

    fig_cm = go.Figure(go.Heatmap(
        z=cm_norm,
        x=[f"{CATEGORY_ICONS.get(l, '')} {l}" for l in cm_labels],
        y=[f"{CATEGORY_ICONS.get(l, '')} {l}" for l in cm_labels],
        colorscale="Blues",
        text=cm_matrix,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig_cm.update_layout(
        xaxis_title="Predicted", yaxis_title="Actual",
        height=420, plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    # Top TF-IDF terms
    st.subheader("Top TF-IDF Terms per Category")
    chosen_cat = st.selectbox(
        "Select category", label_names,
        format_func=lambda c: f"{CATEGORY_ICONS.get(c, '')} {c}",
    )
    terms = top_terms.get(chosen_cat, [])
    terms_df = pd.DataFrame({"Term": terms, "Rank": range(1, len(terms) + 1)})
    col_terms, col_bar = st.columns([1, 2])
    with col_terms:
        st.dataframe(terms_df, use_container_width=True, hide_index=True)
    with col_bar:
        fig_terms = px.bar(
            terms_df.head(10), x="Term", y="Rank",
            orientation="v", color="Rank",
            color_continuous_scale="Blues_r",
        )
        fig_terms.update_layout(height=320, yaxis_autorange="reversed",
                                plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_terms, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Pipeline & API
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.title("⚙️ Pipeline & API")

    col_pipe, col_api = st.columns([1, 1])

    with col_pipe:
        st.subheader("MLOps Pipeline Steps")
        steps = [
            ("1. Data Loading", "Load WorldVQA dataset via HuggingFace `datasets` or synthetic data for demo."),
            ("2. Feature Engineering", "`TfidfVectorizer(max_features=5000, ngram_range=(1, 2))` on question text."),
            ("3. Model Training", "`LogisticRegression(max_iter=1000)` with 80/20 stratified split."),
            ("4. Evaluation", "Accuracy, Macro F1, per-class Precision/Recall, Confusion Matrix."),
            ("5. Artifact Saving", "`vectorizer.joblib` + `model.joblib` via `joblib.dump`."),
            ("6. Serving", "Flask REST API (`/health`, `/predict`) or Streamlit dashboard."),
        ]
        for title, desc in steps:
            with st.expander(title, expanded=True):
                st.write(desc)

    with col_api:
        st.subheader("Flask REST Endpoints")
        st.markdown("""
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Liveness check → `{"status": "ok"}` |
| `POST` | `/predict` | Classify question texts |
""")
        st.subheader("Example cURL")
        st.code("""# Health check
curl http://localhost:5000/health

# Predict
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"texts": ["What is the capital of France?"]}'
""", language="bash")

        st.subheader("Example Response")
        st.code('{"predictions": ["geography"]}', language="json")

        st.subheader("Run the API")
        st.code("""# Install & start Flask
pip install -r requirements.txt
python -c "from flask_app import create_app; create_app().run(port=5000)"
""", language="bash")

    st.markdown("---")
    st.subheader("Project Structure")
    st.code("""worldvqa/
├── src/
│   ├── data/load_data.py          # HuggingFace dataset loader
│   ├── features/build_features.py # TF-IDF feature builder
│   ├── models/train_model.py      # Train & save model artifacts
│   └── utils/config.py            # Directory helpers
├── tests/                         # pytest suite (3 tests, 1 skipped)
├── models/
│   ├── vectorizer.joblib          # Flask API artifact
│   ├── model.joblib               # Flask API artifact
│   └── demo_bundle.pkl            # Streamlit dashboard bundle
├── flask_app.py                   # REST API factory
├── mlops_pipeline.py              # CLI orchestrator
├── train_demo.py                  # Synthetic demo bundle builder
├── streamlit_app.py               # This dashboard
├── requirements.txt               # Runtime dependencies
├── requirements-ci.txt            # CI-only dependencies
└── .github/workflows/ci.yml       # GitHub Actions (3.11 & 3.12)
""")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Classify
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.title("🔮 Classify a Question")
    st.markdown(
        "Enter any world-knowledge question and the model will predict its "
        "category with confidence."
    )

    question_input = st.text_area(
        "Question",
        placeholder="e.g. What is the capital of France?",
        height=100,
        key="classify_input",
    )

    col_btn, _ = st.columns([1, 3])
    run_classify = col_btn.button("🔮 Classify", type="primary", use_container_width=True)

    if run_classify and question_input.strip():
        vec = vectorizer.transform([question_input.strip()])
        pred_label = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        classes = model.classes_

        confidence = proba.max()
        icon = CATEGORY_ICONS.get(pred_label, "")

        st.markdown("---")

        # Result headline
        st.markdown(
            f"<h2 style='text-align:center'>{icon} {pred_label.upper()}</h2>",
            unsafe_allow_html=True,
        )

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(confidence * 100, 1),
            title={"text": "Confidence (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4FC3F7"},
                "steps": [
                    {"range": [0, 50], "color": "#EF5350"},
                    {"range": [50, 75], "color": "#FFA726"},
                    {"range": [75, 100], "color": "#66BB6A"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": round(confidence * 100, 1),
                },
            },
            number={"suffix": "%"},
        ))
        fig_gauge.update_layout(height=280)
        col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
        with col_g2:
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Top-5 probability bar
        st.subheader("Category Probabilities")
        proba_df = (
            pd.DataFrame({"Category": classes, "Probability": proba})
            .sort_values("Probability", ascending=False)
            .head(5)
        )
        proba_df["Category"] = proba_df["Category"].apply(
            lambda c: f"{CATEGORY_ICONS.get(c, '')} {c}"
        )
        fig_prob = px.bar(
            proba_df, x="Category", y="Probability",
            color="Probability",
            color_continuous_scale="Blues",
            text=proba_df["Probability"].apply(lambda v: f"{v:.1%}"),
        )
        fig_prob.update_traces(textposition="outside")
        fig_prob.update_layout(
            height=340, yaxis_range=[0, 1.05],
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    elif run_classify and not question_input.strip():
        st.warning("Please enter a question before classifying.")
