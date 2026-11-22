"""Architecture tab — the system map and experiment story."""

from __future__ import annotations

import streamlit as st

ARCH_DIAGRAM = """
                          ┌──────────────────────────────────────────────┐
                          │ OFFLINE (scripts/, not user-facing)          │
                          │ RecipeNLG → preprocessing → calorie mapping  │
                          │ → recipes.parquet → TF-IDF fit / dense encode │
                          └──────────────────────────────────────────────┘
                                          │ artifacts/
                                          ▼
USER QUERY ──► PIPELINE 1 ──────────────► PIPELINE 2 ──────────────► PIPELINE 3 ──► TOP-K RECIPES
              Query & Constraint            Candidate Retrieval         Multi-Criteria Reranking
              Understanding                 (high recall, Top-N)        (pantry-aware, Top-K)

              approaches/                   approaches/                 approaches/
              ├ rule          ← prod        ├ tfidf       ← prod        ├ weighted (α/β/γ/δ) ← prod
              ├ surface-exact (ablation)    └ dense       (experiment)  └ ablation models A-D
              ├ difflib-0.82  (experiment)
              ├ char-ngram    (experiment)
              └ ner (DistilBERT, experiment)
"""

PIPELINE_NOTES = [
    (
        "Pipeline 1 — Query & Constraint Understanding",
        [
            "Rule-based production parser: longest-match vocabulary scan, aliases, plural folding, compound splitting.",
            "Shared constraint grammar: calorie forms (max / range / target / ~) and exclusion clauses (no / without / exclude / avoid / don't want).",
            "Experimental approaches swap ONLY the ingredient matcher — the constraint parser is held constant, so experiments isolate matching quality.",
            "BERT approach: DistilBERT token classification (B-ING/I-ING) fine-tuned on 15k synthetic masked queries generated from RecipeNLG NER spans; mentions are canonicalized through the same closed vocabulary.",
        ],
    ),
    (
        "Pipeline 2 — Candidate Recipe Retrieval",
        [
            "TF-IDF (max_features=1000, ngram 1-2) + pantry-staple down-weighting over canonical ingredient text; cosine similarity, Top-N=50 candidates.",
            "Dense approach: sentence-transformer (all-MiniLM-L6-v2, 384-dim) embeddings over the same corpus — lexical vs semantic retrieval experiment.",
            "Both emit the same candidate schema, so Pipeline 3 and the evaluation harnesses treat them interchangeably.",
        ],
    ),
    (
        "Pipeline 3 — Multi-Criteria Reranking",
        [
            "Weighted scoring: Score = α·S_tfidf + β·S_coverage + γ·S_calorie − δ·P_missing (PRD §24), with hard exclusion and calorie filters.",
            "Pantry staples (salt, water, oil…) get reduced weights so users aren't penalized for not mentioning them.",
            "Ablation models A→D validate each component's contribution (PRD §35); every card exposes the full breakdown for explainability (PRD §27).",
        ],
    ),
]


def render_architecture() -> None:
    st.subheader("🏗 System architecture")
    st.code(ARCH_DIAGRAM, language="text")

    for title, bullets in PIPELINE_NOTES:
        st.markdown(f"### {title}")
        for bullet in bullets:
            st.markdown(f"- {bullet}")

    st.divider()
    st.markdown("### How to read the experiment story")
    st.markdown(
        """
        1. **Pick an approach per pipeline** in the sidebar — production defaults are preselected.
        2. **Trace tab** shows exactly what each stage did to your query: spans, matches with
           provenance, model token tags, and score composition.
        3. **Comparison tab** runs the course's required experiments: same fixtures, same harnesses
           as `evaluation/`, across all approaches per pipeline.
        4. Everything runs locally — small models only, no frontier LLM API (PRD NFR-01).
        """
    )

    st.markdown("### Documents")
    st.markdown(
        "- [PRD — prd_cookfrompantry.md](../prd_cookfrompantry.md)\n"
        "- [Design — project_design_recipe_generator.md](../project_design_recipe_generator.md)\n"
        "- [Pipeline 1 error analysis — report.md](../report.md)"
    )
