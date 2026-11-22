"""Approach Comparison tab — per-pipeline experiments rendered live."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_comparison() -> None:
    st.subheader("⚖️ Approach comparison")
    st.caption(
        "Runs the same experiment harnesses as the CLI (evaluation/*) over the annotated + stress "
        "fixtures. Results are cached; artifact-dependent approaches appear once trained."
    )

    st.markdown("### Pipeline 1 — ingredient extraction")
    if st.button("Run Pipeline 1 comparison", key="run_p1"):
        with st.spinner("Evaluating all Pipeline 1 approaches (includes NER inference)…"):
            results = _run_p1_experiment()
        st.session_state.p1_results = results
    if st.session_state.get("p1_results"):
        rows = []
        for name, metrics in st.session_state.p1_results.items():
            rows.append(
                {
                    "approach": name,
                    "exact queries": f"{metrics['exact']}/{metrics['total']} ({metrics['exact_pct']:.1%})",
                    "available F1": metrics["available_f1"],
                    "excluded F1": metrics["excluded_f1"],
                    "calorie accuracy": metrics["calorie_accuracy"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("**Error taxonomy (why each approach fails)**")
        error_rows = [
            {"approach": name, **metrics["errors"]}
            for name, metrics in st.session_state.p1_results.items()
        ]
        st.dataframe(pd.DataFrame(error_rows).fillna(0), use_container_width=True, hide_index=True)
        st.caption(
            "Fixture: 60 annotated queries + 20 hand-authored stress cases (typos, surface variants, "
            "OOV ingredients). See `report.md` and `python -m evaluation.experiment --verbose`."
        )

    st.divider()
    st.markdown("### Pipeline 2 — candidate retrieval")
    if st.button("Run Pipeline 2 comparison", key="run_p2"):
        with st.spinner("Retrieval metrics on synthetic queries (TF-IDF vs dense)…"):
            st.session_state.p2_results = _run_p2_experiment()
    if st.session_state.get("p2_results"):
        st.dataframe(pd.DataFrame(st.session_state.p2_results), use_container_width=True, hide_index=True)
        st.caption(
            "Synthetic evaluation: pantry sampled from held-out recipes; recall@k = target recipe "
            "retrieved in top-k; MRR over 100 queries (`python -m evaluation.evaluate_retrieval`)."
        )

    st.divider()
    st.markdown("### Pipeline 3 — reranking ablation (PRD §35)")
    if st.button("Run Pipeline 3 ablation", key="run_p3"):
        with st.spinner("Ablation models A-D over annotated queries…"):
            st.session_state.p3_results = _run_p3_ablation()
    if st.session_state.get("p3_results"):
        st.dataframe(pd.DataFrame(st.session_state.p3_results), use_container_width=True, hide_index=True)
        st.caption(
            "Models A→D add coverage, missing penalty, and calorie/exclusion handling "
            "(`evaluation/evaluate_ranking.py`). Full run: 80 queries."
        )


@st.cache_data(show_spinner=False)
def _run_p1_experiment() -> dict:
    from evaluation.experiment import run_experiment

    return run_experiment()


@st.cache_data(show_spinner=False)
def _run_p2_experiment() -> list[dict]:
    import warnings

    warnings.filterwarnings("ignore")
    import pandas as pd

    from evaluation.evaluate_retrieval import evaluate_retriever, generate_in_memory_queries
    from src.pipeline2.approaches import DenseApproach, TfidfApproach

    corpus = pd.read_parquet("data/processed/recipes.parquet")
    queries = generate_in_memory_queries(corpus, num_queries=100)
    rows = []
    for approach in (TfidfApproach(), DenseApproach()):
        try:
            metrics = evaluate_retriever(approach, queries, ks=(1, 5, 10, 20, 50))
        except Exception as error:  # artifact missing — show why instead of failing the tab
            rows.append({"approach": approach.name, "error": str(error)})
            continue
        rows.append({"approach": approach.name, **metrics})
    return rows


@st.cache_data(show_spinner=False)
def _run_p3_ablation(num_queries: int = 30) -> list[dict]:
    import json
    import sys
    import warnings

    warnings.filterwarnings("ignore")
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from evaluation.evaluate_ranking import LocalTfidfRetriever, evaluate_model
    from src.pipeline3 import RecipeCalorieEstimator
    from src.pipeline3.models import RankingConfig, StructuredQuery
    from src.pipeline3.reranker import MultiCriteriaReranker

    reranker = MultiCriteriaReranker()
    estimator = RecipeCalorieEstimator()
    retriever = LocalTfidfRetriever(root / "data" / "samples" / "recipes_sample.csv", normalizer=reranker.normalizer)
    raw_queries = json.loads((root / "data" / "test" / "annotated_queries.json").read_text(encoding="utf-8"))["queries"][:num_queries]

    query_candidates = []
    calorie_cache: dict[int, float | None] = {}
    for entry in raw_queries:
        sq = StructuredQuery.from_dict(entry["expected"], raw_query=entry["query"])
        cands = retriever.retrieve(sq, estimator=estimator, calorie_cache=calorie_cache, top_n=50)
        query_candidates.append((entry["id"], sq, cands))

    configs = [
        ("Model A: TF-IDF Only", RankingConfig.ablation_model_a()),
        ("Model B: +Coverage", RankingConfig.ablation_model_b()),
        ("Model C: +Missing Pen.", RankingConfig.ablation_model_c()),
        ("Model D: Full", RankingConfig.ablation_model_d()),
    ]
    return [evaluate_model(name, config, reranker, query_candidates, top_k=3) for name, config in configs]
