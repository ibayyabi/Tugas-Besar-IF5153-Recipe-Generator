"""YummyTummy — interactive pipeline interface (Streamlit).

Run:  streamlit run app/streamlit_app.py

Two presentation modes over the same deterministic pipeline:
* 💬 Chat — conversational thread; the assistant reply is the pipeline's own
  templated output (no LLM), with the full trace one expander away.
* 📊 Dashboard — sidebar-driven run with Result / Trace / Comparison /
  Architecture tabs.

Sidebar picks an approach per pipeline (registry-aware), reranking weights,
and Top-K; both modes share one execute_pipeline().
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGO = REPO_ROOT / "app" / "assets" / "logo.png"

import streamlit as st

st.set_page_config(
    page_title="YummyTummy",
    page_icon=str(LOGO),
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.ui.arch_tab import render_architecture
from app.ui.chat_mode import render_chat_mode
from app.ui.comparison_tab import render_comparison
from app.ui.result_tab import render_results
from app.ui.trace_tab import render_p1_trace, render_p2_trace, render_p3_trace

EXAMPLE_QUERIES = [
    "I have chicken, potato, carrot, onion and egg. I want something around 500 calories and no cheese.",
    "I've got chickens, potatoes and some carrots",
    "I have chicken, tomato, onion and garlic. Under 600 calories. No cheese or milk.",
    "chiken and tommato soup, roughly 450 kcal",
    "rice, egg, and garlic — at least 300 calories",
]

MODE_CHAT = "💬 Chat"
MODE_DASHBOARD = "📊 Dashboard"


@st.cache_resource(show_spinner=False)
def load_p1_approach(name: str):
    from src.pipeline1.approaches import get_approach as get_p1

    return get_p1(name)


@st.cache_resource(show_spinner=False)
def load_p2_approach(name: str):
    from src.pipeline2.approaches import get_approach as get_p2

    return get_p2(name)


def _p1_registry() -> list[dict]:
    from src.pipeline1.approaches import available_approaches

    return available_approaches()


def _p2_registry() -> list[dict]:
    from src.pipeline2.approaches import available_approaches

    return available_approaches()


def _format_p1(name: str, options: list[dict]) -> str:
    entry = next(o for o in options if o["name"] == name)
    flag = "" if entry["available"] else " ⚠️ (run scripts/train_ner.py)"
    return f"{name}{flag}"


def _format_p2(name: str, options: list[dict]) -> str:
    entry = next(o for o in options if o["name"] == name)
    flag = "" if entry["available"] else " ⚠️ (build artifacts first)"
    return f"{name}{flag}"


def _preset_weights(preset: str) -> tuple[float, float, float, float]:
    from src.pipeline3.models import RankingConfig

    if preset.startswith("Model A"):
        c = RankingConfig.ablation_model_a()
    elif preset.startswith("Model B"):
        c = RankingConfig.ablation_model_b()
    elif preset.startswith("Model C"):
        c = RankingConfig.ablation_model_c()
    else:
        c = RankingConfig()
    return c.alpha, c.beta, c.gamma, c.delta


def sidebar() -> dict:
    """Collect configuration; returns a dict of run settings."""
    st.sidebar.image(str(LOGO), width=180)
    st.sidebar.caption("Pantry-aware & calorie-constrained recipe recommender — no frontier LLM in the loop.")

    mode = st.sidebar.radio("Mode tampilan", [MODE_CHAT, MODE_DASHBOARD], horizontal=True)

    settings: dict = {"mode": mode}

    if mode == MODE_CHAT:
        st.sidebar.caption("Ketik query di kotak chat di bawah. Approach di bawah berlaku untuk pesan berikutnya.")
        if st.sidebar.button("🗑 Kosongkan percakapan", use_container_width=True):
            st.session_state.messages = []
    else:
        with st.sidebar.expander("📝 Query", expanded=True):
            # set the widget key BEFORE instantiating the text_area so clicks sync
            if "query_box" not in st.session_state:
                st.session_state.query_box = EXAMPLE_QUERIES[0]
            for i, example in enumerate(EXAMPLE_QUERIES, start=1):
                if st.button(f"Example {i}", key=f"ex{i}", use_container_width=True):
                    st.session_state.query_box = example
            settings["query"] = st.text_area(
                "Natural-language pantry request",
                height=90,
                key="query_box",
            )

    with st.sidebar.expander("⚙️ Pipeline 1 — query understanding", expanded=False):
        p1_options = _p1_registry()
        settings["p1_name"] = st.selectbox(
            "Ingredient matching approach",
            [entry["name"] for entry in p1_options],
            format_func=lambda name: _format_p1(name, p1_options),
        )

    with st.sidebar.expander("⚙️ Pipeline 2 — retrieval", expanded=False):
        p2_options = _p2_registry()
        settings["p2_name"] = st.selectbox(
            "Retrieval approach",
            [entry["name"] for entry in p2_options],
            format_func=lambda name: _format_p2(name, p2_options),
        )

    with st.sidebar.expander("⚙️ Pipeline 3 — reranking", expanded=False):
        preset = st.selectbox(
            "Ablation model (PRD §35)",
            ["Full (α.30 β.35 γ.20 δ.15)", "Model A — TF-IDF only", "Model B — +Coverage", "Model C — +Missing pen.", "Custom"],
        )
        if preset == "Custom":
            alpha = st.slider("α — TF-IDF similarity", 0.0, 1.0, 0.30, 0.05)
            beta = st.slider("β — ingredient coverage", 0.0, 1.0, 0.35, 0.05)
            gamma = st.slider("γ — calorie compatibility", 0.0, 1.0, 0.20, 0.05)
            delta = st.slider("δ — missing penalty", 0.0, 1.0, 0.15, 0.05)
        else:
            alpha, beta, gamma, delta = _preset_weights(preset)
        settings["top_k"] = st.slider("Top-K recipes", 1, 10, 3)
        settings["candidate_n"] = st.slider("Candidates from retrieval (Top-N)", 10, 100, 50, 10)

    settings["weights"] = (alpha, beta, gamma, delta)
    settings["run"] = mode == MODE_DASHBOARD and st.sidebar.button("🔍 Recommend", type="primary", use_container_width=True)
    return settings


def execute_pipeline(query: str, settings: dict) -> dict:
    """Run the three pipelines with the selected approaches; shared by both modes."""
    p1_approach = load_p1_approach(settings["p1_name"])
    p2_approach = load_p2_approach(settings["p2_name"])

    from src.pipeline1.approaches import trace_parse_query
    from src.pipeline3.models import RankingConfig
    from src.pipeline3.recommender import recommend

    with st.spinner("Pipeline 1 — understanding the query…"):
        p1_trace = trace_parse_query(query, p1_approach)

    with st.spinner("Pipeline 2 — retrieving candidates…"):
        candidates = p2_approach.retrieve(p1_trace.structured_query, top_n=settings["candidate_n"])
        p2_trace = p2_approach.trace(p1_trace.structured_query, top_n=settings["candidate_n"])

    alpha, beta, gamma, delta = settings["weights"]
    config = RankingConfig(alpha=alpha, beta=beta, gamma=gamma, delta=delta)

    with st.spinner("Pipeline 3 — reranking…"):
        recommendation = recommend(candidates, p1_trace.structured_query, top_k=settings["top_k"], config=config)

    st.session_state.ner_attention = None  # dashboard heatmap slot; chat keys are per-message
    return {
        "p1_trace": p1_trace,
        "p1_approach": p1_approach,
        "p1_approach_name": settings["p1_name"],
        "p2_trace": p2_trace,
        "recommendation": recommendation,
        "config": config,
        "query": query,
    }


def run_dashboard(settings: dict, run: dict) -> None:
    """Render the four dashboard tabs for one pipeline run."""
    result_tab, trace_tab, comparison_tab, arch_tab = st.tabs(
        ["🍽 Result", "🔬 Pipeline Trace", "⚖️ Approach Comparison", "🏗 Architecture"]
    )
    with result_tab:
        render_results(run["recommendation"], run["p1_trace"].structured_query["excluded_ingredients"])
    with trace_tab:
        render_p1_trace(run["p1_trace"], approach=run["p1_approach"])
        st.divider()
        render_p2_trace(run["p2_trace"])
        st.divider()
        render_p3_trace(run["recommendation"], run["config"])
    with comparison_tab:
        render_comparison()
    with arch_tab:
        render_architecture()


def render_landing() -> None:
    st.image(str(LOGO), width=420)
    st.markdown(
        """
        Tell the system what is in your pantry, any calorie target, and what you don't want —
        it finds recipes that are *actually cookable*.

        **Configure the run in the sidebar**, then press **🔍 Recommend** — or switch the sidebar
        mode to **💬 Chat** and just talk to it. Same pipelines, same evidence.

        Behind the scenes three pipelines run in sequence:

        | Stage | Role | Approaches |
        | :--- | :--- | :--- |
        | Pipeline 1 | Query & constraint understanding | rule · surface-exact · difflib · char-ngram · BERT-NER |
        | Pipeline 2 | Candidate retrieval (high recall) | TF-IDF · dense embeddings |
        | Pipeline 3 | Pantry-aware multi-criteria reranking | weighted α/β/γ/δ + ablation A–D |
        """
    )
    render_architecture()


def main() -> None:
    settings = sidebar()

    if settings["mode"] == MODE_CHAT:
        render_chat_mode(settings, exec_fn=execute_pipeline)
        return

    if settings["run"] and settings.get("query", "").strip():
        run = execute_pipeline(settings["query"], settings)
        st.session_state.last_run = run
        run_dashboard(settings, run)
    elif st.session_state.get("last_run"):
        run_dashboard(settings, st.session_state.last_run)
    else:
        render_landing()


if __name__ == "__main__":
    main()
