"""Chat mode — conversational front for the deterministic pipeline.

The "assistant" is the pipeline itself (parse → retrieve → rerank), not a
generative LLM: every reply is templated from the run's actual outputs and the
same PRD §26 explainability data. Each message is a fresh, stateless query
(PRD §6 keeps conversational multi-turn recommendation out of scope); the
thread is presentation, not context.
"""

from __future__ import annotations

import streamlit as st

from app.ui.trace_tab import render_p1_trace, render_p2_trace, render_p3_trace

WELCOME = (
    "Halo! Saya asisten **CookFromPantry** — bukan LLM, melainkan tiga pipeline NLP deterministik "
    "(query understanding → retrieval → pantry-aware reranking). "
    "Ceritakan bahan yang Anda punya, target kalori, dan bahan yang tidak ingin dipakai. "
    "Contoh: *\"I have chicken, potato and garlic. Around 500 calories, no cheese.\"*"
)


def render_chat_mode(settings: dict, exec_fn) -> None:
    """Conversation thread + input; ``exec_fn`` runs the pipeline for one query."""
    history = st.session_state.setdefault("messages", [])
    if not history:
        with st.chat_message("assistant"):
            st.markdown(WELCOME)

    for index, message in enumerate(history):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                _render_assistant_message(message, index)

    prompt = st.chat_input("Ketik bahan / target kalori / larangan…")
    if not prompt or not prompt.strip():
        return

    st.session_state.messages.append({"role": "user", "content": prompt.strip()})
    with st.chat_message("user"):
        st.markdown(prompt.strip())

    with st.chat_message("assistant"):
        run = exec_fn(prompt.strip(), settings)
        _render_assistant_message(
            {"role": "assistant", "content": _summary(run), "run": run},
            index=len(history),  # the message being appended below
            fresh=True,
        )
    st.session_state.messages.append(
        {"role": "assistant", "content": _summary(run), "run": run}
    )
    st.rerun()


def _summary(run: dict) -> str:
    """Deterministic reply text built from the run's actual numbers."""
    recommendation = run["recommendation"]
    query = run["p1_trace"].structured_query
    excluded = ", ".join(query["excluded_ingredients"]) or "—"
    calorie = query.get("calorie_target") or query.get("calorie_max") or query.get("calorie_min")
    parts = [f"**{len(recommendation.recommendations)} resep** dari {recommendation.total_candidates} kandidat"]
    if recommendation.filtered_out_exclusion:
        parts.append(f"{recommendation.filtered_out_exclusion} tersaring karena larangan: `{excluded}`")
    if calorie is not None:
        parts.append(f"target kalori **{calorie} kcal**")
    summary = "Saya menemukan " + ", ".join(parts[:-1]) + (f", dan {parts[-1]}" if len(parts) > 1 else parts[0]) + "."
    if recommendation.is_fallback:
        summary += f"\n\n⚠️ {recommendation.fallback_message or 'Batasan dilonggarkan untuk alternatif terdekat.'}"
    top = recommendation.recommendations[0] if recommendation.recommendations else None
    if top is not None:
        summary += f"\n\nTerbaik: **#{top.rank} {top.recipe.title}** — match {top.overall_match_pct}%."
    return summary


def _render_assistant_message(message: dict, index: int, fresh: bool = False) -> None:
    st.markdown(message["content"])
    run = message.get("run")
    if run is None:
        return

    _render_recipe_cards(run["recommendation"])
    _render_behind_the_scenes(run, index)


def _render_recipe_cards(recommendation) -> None:
    """Compact PRD §26 cards sized for a chat bubble."""
    for ranked in recommendation.recommendations:
        cov = ranked.coverage
        cal = ranked.calorie
        title = f"#{ranked.rank} {ranked.recipe.title}"
        if ranked.is_fallback_candidate:
            title += f" · alternatif terdekat ({', '.join(ranked.relaxed_constraints)})"
        with st.container(border=True):
            st.markdown(f"**{title}**")
            cols = st.columns(3)
            cols[0].metric("Match", f"{ranked.overall_match_pct}%", f"score {ranked.final_score:.3f}")
            cols[1].metric("Coverage", f"{cov.coverage_score:.0%}", f"{len(cov.available_matched)}/{len(cov.recipe_ingredients)}")
            cols[2].metric(
                "Est. kalori",
                f"{round(cal.estimated_calories)} kcal" if cal.estimated_calories is not None else "N/A",
                cal.status_label if cal.estimated_calories is not None else "not estimated",
            )
            have, missing = st.columns(2)
            have.caption("✓ " + (", ".join(cov.available_matched) or "—"))
            missing.caption("✗ " + (", ".join(cov.missing_ingredients) or "—"))


def _render_behind_the_scenes(run: dict, index: int) -> None:
    label = f"🔬 Balik layar — pipeline trace ({run['p1_trace'].approach} → {run['p2_trace'].approach})"
    with st.expander(label):
        render_p1_trace(run["p1_trace"], approach=run["p1_approach"], key_prefix=f"m{index}-")
        st.divider()
        render_p2_trace(run["p2_trace"])
        st.divider()
        render_p3_trace(run["recommendation"], run["config"])
