"""Pipeline Trace tab — how the system works behind the scenes, stage by stage."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui.helpers import (
    COLOR_CALORIE,
    COLOR_EXCLUSION,
    highlight_matches,
    highlight_spans,
    score_components,
)


def render_p1_trace(p1_trace, approach=None, key_prefix: str = "") -> None:
    """Query understanding: normalization, constraint spans, matching evidence."""
    st.subheader("Pipeline 1 — Query & Constraint Understanding")
    st.caption(f"Approach: `{p1_trace.approach}`")

    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Step 1 — text normalization + constraint spans**")
        calorie_html = highlight_spans(p1_trace.normalized_text, p1_trace.calorie_spans, COLOR_CALORIE, "calorie constraint")
        st.markdown(
            f"<div style='font-family:monospace;font-size:0.95rem;line-height:1.6'>{calorie_html}</div>",
            unsafe_allow_html=True,
        )
        if p1_trace.exclusion_spans:
            st.caption("Red = exclusion clause (blanked for ingredient matching); amber = calorie span (blanked first).")
        st.markdown("**Step 2 — masked matching context (what the matcher actually sees)**")
        st.markdown(
            f"<div style='font-family:monospace;font-size:0.95rem;color:#666'>{p1_trace.available_context.strip() or '(empty)'}</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("**Step 3 — structured query**")
        st.json(p1_trace.structured_query, expanded=True)

    st.markdown("**Step 4 — ingredient matching evidence**")
    available_matches = p1_trace.available_trace.matches
    if available_matches:
        rows = [
            {
                "surface": m.surface,
                "canonical": m.canonical,
                "method": m.method,
                "score": m.score,
                "span": f"[{m.start}:{m.end}]",
            }
            for m in available_matches
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No ingredient matches in the available-ingredient context.")

    st.markdown("**Highlighted matches on the available context**")
    st.markdown(
        f"<div style='font-family:monospace;font-size:0.95rem;line-height:1.6'>"
        f"{highlight_matches(p1_trace.available_context, available_matches)}</div>",
        unsafe_allow_html=True,
    )
    st.caption("Green = vocabulary match; blue = model-detected mention (hover for provenance).")

    excluded_matches = p1_trace.excluded_trace.matches
    if p1_trace.exclusion_context.strip():
        st.markdown("**Exclusion clause matching**")
        st.markdown(
            f"<div style='font-family:monospace;font-size:0.95rem;line-height:1.6'>"
            f"{highlight_matches(p1_trace.exclusion_context, excluded_matches, base_color=COLOR_EXCLUSION)}</div>",
            unsafe_allow_html=True,
        )

    approach_extras = p1_trace.available_trace.extras
    if approach_extras.get("tokens") and approach is not None:
        render_ner_evidence(approach_extras, approach, p1_trace.raw_query, key_prefix=key_prefix)


def render_ner_evidence(extras: dict, approach, raw_query: str, key_prefix: str = "") -> None:
    """BERT behind the scenes: token tags, probabilities, and attention."""
    st.divider()
    st.markdown("#### 🤖 BERT visualizer — token classification evidence")

    token_rows = [
        {"token": t["token"], "tag": t["tag"], "prob": t["prob"]}
        for t in extras["tokens"]
        if t["token"].strip()
    ]
    if not token_rows:
        st.caption("No tokens to display.")
        return

    tag_style = {"B-ING": "background:#9fd3f0", "I-ING": "background:#d7ecf9", "O": ""}
    chips = " ".join(
        f"<code style='{tag_style.get(row['tag'], '')};border-radius:3px;padding:1px 4px'>{_html_token(row['token'])}"
        f"<sub style='color:#888'>{row['tag'][0] if row['tag'] != 'O' else ''}</sub></code>"
        for row in token_rows
    )
    st.markdown(
        f"<div style='line-height:2.0;font-size:0.95rem'>Token tags: {chips}</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(pd.DataFrame(token_rows), use_container_width=True, hide_index=True)

    if extras.get("uncanonicalized"):
        st.warning(
            "Model-detected mentions the closed vocabulary cannot canonicalize (dropped): "
            + ", ".join(f"`{u['surface']}`" for u in extras["uncanonicalized"])
        )

    attention_key = f"{key_prefix}ner_attention"
    if st.button("Show attention heatmap (last layer)", key=f"{key_prefix}btn_attention"):
        with st.spinner("Computing attention…"):
            st.session_state[attention_key] = approach.attention(raw_query)
    if st.session_state.get(attention_key):
        render_attention_heatmap(st.session_state[attention_key])


def _html_token(token: str) -> str:
    import html

    return html.escape(token)


def render_attention_heatmap(attention: dict) -> None:
    import altair as alt

    tokens = attention["tokens"]
    matrix = attention["matrix"]
    if len(tokens) > 48:
        tokens, matrix = tokens[:48], [row[:48] for row in matrix[:48]]
        st.caption("Truncated to the first 48 tokens for readability.")
    records = [
        {"from": tokens[i], "to": tokens[j], "attention": value}
        for i, row in enumerate(matrix)
        for j, value in enumerate(row)
    ]
    df = pd.DataFrame(records)
    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("to:N", sort=list(tokens), axis=alt.Axis(labelAngle=-90, title="attended to")),
            y=alt.Y("from:N", sort=list(tokens), axis=alt.Axis(title="token")),
            color=alt.Color("attention:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["from:N", "to:N", alt.Tooltip("attention:Q", format=".3f")],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"Layer {attention['layers']} (last), averaged over {attention['heads']} attention heads — "
        "how much each token attends to every other token when tagging ingredients."
    )


def render_p2_trace(p2_trace) -> None:
    """Candidate retrieval: query representation and scored candidates."""
    st.subheader("Pipeline 2 — Candidate Recipe Retrieval")
    st.caption(f"Approach: `{p2_trace.approach}` ({p2_trace.score_type}) · top-{p2_trace.extras.get('top_n', '?')}")

    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Query representation**")
        st.code(p2_trace.query_text or "(empty)")
        if p2_trace.query_terms and p2_trace.score_type == "tfidf-cosine":
            st.caption("TF-IDF weights of the query vector (top 20)")
            terms_df = pd.DataFrame(p2_trace.query_terms).set_index("term")
            st.bar_chart(terms_df, height=280)
        elif p2_trace.query_terms:
            st.caption("Query tokens (dense: single embedding, no per-term weights)")
            st.dataframe(pd.DataFrame(p2_trace.query_terms), use_container_width=True, hide_index=True)
    with right:
        st.markdown(f"**Top candidates by {p2_trace.score_type}**")
        candidates = [
            {
                "rank": i + 1,
                "recipe_id": c["recipe_id"],
                "title": c["title"][:60],
                "score": c["tfidf_score"],
                "#ingredients": len(c["canonical_ingredients"]),
                "kcal": c["total_calories"],
            }
            for i, c in enumerate(p2_trace.candidates)
        ]
        st.dataframe(pd.DataFrame(candidates), use_container_width=True, hide_index=True)
        st.caption("These candidates (high recall) flow into Pipeline 3 for multi-criteria reranking.")
        if p2_trace.extras.get("matrix_shape") or p2_trace.extras.get("vectors_shape"):
            st.caption(f"Corpus representation: {p2_trace.extras.get('matrix_shape') or p2_trace.extras.get('vectors_shape')}")


def render_p3_trace(recommendation, config) -> None:
    """Reranking: weighted contributions and per-recipe breakdowns."""
    st.subheader("Pipeline 3 — Multi-Criteria Reranking")
    st.caption(
        f"α={config.alpha} β={config.beta} γ={config.gamma} δ={config.delta} · "
        f"{recommendation.total_candidates} candidates → Top-{len(recommendation.recommendations)} · "
        f"filtered: {recommendation.filtered_out_exclusion} by exclusion, {recommendation.filtered_out_calorie} by calorie"
    )

    st.markdown("**Score composition (weighted components per recipe)**")
    rows = []
    for ranked in recommendation.recommendations:
        components = score_components(ranked, config)
        row = {"recipe": f"#{ranked.rank} {ranked.recipe.title[:38]}"}
        row.update(components)
        row["final"] = round(ranked.final_score, 4)
        rows.append(row)
    chart_df = pd.DataFrame(rows).set_index("recipe")
    st.bar_chart(chart_df.drop(columns=["final"]), height=240)
    st.dataframe(chart_df, use_container_width=True)

    st.markdown("**Per-recipe breakdown**")
    for ranked in recommendation.recommendations:
        cov = ranked.coverage
        cal = ranked.calorie
        with st.expander(f"#{ranked.rank} {ranked.recipe.title} — score {ranked.final_score:.3f} ({ranked.overall_match_pct}%)"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Coverage", f"{cov.coverage_score:.0%}", f"{len(cov.available_matched)}/{len(cov.recipe_ingredients)}")
            if cal.estimated_calories is not None:
                col2.metric("Est. calories", f"{round(cal.estimated_calories)} kcal", cal.status_label)
            else:
                col2.metric("Est. calories", "N/A", "not estimated")
            col3.metric("Missing", cov.missing_count, f"non-staple: {cov.missing_non_staple_count}")

            col_a, col_b = st.columns(2)
            col_a.markdown("**✓ You have**")
            col_a.write(", ".join(cov.available_matched) or "—")
            col_b.markdown("**✗ Missing**")
            col_b.write(", ".join(cov.missing_ingredients) or "—")

            if ranked.exclusion.violated:
                st.error(f"Excluded ingredients violated: {ranked.exclusion.violated_ingredients}")
            else:
                st.success("No excluded ingredients.")
            st.caption(f"Ingredient weights: {cov.ingredient_weights}")
