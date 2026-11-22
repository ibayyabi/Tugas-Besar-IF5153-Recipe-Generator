"""Pipeline Trace tab — how the system works behind the scenes, stage by stage."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st

from app.ui.helpers import (
    COLOR_CALORIE,
    COLOR_EXCLUSION,
    highlight_matches,
    highlight_spans,
    score_components,
)


def _mono_html(html_text: str, dim: bool = False) -> None:
    """Monospace inline-HTML block; ``dim`` softens secondary contexts."""
    style = "font-family:monospace;font-size:0.95rem;line-height:1.6"
    if dim:
        style += ";opacity:0.75"
    st.markdown(f"<div style='{style}'>{html_text}</div>", unsafe_allow_html=True)


def _match_table(matches: Iterable) -> None:
    """Ingredient matching evidence as a table."""
    if not matches:
        st.info("No ingredient matches in the available-ingredient context.")
        return
    rows = [
        {
            "surface": m.surface,
            "canonical": m.canonical,
            "method": m.method,
            "score": m.score,
            "span": f"[{m.start}:{m.end}]",
        }
        for m in matches
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _candidate_rows(p2_trace) -> list[dict]:
    return [
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


def render_p1_trace(p1_trace, approach=None, key_prefix: str = "", compact: bool = False) -> None:
    """Query understanding: normalization, constraint spans, matching evidence.

    ``compact`` (chat bubbles) leads with the highlighted contexts and nests
    the tabular evidence one disclosure deep; the dashboard keeps it all open.
    """
    if compact:
        _render_p1_compact(p1_trace, approach, key_prefix)
        return

    st.subheader("Pipeline 1 — Query & Constraint Understanding")
    st.caption(f"Approach: `{p1_trace.approach}`")

    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Step 1 — text normalization + constraint spans**")
        calorie_html = highlight_spans(p1_trace.normalized_text, p1_trace.calorie_spans, COLOR_CALORIE, "calorie constraint")
        _mono_html(calorie_html)
        if p1_trace.exclusion_spans:
            st.caption("Red = exclusion clause (blanked for ingredient matching); amber = calorie span (blanked first).")
        st.markdown("**Step 2 — masked matching context (what the matcher actually sees)**")
        _mono_html(p1_trace.available_context.strip() or "(empty)", dim=True)

    with right:
        st.markdown("**Step 3 — structured query**")
        st.json(p1_trace.structured_query, expanded=False)

    st.markdown("**Step 4 — ingredient matching evidence**")
    _match_table(p1_trace.available_trace.matches)

    st.markdown("**Highlighted matches on the available context**")
    _mono_html(highlight_matches(p1_trace.available_context, p1_trace.available_trace.matches))
    st.caption("Green = vocabulary match; blue = model-detected mention (hover for provenance).")

    excluded_matches = p1_trace.excluded_trace.matches
    if p1_trace.exclusion_context.strip():
        st.markdown("**Exclusion clause matching**")
        _mono_html(highlight_matches(p1_trace.exclusion_context, excluded_matches, base_color=COLOR_EXCLUSION))

    approach_extras = p1_trace.available_trace.extras
    if approach_extras.get("tokens") and approach is not None:
        render_ner_evidence(approach_extras, approach, p1_trace.raw_query, key_prefix=key_prefix)


def _render_p1_compact(p1_trace, approach, key_prefix: str) -> None:
    """Chat-bubble variant: highlighted contexts first, evidence nested one level."""
    _mono_html(highlight_spans(p1_trace.normalized_text, p1_trace.calorie_spans, COLOR_CALORIE, "calorie constraint"))
    _mono_html(highlight_matches(p1_trace.available_context, p1_trace.available_trace.matches))
    if p1_trace.exclusion_context.strip():
        _mono_html(
            highlight_matches(p1_trace.exclusion_context, p1_trace.excluded_trace.matches, base_color=COLOR_EXCLUSION)
        )
    st.caption("Amber = kalori (dibuang duluan) · merah = larangan · hijau/biru = bahan yang cocok (hover untuk detail).")

    with st.expander("📋 Evidence — masked context, structured query & tabel match"):
        st.markdown("**Masked matching context**")
        _mono_html(p1_trace.available_context.strip() or "(empty)", dim=True)
        st.markdown("**Structured query**")
        st.json(p1_trace.structured_query, expanded=False)
        st.markdown("**Ingredient matching evidence**")
        _match_table(p1_trace.available_trace.matches)

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

    tag_style = {"B-ING": "background:#9fd3f0;color:#1a1a1a", "I-ING": "background:#d7ecf9;color:#1a1a1a", "O": ""}
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


def render_p2_trace(p2_trace, compact: bool = False) -> None:
    """Candidate retrieval: query representation and scored candidates.

    ``compact`` (chat bubbles) leads with the candidate table and nests the
    query representation; the dashboard keeps the two side by side.
    """
    if compact:
        _render_p2_compact(p2_trace)
        return

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
        st.dataframe(pd.DataFrame(_candidate_rows(p2_trace)), use_container_width=True, hide_index=True)
        st.caption("These candidates (high recall) flow into Pipeline 3 for multi-criteria reranking.")
        if p2_trace.extras.get("matrix_shape") or p2_trace.extras.get("vectors_shape"):
            st.caption(f"Corpus representation: {p2_trace.extras.get('matrix_shape') or p2_trace.extras.get('vectors_shape')}")


def _render_p2_compact(p2_trace) -> None:
    """Chat-bubble variant: candidates first, query representation nested."""
    st.dataframe(pd.DataFrame(_candidate_rows(p2_trace)), use_container_width=True, hide_index=True)
    with st.expander("🔎 Query representation & term weights"):
        st.code(p2_trace.query_text or "(empty)")
        if p2_trace.query_terms and p2_trace.score_type == "tfidf-cosine":
            st.caption("TF-IDF weights of the query vector (top 20)")
            terms_df = pd.DataFrame(p2_trace.query_terms).set_index("term")
            st.bar_chart(terms_df, height=280)
        elif p2_trace.query_terms:
            st.caption("Query tokens (dense: single embedding, no per-term weights)")
            st.dataframe(pd.DataFrame(p2_trace.query_terms), use_container_width=True, hide_index=True)


def render_p3_trace(recommendation, config, compact: bool = False) -> None:
    """Reranking: weighted contributions and per-recipe breakdowns.

    ``compact`` (chat bubbles) shows only the score-composition chart — the
    per-recipe detail already sits in the cards above the trace.
    """
    if compact:
        _render_p3_compact(recommendation, config)
        return

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


def _render_p3_compact(recommendation, config) -> None:
    """Chat-bubble variant: the score chart only; cards above carry per-recipe detail."""
    rows = []
    for ranked in recommendation.recommendations:
        row = {"recipe": f"#{ranked.rank} {ranked.recipe.title[:38]}"}
        row.update(score_components(ranked, config))
        rows.append(row)
    st.bar_chart(pd.DataFrame(rows).set_index("recipe"), height=220)
    st.caption(
        f"Bobot: α={config.alpha} β={config.beta} γ={config.gamma} δ={config.delta} · "
        f"{recommendation.total_candidates} kandidat → Top-{len(recommendation.recommendations)} · "
        f"tersaring {recommendation.filtered_out_exclusion} larangan, {recommendation.filtered_out_calorie} kalori"
    )
