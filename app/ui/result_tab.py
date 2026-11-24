"""Result tab — explainable Top-K recommendation cards (PRD §26)."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_results(recommendation, excluded_ingredients: list[str]) -> None:
    if recommendation.is_fallback:
        st.warning(recommendation.fallback_message or "No exact match — showing closest alternatives (relaxed constraints).")

    if not recommendation.recommendations:
        reason = getattr(recommendation, "no_match_reason", None)
        st.warning("No recipe matched your request." + (f"\n\n{reason}" if reason else ""))
        st.caption(
            f"{recommendation.total_candidates} candidates checked · "
            f"{recommendation.filtered_out_exclusion} removed by excluded ingredients · "
            f"{recommendation.filtered_out_calorie} removed by calorie bound"
        )
        return

    st.subheader(f"Top-{len(recommendation.recommendations)} recommendations")
    st.caption(
        f"{recommendation.total_candidates} candidates reranked · "
        f"{recommendation.filtered_out_exclusion} removed by excluded ingredients · "
        f"{recommendation.filtered_out_calorie} removed by calorie bound"
    )

    for ranked in recommendation.recommendations:
        recipe = ranked.recipe
        cov = ranked.coverage
        cal = ranked.calorie

        header = f"#{ranked.rank}  {recipe.title}"
        if ranked.is_fallback_candidate:
            header += f"  · closest alternative (relaxed: {', '.join(ranked.relaxed_constraints)})"
        with st.container(border=True):
            col_score, col_coverage, col_calorie = st.columns([1, 1, 1])
            col_score.metric("Overall match", f"{ranked.overall_match_pct}%", f"score {ranked.final_score:.3f}")
            col_coverage.metric(
                "Ingredient coverage",
                f"{cov.coverage_score:.0%}",
                f"{len(cov.available_matched)}/{len(cov.recipe_ingredients)} ingredients",
            )
            if cal.estimated_calories is not None:
                col_calorie.metric("Estimated calories / serving", f"{round(cal.estimated_calories)} kcal", cal.status_label)
            else:
                col_calorie.metric("Estimated calories / serving", "N/A", "not estimated")

            have, missing = st.columns(2)
            have.markdown("**✓ You have**")
            have.write(", ".join(cov.available_matched) or "—")
            missing.markdown("**✗ Missing (buy)**")
            missing.write(", ".join(cov.missing_ingredients) or "—")

            if excluded_ingredients:
                status = "✗ violated" if ranked.exclusion.violated else f"✓ none detected (checked: {', '.join(excluded_ingredients)})"
                st.caption(f"Excluded ingredients: {status}")

            with st.expander("Ingredients & directions"):
                st.markdown("**Ingredients**")
                st.write(" · ".join(recipe.ingredients) or "—")
                st.markdown("**Directions**")
                for i, step in enumerate(recipe.directions, start=1):
                    st.write(f"{i}. {step}")

    summary = pd.DataFrame(
        [
            {
                "rank": r.rank,
                "title": r.recipe.title[:50],
                "overall %": r.overall_match_pct,
                "coverage": round(r.coverage.coverage_score, 3),
                "kcal": r.calorie.estimated_calories,
                "missing": r.coverage.missing_count,
                "final score": round(r.final_score, 4),
            }
            for r in recommendation.recommendations
        ]
    )
    st.markdown("**Comparison summary**")
    st.dataframe(summary, use_container_width=True, hide_index=True)
