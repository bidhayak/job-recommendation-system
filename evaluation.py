"""Calculating standard ranking metrics from manually judged recommendations.

Keeping the evaluation independent from the recommendation model prevents the
model from assigning its own ground-truth labels. Building a blank judgment
table supports human ratings of 0, 1, or 2, while validating the completed
table prevents partial or invalid ratings from producing misleading metrics.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def build_judgment_template(
    recommendations: pd.DataFrame,
    query_id: str = "resume_1",
    query_label: str = "Current resume",
) -> pd.DataFrame:
    """Building a blank human-labeling table from the latest ranked results.

    Copying descriptive job fields gives the evaluator enough evidence for a
    rating without changing the original recommendation table. Keeping the
    cosine score in the downloadable table supports later auditing, although
    the browser hides that score while the evaluator assigns ratings.
    """

    # Selecting identity, context, explanation, and audit fields when they are available
    columns = [
        "rank",
        "job_id",
        "job_title",
        "job_company",
        "job_location",
        "job_source",
        "job_posted_at",
        "job_description_raw",
        "matched_skills",
        "missing_skills",
        "cosine_similarity",
        "job_url",
    ]
    available = [column for column in columns if column in recommendations.columns]
    template = recommendations[available].copy()

    # Recording the resume query without storing personal resume text in the evaluation file
    template.insert(0, "query_label", str(query_label))
    template.insert(0, "query_id", str(query_id))

    # Creating nullable ratings so blank rows remain visibly incomplete in Streamlit and CSV
    template["relevance"] = pd.Series(pd.array([pd.NA] * len(template), dtype="Int64"), index=template.index)
    template["judgment_note"] = ""
    return template


def _discounted_gain(relevance: np.ndarray, k: int) -> float:
    """Calculating graded discounted gain through rank k.

    Rewarding higher relevance grades and discounting lower-ranked results
    produces the numerator used by normalized discounted cumulative gain.
    """

    # Limiting the calculation to the requested evaluation depth
    limited = relevance[:k]
    return float(
        sum((2.0 ** float(grade) - 1.0) / math.log2(rank + 2) for rank, grade in enumerate(limited))
    )


def evaluate_judgments(
    judgments: pd.DataFrame,
    k_precision: int = 5,
    k_ndcg: int = 10,
    require_complete: bool = True,
) -> dict[str, float]:
    """Averaging Precision@K, NDCG@K, and reciprocal rank across resume queries.

    Treating grades 1 and 2 as relevant supports binary Precision@K and MRR,
    while preserving the full 0-to-2 scale supports graded NDCG. Requiring all
    rows by default prevents incomplete browser work from being reported as a
    finished academic evaluation.
    """

    # Requiring the minimum fields needed for ordering and judging recommendations
    required = {"rank", "relevance"}
    if not required.issubset(judgments.columns):
        raise KeyError("The judgment file requires rank and relevance columns.")

    # Copying only metric inputs so descriptive columns cannot affect calculations
    selected_columns = ["rank", "relevance"] + (["query_id"] if "query_id" in judgments.columns else [])
    working = judgments[selected_columns].copy()
    if "query_id" not in working.columns:
        working["query_id"] = "single_query"

    # Converting spreadsheet strings into numeric values while exposing invalid entries as missing
    working["rank"] = pd.to_numeric(working["rank"], errors="coerce")
    working["relevance"] = pd.to_numeric(working["relevance"], errors="coerce")
    if working["rank"].isna().any():
        raise ValueError("Every evaluation row requires a valid numeric rank.")
    if require_complete and working["relevance"].isna().any():
        missing_count = int(working["relevance"].isna().sum())
        raise ValueError(f"Assigning a relevance rating to every row is required; {missing_count} remain blank.")

    # Allowing explicitly requested partial analysis while retaining complete evaluation by default
    working = working.dropna(subset=["relevance"])
    if working.empty:
        raise ValueError("No completed relevance labels were found.")
    if not working["relevance"].between(0, 2).all():
        raise ValueError("Relevance labels must be 0, 1, or 2.")
    if not np.equal(working["relevance"], np.floor(working["relevance"])).all():
        raise ValueError("Relevance labels must be whole numbers: 0, 1, or 2.")
    if working["query_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Every evaluation row requires a query_id.")
    if working.duplicated(["query_id", "rank"]).any():
        raise ValueError("Each query_id and rank combination must be unique.")

    # Calculating one metric set per resume before averaging across all evaluated resumes
    query_metrics: list[tuple[float, float, float]] = []
    for _, query_rows in working.groupby("query_id", sort=False):
        relevance = query_rows.sort_values("rank")["relevance"].to_numpy(dtype=float)

        # Calculating Precision@K by treating partly and highly relevant jobs as relevant
        precision_denominator = min(k_precision, len(relevance))
        precision = float(np.count_nonzero(relevance[:k_precision] > 0) / precision_denominator)

        # Calculating graded NDCG by comparing observed ordering with the ideal ordering
        dcg = _discounted_gain(relevance, k_ndcg)
        ideal = _discounted_gain(np.sort(relevance)[::-1], k_ndcg)
        ndcg = dcg / ideal if ideal > 0 else 0.0

        # Calculating reciprocal rank from the first partly or highly relevant result
        relevant_positions = np.flatnonzero(relevance > 0)
        reciprocal_rank = 1.0 / (int(relevant_positions[0]) + 1) if len(relevant_positions) else 0.0
        query_metrics.append((precision, ndcg, reciprocal_rank))

    # Averaging query-level values so each resume contributes equal evaluation weight
    metric_array = np.asarray(query_metrics, dtype=float)
    mean_reciprocal_rank = float(metric_array[:, 2].mean())

    # Returning machine-readable values for browser metrics, tests, and report evidence
    return {
        f"precision_at_{k_precision}": float(metric_array[:, 0].mean()),
        f"ndcg_at_{k_ndcg}": float(metric_array[:, 1].mean()),
        "mean_reciprocal_rank": mean_reciprocal_rank,
        "reciprocal_rank": mean_reciprocal_rank,
        "evaluated_queries": float(len(query_metrics)),
        "judged_results": float(len(working)),
    }
