"""Testing API normalization, TF-IDF ranking, caching, and evaluation metrics.

Using local representative responses avoids consuming API quota or requiring
credentials. Checking deterministic fixtures verifies ranking, duplicate
removal, blank human-label creation, strict label validation, and metric values.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluation import build_judgment_template, evaluate_judgments
from job_recommender import JobRecommendationModel, ModelSettings, prepare_job_data, prepare_resume_data
from job_sources import ResponseCache, combine_job_frames, fetch_adzuna, fetch_usajobs


class LiveSourceTests(unittest.TestCase):
    """Checking standardized records from representative official API responses."""

    def test_adzuna_normalizing_and_caching(self) -> None:
        """Checking Adzuna fields and avoiding a repeated request through SQLite."""

        calls: list[str] = []

        def requesting(url: str, parameters: dict) -> dict:
            """Returning a representative Adzuna payload while counting requests."""

            calls.append(url)
            self.assertNotIn("app_key", url)
            self.assertEqual(parameters["what"], "Data Analyst")
            return {
                "results": [
                    {
                        "id": "adz-1",
                        "title": "Data Analyst",
                        "description": "Using SQL and Python for dashboard analysis.",
                        "company": {"display_name": "Example Company"},
                        "location": {"display_name": "New York"},
                        "redirect_url": "https://example.org/jobs/adz-1",
                        "created": "2026-08-01T00:00:00Z",
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as folder:
            cache = ResponseCache(Path(folder) / "cache.sqlite3")
            first = fetch_adzuna(
                "Data Analyst",
                "New York",
                "us",
                "id-value",
                "secret-value",
                30,
                20,
                cache=cache,
                requester=requesting,
            )
            second = fetch_adzuna(
                "Data Analyst",
                "New York",
                "us",
                "id-value",
                "secret-value",
                30,
                20,
                cache=cache,
                requester=requesting,
            )

        self.assertEqual(len(calls), 1)
        self.assertFalse(first.from_cache)
        self.assertTrue(second.from_cache)
        self.assertEqual(first.jobs.loc[0, "company_name"], "Example Company")

    def test_usajobs_normalizing(self) -> None:
        """Checking nested USAJOBS fields and required authentication headers."""

        def requesting(url: str, parameters: dict, headers: dict) -> dict:
            """Returning a representative USAJOBS payload."""

            self.assertEqual(headers["User-Agent"], "student@example.com")
            self.assertEqual(parameters["DatePosted"], 14)
            return {
                "SearchResult": {
                    "SearchResultItems": [
                        {
                            "MatchedObjectId": "123",
                            "MatchedObjectDescriptor": {
                                "PositionID": "FED-123",
                                "PositionTitle": "Data Scientist",
                                "PositionURI": "https://www.usajobs.gov/job/123",
                                "PositionLocationDisplay": "Washington, DC",
                                "OrganizationName": "Department of Example",
                                "QualificationSummary": "Python and statistics required.",
                                "PublicationStartDate": "2026-08-01T00:00:00Z",
                                "ApplicationCloseDate": "2026-08-20T00:00:00Z",
                                "UserArea": {"Details": {"MajorDuties": "Building machine learning models."}},
                            },
                        }
                    ]
                }
            }

        result = fetch_usajobs(
            "Data Scientist",
            "Washington, DC",
            "student@example.com",
            "test-key",
            14,
            25,
            requester=requesting,
        )
        self.assertEqual(result.jobs.loc[0, "source"], "USAJOBS")
        self.assertIn("machine learning", result.jobs.loc[0, "description"].lower())

    def test_combining_sources_removing_duplicate_urls(self) -> None:
        """Combining local source records while retaining only one copy of a repeated URL."""

        # Creating two standardized rows representing the same listing from repeated imports
        first = pd.DataFrame(
            [
                {
                    "job_id": "1",
                    "title": "Data Analyst",
                    "description": "Analyzing data with SQL.",
                    "skills_desc": "SQL",
                    "location": "New York",
                    "company_name": "Example Company",
                    "job_posting_url": "https://example.org/job/1",
                    "source": "Adzuna",
                    "posted_at": "2026-08-01T00:00:00Z",
                    "expires_at": "",
                }
            ]
        )
        second = first.assign(source="Local table")

        # Verifying URL-based duplicate removal before TF-IDF fitting
        combined = combine_job_frames([first, second])
        self.assertEqual(len(combined), 1)
        self.assertEqual(combined.loc[0, "source"], "Adzuna")


class RecommendationTests(unittest.TestCase):
    """Checking ranking and explanation behavior on a deterministic fixture."""

    def test_ranking_relevant_job_first(self) -> None:
        """Ranking a data role above unrelated sales and nursing roles."""

        resumes = prepare_resume_data(
            pd.DataFrame(
                {
                    "Resume_str": [
                        "Data analyst experienced with Python SQL pandas Tableau machine learning and statistics."
                    ],
                    "Category": ["Data Science"],
                }
            )
        )
        jobs = prepare_job_data(
            pd.DataFrame(
                {
                    "title": ["Data Analyst", "Sales Manager", "Registered Nurse"],
                    "description": [
                        "Analyzing data with Python SQL pandas and Tableau dashboards.",
                        "Leading account sales and customer pipeline forecasting.",
                        "Providing patient care and clinical nursing support.",
                    ],
                    "company_name": ["A", "B", "C"],
                    "location": ["New York", "New York", "New York"],
                    "source": ["Fixture", "Fixture", "Fixture"],
                }
            )
        )
        settings = ModelSettings(min_df=1, max_df=1.0, max_features=5000)
        model = JobRecommendationModel(settings)
        model.fit(resumes, jobs)
        query_vector, _ = model.create_query_vector(0, None)
        recommendations, _, _ = model.recommend(query_vector, 3)
        self.assertEqual(recommendations.loc[0, "job_title"], "Data Analyst")
        self.assertIn("python", recommendations.loc[0, "matched_skills"])
        self.assertGreater(recommendations.loc[0, "cosine_similarity"], recommendations.loc[1, "cosine_similarity"])


class EvaluationTests(unittest.TestCase):
    """Checking standard information-retrieval metric calculations."""

    def test_building_blank_judgment_template(self) -> None:
        """Creating editable ratings while retaining job evidence for human review."""

        # Building one representative recommendation containing display and audit fields
        recommendations = pd.DataFrame(
            {
                "rank": [1],
                "job_id": ["job-1"],
                "job_title": ["Data Analyst"],
                "job_company": ["Example Company"],
                "job_location": ["New York"],
                "job_source": ["Adzuna"],
                "job_description_raw": ["Analyzing data with SQL and Python."],
                "cosine_similarity": [0.42],
            }
        )

        # Verifying blank human labels and preserved context without storing resume text
        template = build_judgment_template(
            recommendations,
            query_id="resume_1",
            query_label="Uploaded resume",
        )
        self.assertTrue(pd.isna(template.loc[0, "relevance"]))
        self.assertEqual(template.loc[0, "job_title"], "Data Analyst")
        self.assertEqual(template.loc[0, "query_label"], "Uploaded resume")

    def test_rejecting_incomplete_judgments(self) -> None:
        """Preventing partial human ratings from becoming final academic metrics."""

        # Leaving one rating blank to represent unfinished browser evaluation work
        incomplete = pd.DataFrame(
            {
                "query_id": ["resume_1", "resume_1"],
                "rank": [1, 2],
                "relevance": [2, pd.NA],
            }
        )

        # Requiring an informative error instead of silently dropping the unfinished row
        with self.assertRaisesRegex(ValueError, "every row"):
            evaluate_judgments(incomplete)

    def test_rejecting_invalid_judgments(self) -> None:
        """Rejecting ratings outside the defined zero-to-two relevance scale."""

        # Supplying an invalid grade for checking strict scale enforcement
        invalid = pd.DataFrame({"rank": [1], "relevance": [3]})
        with self.assertRaisesRegex(ValueError, "0, 1, or 2"):
            evaluate_judgments(invalid)

    def test_evaluation_metrics(self) -> None:
        """Calculating perfect early precision and reciprocal rank."""

        judgments = pd.DataFrame({"rank": [1, 2, 3, 4, 5], "relevance": [2, 1, 0, 1, 0]})
        metrics = evaluate_judgments(judgments)
        self.assertAlmostEqual(metrics["precision_at_5"], 0.6)
        self.assertAlmostEqual(metrics["reciprocal_rank"], 1.0)
        self.assertGreater(metrics["ndcg_at_10"], 0.9)


if __name__ == "__main__":
    unittest.main()
