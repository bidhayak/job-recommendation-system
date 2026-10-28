"""Running the local TF-IDF resume-to-job recommendation engine.

Loading and cleaning resume and job datasets, fitting one shared TF-IDF
vocabulary, calculating cosine similarity, producing transparent term and skill
explanations, and exporting reproducible evidence keeps the modeling workflow
independent from the Streamlit interface and external job-source adapters.
"""

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Selecting a writable temporary cache folder while respecting an existing user setting
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "job_recommender_matplotlib"))

import matplotlib

# Selecting a non-interactive backend for saving figures on local computers
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Defining reproducible settings matching the final capstone notebook
@dataclass(frozen=True)
class ModelSettings:
    """Storing reproducible limits and TF-IDF settings."""

    max_resumes: int = 500
    max_jobs: int = 10000
    top_n: int = 10
    random_state: int = 42
    max_features: int = 50000
    min_df: int = 2
    max_df: float = 0.95
    ngram_min: int = 1
    ngram_max: int = 2


# Defining expected columns for handling common dataset versions
RESUME_TEXT_COLUMNS = ["Resume_str", "resume", "resume_text", "cv", "Resume_html"]
RESUME_CATEGORY_COLUMNS = ["Category", "category", "job_category", "label"]
JOB_ID_COLUMNS = ["job_id", "jobid", "id"]
JOB_TITLE_COLUMNS = ["title", "job_title", "name"]
JOB_DESCRIPTION_COLUMNS = ["description", "job_description", "details", "content"]
JOB_SKILLS_COLUMNS = ["skills_desc", "skills", "skill", "required_skills"]
JOB_LOCATION_COLUMNS = ["location", "job_location", "city"]
JOB_COMPANY_COLUMNS = ["company_name", "company", "organization"]
JOB_URL_COLUMNS = [
    "job_posting_url",
    "application_url",
    "job_url",
    "posting_url",
    "apply_url",
    "url",
]
JOB_SOURCE_COLUMNS = ["source", "job_source", "job_board", "platform", "site", "_input_source"]
JOB_POSTED_COLUMNS = ["posted_at", "created", "listed_time", "original_listed_time", "date_posted"]
JOB_EXPIRES_COLUMNS = ["expires_at", "expiry", "application_close_date", "closed_at"]


# Defining a transparent technical-skill vocabulary for producing explanations
KNOWN_SKILLS = {
    ".net",
    "agile",
    "airflow",
    "angular",
    "ansible",
    "apache spark",
    "aws",
    "azure",
    "c#",
    "c++",
    "cassandra",
    "ci/cd",
    "computer vision",
    "css",
    "data analysis",
    "data engineering",
    "data visualization",
    "deep learning",
    "django",
    "docker",
    "excel",
    "fastapi",
    "flask",
    "gcp",
    "git",
    "github",
    "hadoop",
    "html",
    "java",
    "javascript",
    "jenkins",
    "jira",
    "kafka",
    "keras",
    "kubernetes",
    "linux",
    "machine learning",
    "matplotlib",
    "mongodb",
    "mysql",
    "natural language processing",
    "nlp",
    "node.js",
    "nosql",
    "numpy",
    "pandas",
    "plotly",
    "postgresql",
    "power bi",
    "python",
    "pytorch",
    "r",
    "react",
    "rest api",
    "sas",
    "scikit-learn",
    "scrum",
    "seaborn",
    "snowflake",
    "spark",
    "sql",
    "sqlite",
    "tableau",
    "tensorflow",
    "terraform",
    "typescript",
    "unix",
    "visual basic",
    "vue",
}


def clean_text(value: object) -> str:
    """Cleaning text while preserving common technology symbols."""

    # Converting missing values into empty strings before applying regular expressions
    text = "" if pd.isna(value) else str(value)
    # Removing markup and web addresses because they do not describe applicant-job fit
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Normalizing case while retaining symbols used by skills such as C++, C#, and .NET
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./\- ]+", " ", text)
    # Collapsing repeated spaces into one consistent token boundary
    return re.sub(r"\s+", " ", text).strip()


def extract_known_skills(value: object) -> set[str]:
    """Extracting recognizable technical skills with literal boundary-aware matching."""

    # Padding cleaned text so skills at the beginning and end receive boundaries
    text = f" {clean_text(value)} "
    found: set[str] = set()
    for skill in KNOWN_SKILLS:
        # Escaping punctuation while allowing flexible spaces inside multi-word skills
        escaped = re.escape(skill).replace(r"\ ", r"\s+")
        if re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text, flags=re.IGNORECASE):
            found.add(skill)
    # Removing shorter aliases when a more descriptive version is already present
    if "natural language processing" in found:
        found.discard("nlp")
    if "apache spark" in found:
        found.discard("spark")
    return found


def find_column(dataframe: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Finding the first matching column without depending on capitalization."""

    # Creating a lowercase lookup while preserving each table's original column spelling
    lookup = {str(column).strip().lower(): str(column) for column in dataframe.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def is_generated_output(path: Path, root_folder: Path) -> bool:
    """Identifying generated folders that should not be selected as source data."""

    try:
        parts = [part.lower() for part in path.relative_to(root_folder).parts]
    except ValueError:
        return False
    return any(
        part.startswith("outputs")
        or part.startswith("figures")
        or part.startswith("rendered")
        or part in {"integration_outputs", "final_system_outputs", "__pycache__", ".venv"}
        for part in parts
    )


def find_project_directory(start_folder: Path, requested_folder: Optional[Path]) -> Path:
    """Finding the capstone root by checking current and parent folders."""

    # Respecting an explicit project path before attempting automatic discovery
    if requested_folder is not None:
        project_directory = requested_folder.expanduser().resolve()
        if not project_directory.is_dir():
            raise FileNotFoundError(f"Project folder not found: {project_directory}")
        return project_directory

    # Walking upward so the script can run from the application or capstone folder
    start_folder = start_folder.resolve()
    for folder in [start_folder, *start_folder.parents]:
        resume_file = folder / "Resume" / "Resume" / "Resume.csv"
        job_file = folder / "Jobs" / "postings.csv"
        if resume_file.exists() and job_file.exists():
            return folder

    raise FileNotFoundError(
        "The Resume/Resume/Resume.csv and Jobs/postings.csv files were not found. "
        "Run the program inside the capstone folder or provide --project-dir."
    )


def discover_data_files(folder: Path) -> list[Path]:
    """Finding CSV and Excel source tables recursively."""

    # Using a set to prevent repeated paths across file-extension patterns
    discovered: set[Path] = set()
    for pattern in ("**/*.csv", "**/*.xlsx", "**/*.xls"):
        for path in folder.glob(pattern):
            resolved_path = path.resolve()
            if not is_generated_output(resolved_path, folder.resolve()):
                discovered.add(resolved_path)
    return sorted(discovered)


def read_header(path: Path) -> pd.DataFrame:
    """Reading only table headers for detecting source schemas."""

    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, nrows=0, low_memory=False)
        return pd.read_excel(path, nrows=0)
    except (OSError, ValueError, UnicodeError):
        return pd.DataFrame()


def score_schema(columns: pd.Index, table_type: str) -> int:
    """Scoring schemas for selecting the most likely resume or job table."""

    # Normalizing names before applying stronger weights to essential text columns
    normalized = {str(column).strip().lower() for column in columns}
    if table_type == "resume":
        text_matches = normalized.intersection({column.lower() for column in RESUME_TEXT_COLUMNS})
        category_matches = normalized.intersection({column.lower() for column in RESUME_CATEGORY_COLUMNS})
        return (4 * len(text_matches)) + len(category_matches)

    title_matches = normalized.intersection({column.lower() for column in JOB_TITLE_COLUMNS})
    description_matches = normalized.intersection({column.lower() for column in JOB_DESCRIPTION_COLUMNS})
    supporting = normalized.intersection(
        {"job_id", "company_id", "company_name", "location", "skills_desc", "formatted_work_type"}
    )
    return (3 * len(title_matches)) + (4 * len(description_matches)) + len(supporting)


def select_dataset_path(
    project_directory: Path,
    table_type: str,
    requested_path: Optional[Path],
) -> Path:
    """Selecting an explicit, conventional, or schema-detected dataset table."""

    # Selecting an explicit file when the caller supplies one
    if requested_path is not None:
        resolved_path = requested_path.expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"{table_type.title()} data file not found: {resolved_path}")
        return resolved_path

    # Checking the folder layout used by the original capstone datasets
    conventional_path = (
        project_directory / "Resume" / "Resume" / "Resume.csv"
        if table_type == "resume"
        else project_directory / "Jobs" / "postings.csv"
    )
    if conventional_path.is_file():
        return conventional_path.resolve()

    # Scoring remaining tables by schema when conventional paths are unavailable
    candidates: list[tuple[int, str, Path]] = []
    for path in discover_data_files(project_directory):
        score = score_schema(read_header(path).columns, table_type)
        candidates.append((score, path.name.lower(), path))

    # Selecting the strongest schema match while using filename order for stable ties
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if not candidates or candidates[0][0] == 0:
        raise FileNotFoundError(f"No {table_type} dataset could be detected in {project_directory}")
    return candidates[0][2]


def load_table(path: Path, row_limit: Optional[int] = None) -> pd.DataFrame:
    """Loading a CSV or Excel table while controlling memory usage."""

    # Using pandas readers while limiting rows when large research files are detected
    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path, nrows=row_limit, low_memory=False)
        except UnicodeDecodeError:
            # Retrying older CSV exports with a common single-byte encoding
            return pd.read_csv(path, nrows=row_limit, low_memory=False, encoding="latin-1")
    return pd.read_excel(path, nrows=row_limit)


def sample_rows(dataframe: pd.DataFrame, row_limit: int, random_state: int) -> pd.DataFrame:
    """Sampling large tables reproducibly before resetting row indexes."""

    # Sampling only oversized tables while preserving smaller tables completely
    if len(dataframe) > row_limit:
        dataframe = dataframe.sample(row_limit, random_state=random_state)
    return dataframe.reset_index(drop=True)


def prepare_resume_data(resume_raw: pd.DataFrame) -> pd.DataFrame:
    """Standardizing and cleaning resume records for vectorization."""

    # Locating required text and optional category columns across dataset versions
    text_column = find_column(resume_raw, RESUME_TEXT_COLUMNS)
    category_column = find_column(resume_raw, RESUME_CATEGORY_COLUMNS)
    if text_column is None:
        raise KeyError("No resume text column was found in the selected resume table.")

    # Filling missing categories without making category values part of the model text
    category_values = (
        resume_raw[category_column].fillna("Unknown").astype(str)
        if category_column is not None
        else pd.Series("Unknown", index=resume_raw.index)
    )
    # Building a fixed internal schema before applying text cleaning
    resume_data = pd.DataFrame(
        {
            "resume_id": np.arange(len(resume_raw)),
            "resume_text_raw": resume_raw[text_column].fillna("").astype(str),
            "resume_category": category_values,
        }
    )
    # Removing unreadable short records and exact duplicate resume text
    resume_data["resume_text"] = resume_data["resume_text_raw"].map(clean_text)
    resume_data = resume_data[resume_data["resume_text"].str.len() >= 30]
    resume_data = resume_data.drop_duplicates("resume_text").reset_index(drop=True)
    if resume_data.empty:
        raise ValueError("No usable resume records remained after text cleaning.")
    return resume_data


def prepare_job_data(job_raw: pd.DataFrame) -> pd.DataFrame:
    """Standardizing and cleaning job records for vectorization."""

    # Resolving each optional and required field across common job-table schemas
    job_id_column = find_column(job_raw, JOB_ID_COLUMNS)
    title_column = find_column(job_raw, JOB_TITLE_COLUMNS)
    description_column = find_column(job_raw, JOB_DESCRIPTION_COLUMNS)
    skills_column = find_column(job_raw, JOB_SKILLS_COLUMNS)
    location_column = find_column(job_raw, JOB_LOCATION_COLUMNS)
    company_column = find_column(job_raw, JOB_COMPANY_COLUMNS)
    url_column = find_column(job_raw, JOB_URL_COLUMNS)
    source_column = find_column(job_raw, JOB_SOURCE_COLUMNS)
    posted_column = find_column(job_raw, JOB_POSTED_COLUMNS)
    expires_column = find_column(job_raw, JOB_EXPIRES_COLUMNS)

    if title_column is None or description_column is None:
        raise KeyError("Job title and description columns are required in the selected job table.")

    def values_or_default(column: Optional[str], default: str = "") -> pd.Series:
        """Returning cleaned string values or a repeated default value."""

        if column is None:
            return pd.Series(default, index=job_raw.index, dtype="object")
        return job_raw[column].fillna(default).astype(str)

    # Building a consistent internal job schema without mutating the original input
    job_data = pd.DataFrame(
        {
            "job_id": values_or_default(job_id_column),
            "job_title": values_or_default(title_column, "Untitled Job"),
            "job_description_raw": values_or_default(description_column),
            "job_skills": values_or_default(skills_column),
            "job_location": values_or_default(location_column),
            "job_company": values_or_default(company_column, "Not provided"),
            "job_url": values_or_default(url_column),
            "job_source": values_or_default(source_column, "Local dataset"),
            "job_posted_at": values_or_default(posted_column),
            "job_expires_at": values_or_default(expires_column),
        }
    )
    # Assigning stable local identifiers when a source omits job IDs
    missing_ids = job_data["job_id"].str.strip().eq("")
    job_data.loc[missing_ids, "job_id"] = job_data.index[missing_ids].map(lambda value: f"local_{value}")

    # Inferring recognizable job-board names from available listing links
    inferred_sources = job_data["job_url"].map(infer_job_source)
    missing_sources = job_data["job_source"].str.strip().isin(["", "Local dataset"])
    job_data.loc[missing_sources & inferred_sources.ne(""), "job_source"] = inferred_sources

    # Building LinkedIn listing links when the dataset only provides numeric job identifiers
    linkedin_rows = job_data["job_source"].str.contains("linkedin", case=False, na=False)
    numeric_ids = job_data["job_id"].str.fullmatch(r"\d+", na=False)
    missing_urls = job_data["job_url"].str.strip().eq("")
    fallback_rows = linkedin_rows & numeric_ids & missing_urls
    job_data.loc[fallback_rows, "job_url"] = (
        "https://www.linkedin.com/jobs/view/" + job_data.loc[fallback_rows, "job_id"]
    )

    # Weighting titles twice while combining the available job text fields
    job_data["job_document_raw"] = (
        job_data["job_title"]
        + " "
        + job_data["job_title"]
        + " "
        + job_data["job_description_raw"]
        + " "
        + job_data["job_skills"]
    )
    # Cleaning combined documents before dropping unusable or exact duplicate text
    job_data["job_document"] = job_data["job_document_raw"].map(clean_text)
    job_data = job_data[job_data["job_document"].str.len() >= 30]
    job_data = job_data.drop_duplicates("job_document").reset_index(drop=True)
    if job_data.empty:
        raise ValueError("No usable job records remained after text cleaning.")
    return job_data


def infer_job_source(url: object) -> str:
    """Inferring a job-board label from a listing URL."""

    # Normalizing URL case before checking recognizable public job-board domains
    normalized_url = str(url).lower()
    known_sources = {
        "linkedin.com": "LinkedIn",
        "indeed.com": "Indeed",
        "glassdoor.com": "Glassdoor",
        "ziprecruiter.com": "ZipRecruiter",
        "monster.com": "Monster",
        "dice.com": "Dice",
    }
    for domain, source in known_sources.items():
        if domain in normalized_url:
            return source
    return ""


def read_resume_file(path: Path) -> str:
    """Reading resume text from TXT, PDF, or DOCX files."""

    # Resolving the user path before choosing a parser from the file extension
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Resume file not found: {resolved_path}")

    # Reading text directly while using established parsers for PDF and DOCX files
    suffix = resolved_path.suffix.lower()
    if suffix == ".txt":
        return resolved_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(resolved_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        from docx import Document

        document = Document(str(resolved_path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    raise ValueError("Supported resume file types are .txt, .pdf, and .docx.")


class JobRecommendationModel:
    """Building and applying the TF-IDF cosine-similarity model."""

    def __init__(self, settings: ModelSettings) -> None:
        """Initializing model storage before fitting the vectorizer."""

        # Retaining settings so exported run details can reproduce the vectorizer
        self.settings = settings
        # Configuring explainable word and phrase features with sublinear term frequency
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(settings.ngram_min, settings.ngram_max),
            min_df=settings.min_df,
            max_df=settings.max_df,
            sublinear_tf=True,
            max_features=settings.max_features,
        )
        # Initializing fitted data and sparse-vector holders as empty state
        self.resume_data: Optional[pd.DataFrame] = None
        self.job_data: Optional[pd.DataFrame] = None
        self.resume_vectors = None
        self.job_vectors = None
        self.query_text = ""

    def fit(self, resume_data: pd.DataFrame, job_data: pd.DataFrame) -> None:
        """Fitting one shared vocabulary across resume and job documents."""

        # Combining resume and job documents so inverse-document frequencies share one corpus
        all_documents = pd.concat(
            [resume_data["resume_text"], job_data["job_document"]],
            ignore_index=True,
        )
        # Learning vocabulary weights and transforming every document in one sparse operation
        all_vectors = self.vectorizer.fit_transform(all_documents)
        # Splitting the shared matrix back into resume and job sections by row count
        self.resume_data = resume_data
        self.job_data = job_data
        self.resume_vectors = all_vectors[: len(resume_data)]
        self.job_vectors = all_vectors[len(resume_data) :]

    def create_query_vector(
        self,
        resume_index: Optional[int],
        external_resume_text: Optional[str],
    ):
        """Creating a query vector from a dataset resume or external resume."""

        # Preventing query transformation before the shared vocabulary has been fitted
        if self.resume_data is None or self.resume_vectors is None:
            raise RuntimeError("Fitting the recommendation model is required before creating a query.")

        # Transforming a personal resume without adding it to persistent output files
        if external_resume_text is not None:
            cleaned_resume = clean_text(external_resume_text)
            if len(cleaned_resume) < 30:
                raise ValueError("The supplied resume contains too little usable text after cleaning.")
            self.query_text = cleaned_resume
            return self.vectorizer.transform([cleaned_resume]), "External resume"

        # Defaulting to the first cleaned dataset row for reproducible demonstrations
        selected_index = 0 if resume_index is None else resume_index
        if selected_index < 0 or selected_index >= len(self.resume_data):
            raise IndexError(
                f"Resume index must be between 0 and {len(self.resume_data) - 1} after cleaning."
            )
        category = str(self.resume_data.loc[selected_index, "resume_category"])
        self.query_text = str(self.resume_data.loc[selected_index, "resume_text"])
        return self.resume_vectors[selected_index], f"Dataset resume {selected_index} ({category})"

    def matching_terms(self, query_vector, job_index: int, term_limit: int = 8) -> list[tuple[str, float]]:
        """Calculating shared weighted terms for explaining one match."""

        if self.job_vectors is None:
            raise RuntimeError("Fitting the recommendation model is required before explaining matches.")
        # Multiplying sparse TF-IDF weights to measure each shared term's contribution
        feature_names = np.asarray(self.vectorizer.get_feature_names_out())
        contribution = query_vector.multiply(self.job_vectors[job_index]).toarray().ravel()
        # Keeping only the strongest positive shared terms for a compact explanation
        ranked = np.argsort(contribution)[::-1]
        ranked = ranked[contribution[ranked] > 0][:term_limit]
        return [(str(feature_names[index]), float(contribution[index])) for index in ranked]

    def recommend(
        self,
        query_vector,
        top_n: int,
        location_filter: Optional[str] = None,
        title_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """Ranking eligible jobs and returning recommendations with explanations."""

        # Requiring fitted job vectors before applying filters and similarity calculations
        if self.job_data is None or self.job_vectors is None:
            raise RuntimeError("Fitting the recommendation model is required before ranking jobs.")
        if top_n < 1:
            raise ValueError("Top-N must be at least 1.")

        # Starting with every job eligible before applying optional contains filters
        eligible = np.ones(len(self.job_data), dtype=bool)
        if location_filter:
            eligible &= self.job_data["job_location"].str.contains(
                location_filter, case=False, regex=False, na=False
            ).to_numpy()
        if title_filter:
            eligible &= self.job_data["job_title"].str.contains(
                title_filter, case=False, regex=False, na=False
            ).to_numpy()
        if source_filter:
            eligible &= self.job_data["job_source"].str.contains(
                source_filter, case=False, regex=False, na=False
            ).to_numpy()

        # Converting the Boolean mask into source row positions used by the sparse matrix
        eligible_indices = np.flatnonzero(eligible)
        if len(eligible_indices) == 0:
            raise ValueError("No jobs matched the supplied title, location, and source filters.")

        # Calculating resume-to-job cosine similarity across all remaining sparse vectors
        scores = cosine_similarity(query_vector, self.job_vectors[eligible_indices]).ravel()
        # Sorting descending with a stable algorithm so equal scores remain reproducible
        ranked_positions = np.argsort(-scores, kind="stable")[: min(top_n, len(scores))]
        ranked_job_indices = eligible_indices[ranked_positions]

        result_columns = [
            "job_id",
            "job_title",
            "job_company",
            "job_location",
            "job_source",
            "job_url",
            "job_skills",
            "job_description_raw",
            "job_posted_at",
            "job_expires_at",
        ]
        # Copying human-readable fields before adding rank and model score columns
        recommendations = self.job_data.loc[ranked_job_indices, result_columns].copy()
        recommendations.insert(0, "rank", np.arange(1, len(recommendations) + 1))
        recommendations["cosine_similarity"] = scores[ranked_positions]

        # Preparing row-level and compact string explanations for every selected job
        explanation_rows: list[dict[str, object]] = []
        matched_term_strings: list[str] = []
        matched_skill_strings: list[str] = []
        missing_skill_strings: list[str] = []
        skill_coverages: list[float] = []
        # Extracting resume skills once before comparing them with each recommendation
        resume_skills = extract_known_skills(self.query_text)
        for rank, job_index in enumerate(ranked_job_indices, start=1):
            # Collecting weighted terms from the model and literal skills from the vocabulary
            terms = self.matching_terms(query_vector, int(job_index))
            matched_term_strings.append(", ".join(term for term, _ in terms))
            job_text = self.job_data.loc[job_index, "job_document_raw"]
            job_skills = extract_known_skills(job_text)
            matched_skills = sorted(resume_skills.intersection(job_skills))
            missing_skills = sorted(job_skills.difference(resume_skills))
            # Reporting coverage as an explanation without changing the cosine ranking
            skill_coverages.append(len(matched_skills) / len(job_skills) if job_skills else np.nan)
            matched_skill_strings.append(", ".join(matched_skills))
            missing_skill_strings.append(", ".join(missing_skills[:10]))
            # Storing one detailed row per contributing term for downloadable evidence
            for term, contribution in terms:
                explanation_rows.append(
                    {
                        "rank": rank,
                        "job_id": self.job_data.loc[job_index, "job_id"],
                        "job_title": self.job_data.loc[job_index, "job_title"],
                        "term": term,
                        "contribution": contribution,
                    }
                )
        recommendations["matched_terms"] = matched_term_strings
        recommendations["matched_skills"] = matched_skill_strings
        recommendations["missing_skills"] = missing_skill_strings
        recommendations["skill_coverage"] = skill_coverages
        return recommendations.reset_index(drop=True), scores, pd.DataFrame(explanation_rows)


def save_recommendation_plot(recommendations: pd.DataFrame, output_path: Path) -> None:
    """Saving a readable Top-N score chart for reporting results."""

    # Sorting ascending so the highest horizontal bar appears at the top of the figure
    plotted = recommendations.sort_values("cosine_similarity").copy()
    plotted["display_title"] = plotted.apply(
        lambda row: f"{int(row['rank'])}. {str(row['job_title'])[:65]}", axis=1
    )
    figure_height = max(5.0, 0.55 * len(plotted))
    plt.figure(figsize=(10, figure_height))
    plt.barh(plotted["display_title"], plotted["cosine_similarity"], edgecolor="black")
    plt.title("Top Recommended Jobs Using TF-IDF and Cosine Similarity")
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Job Title")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def export_results(
    recommendations: pd.DataFrame,
    explanation: pd.DataFrame,
    eligible_scores: np.ndarray,
    output_directory: Path,
    summary: dict[str, object],
    create_plot: bool,
) -> dict[str, Path]:
    """Exporting recommendation tables, explanations, summary data, and a figure."""

    # Creating the output folder only when a run produces exportable results
    output_directory.mkdir(parents=True, exist_ok=True)
    recommendation_path = output_directory / "job_recommendations.csv"
    explanation_path = output_directory / "matching_term_explanations.csv"
    summary_path = output_directory / "run_summary.json"

    # Saving recommendations, term contributions, and reproducibility metadata separately
    recommendations.to_csv(recommendation_path, index=False)
    explanation.to_csv(explanation_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    paths = {
        "recommendations": recommendation_path,
        "explanations": explanation_path,
        "summary": summary_path,
    }
    # Creating the report chart only when the caller has not disabled plotting
    if create_plot:
        plot_path = output_directory / "top_recommendations.png"
        save_recommendation_plot(recommendations, plot_path)
        paths["plot"] = plot_path
    return paths


def build_argument_parser() -> argparse.ArgumentParser:
    """Defining command-line options for running the local system."""

    # Preserving a command-line fallback while the browser remains the main interface
    parser = argparse.ArgumentParser(
        description="Recommending LinkedIn jobs from resume text using TF-IDF and cosine similarity."
    )
    parser.add_argument("--project-dir", type=Path, help="Capstone folder containing Resume and Jobs")
    parser.add_argument("--resume-data", type=Path, help="Optional explicit resume CSV or Excel table")
    parser.add_argument("--job-data", type=Path, help="Optional explicit job CSV or Excel table")

    resume_source = parser.add_mutually_exclusive_group()
    resume_source.add_argument("--resume-index", type=int, help="Cleaned dataset resume index, defaulting to 0")
    resume_source.add_argument("--resume-file", type=Path, help="External resume in TXT, PDF, or DOCX format")
    resume_source.add_argument("--resume-text", type=str, help="External resume text supplied directly")

    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations, defaulting to 10")
    parser.add_argument("--max-resumes", type=int, default=500, help="Maximum sampled resumes")
    parser.add_argument("--max-jobs", type=int, default=10000, help="Maximum sampled jobs")
    parser.add_argument("--location", type=str, help="Optional case-insensitive location filter")
    parser.add_argument("--title", type=str, help="Optional case-insensitive job-title filter")
    parser.add_argument("--source", type=str, help="Optional case-insensitive job-source filter")
    parser.add_argument("--output-dir", type=Path, help="Output folder for CSV, JSON, and PNG files")
    parser.add_argument("--no-plot", action="store_true", help="Skipping PNG chart generation")
    return parser


def run(arguments: argparse.Namespace) -> dict[str, Path]:
    """Running data loading, model fitting, ranking, and result exporting."""

    # Rejecting invalid limits before reading any large dataset files
    if arguments.max_resumes < 1 or arguments.max_jobs < 1:
        raise ValueError("Maximum resume and job counts must be positive integers.")

    # Resolving the project folder while allowing explicit data files from any location
    both_data_paths_given = arguments.resume_data is not None and arguments.job_data is not None
    if both_data_paths_given and arguments.project_dir is None:
        project_directory = Path.cwd().resolve()
    else:
        project_directory = find_project_directory(Path.cwd(), arguments.project_dir)

    resume_path = select_dataset_path(project_directory, "resume", arguments.resume_data)
    job_path = select_dataset_path(project_directory, "job", arguments.job_data)
    output_directory = (
        arguments.output_dir.expanduser().resolve()
        if arguments.output_dir is not None
        else project_directory / "final_system_outputs"
    )

    settings = ModelSettings(
        max_resumes=arguments.max_resumes,
        max_jobs=arguments.max_jobs,
        top_n=arguments.top_n,
    )

    print(f"Loading resume data: {resume_path}")
    resume_raw = load_table(resume_path)
    resume_raw = sample_rows(resume_raw, settings.max_resumes, settings.random_state)

    print(f"Loading job data: {job_path}")
    job_read_limit = max(settings.max_jobs * 4, 30000)
    job_raw = load_table(job_path, row_limit=job_read_limit)
    job_raw = sample_rows(job_raw, settings.max_jobs, settings.random_state)

    print("Cleaning and standardizing text fields...")
    resume_data = prepare_resume_data(resume_raw)
    job_data = prepare_job_data(job_raw)

    print("Fitting the TF-IDF unigram and bigram model...")
    model = JobRecommendationModel(settings)
    model.fit(resume_data, job_data)

    external_resume_text: Optional[str] = None
    if arguments.resume_file is not None:
        external_resume_text = read_resume_file(arguments.resume_file)
    elif arguments.resume_text is not None:
        external_resume_text = arguments.resume_text

    query_vector, query_label = model.create_query_vector(arguments.resume_index, external_resume_text)
    print(f"Ranking jobs for: {query_label}")
    recommendations, eligible_scores, explanation = model.recommend(
        query_vector=query_vector,
        top_n=settings.top_n,
        location_filter=arguments.location,
        title_filter=arguments.title,
        source_filter=arguments.source,
    )

    summary: dict[str, object] = {
        "method": "TF-IDF unigram and bigram vectors with cosine similarity",
        "resume_data_file": str(resume_path),
        "job_data_file": str(job_path),
        "query": query_label,
        "cleaned_resume_count": int(len(resume_data)),
        "cleaned_job_count": int(len(job_data)),
        "eligible_job_count": int(len(eligible_scores)),
        "vocabulary_size": int(len(model.vectorizer.get_feature_names_out())),
        "median_similarity": float(np.median(eligible_scores)),
        "maximum_similarity": float(np.max(eligible_scores)),
        "top_job_title": str(recommendations.loc[0, "job_title"]),
        "top_job_score": float(recommendations.loc[0, "cosine_similarity"]),
        "location_filter": arguments.location,
        "title_filter": arguments.title,
        "source_filter": arguments.source,
        "settings": asdict(settings),
        "score_interpretation": "Using scores for ranking rather than treating them as probabilities.",
    }
    paths = export_results(
        recommendations=recommendations,
        explanation=explanation,
        eligible_scores=eligible_scores,
        output_directory=output_directory,
        summary=summary,
        create_plot=not arguments.no_plot,
    )

    display_columns = [
        "rank",
        "job_title",
        "job_company",
        "job_location",
        "cosine_similarity",
        "matched_terms",
    ]
    print("\nTop recommendations")
    print(
        recommendations[display_columns].to_string(
            index=False,
            formatters={"cosine_similarity": lambda score: f"{score:.4f}"},
        )
    )
    print(f"\nSaving results in: {output_directory}")
    return paths


def main() -> int:
    """Parsing arguments and reporting expected local setup errors clearly."""

    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        run(arguments)
    except (FileNotFoundError, KeyError, ValueError, IndexError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
