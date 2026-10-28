"""Building the browser part of the AI job recommendation project.

Connecting the page controls, resume readers, job APIs, TF-IDF model, result
cards, and evaluation table in one file is making the complete workflow easier
to follow during the capstone demonstration. Keeping the resume on the local
computer and keeping credentials out of exported results is supporting the
privacy focus of the project.
"""

from __future__ import annotations

# Importing built-in Python tools for handling files, dates, JSON, and type hints
import io
import json
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Importing data and interface libraries used throughout the application
import numpy as np
import pandas as pd
import streamlit as st

# Importing evaluation functions for building rating rows and calculating metrics
from evaluation import build_judgment_template, evaluate_judgments

# Importing recommendation functions for cleaning, modeling, ranking, and exporting
from job_recommender import (
    JOB_SOURCE_COLUMNS,
    JobRecommendationModel,
    ModelSettings,
    clean_text,
    export_results,
    find_column,
    load_table,
    prepare_job_data,
    prepare_resume_data,
    sample_rows,
)

# Importing job-source functions for retrieving, caching, and combining API jobs
from job_sources import (
    JobSourceError,
    ResponseCache,
    SourceResult,
    combine_job_frames,
    fetch_adzuna,
    fetch_usajobs,
)


# Configuring the browser page before creating any visible Streamlit element
# Keeping this call near the top because Streamlit requires page setup to run first
st.set_page_config(
    page_title="AI-Based Job Recommendation System",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Applying simple styling so the application is readable without looking too flashy
# Hiding Streamlit's running figure and spinner because they distract during processing
st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
    [data-testid="stMetric"] {background: transparent; border: 1px solid #6b7280; padding: 12px; border-radius: 8px;}
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color: inherit;}
    [data-testid="stSidebar"] {border-right: 1px solid #e4e7eb;}
    [data-testid="stStatusWidget"] {display: none !important;}
    [data-testid="stSpinner"] {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


# Finding the folder containing app.py so paths work from any starting location
APP_DIRECTORY = Path(__file__).resolve().parent

# Creating fixed local paths for the public-job cache and generated result files
CACHE_PATH = APP_DIRECTORY / "runtime_data" / "job_cache.sqlite3"
OUTPUT_DIRECTORY = APP_DIRECTORY / "final_system_outputs"

# Listing file extensions accepted by the two different upload controls
SUPPORTED_TABLE_TYPES = ["csv", "xlsx", "xls"]
SUPPORTED_RESUME_TYPES = ["pdf", "docx", "txt"]

# Mapping readable country names to the short codes required by the Adzuna API
ADZUNA_COUNTRIES = {
    "United States": "us",
    "United Kingdom": "gb",
    "Canada": "ca",
    "Australia": "au",
    "India": "in",
    "Germany": "de",
    "France": "fr",
    "Netherlands": "nl",
    "New Zealand": "nz",
    "Poland": "pl",
    "Singapore": "sg",
    "South Africa": "za",
}


# Allowing the evaluator to change only the rating and note columns
# Protecting job details from accidental editing inside the Streamlit table
EVALUATION_EDITABLE_COLUMNS = {"relevance", "judgment_note"}


def find_local_dataset(filename_parts: tuple[str, ...]) -> Optional[Path]:
    """Finding a known dataset path in the project folder or its parent folders."""

    # Remembering checked folders because the app path and working path can overlap
    checked_folders: set[Path] = set()

    # Starting from both the app folder and the folder used to launch Streamlit
    for starting_folder in [APP_DIRECTORY, Path.cwd().resolve()]:
        # Moving upward through each parent folder to find the expected data folder
        for folder in [starting_folder, *starting_folder.parents]:
            # Skipping a folder when it has already been checked from another route
            if folder in checked_folders:
                continue
            checked_folders.add(folder)

            # Joining pieces such as Jobs and postings.csv into one possible path
            candidate = folder.joinpath(*filename_parts)

            # Returning immediately after finding a real file at the possible path
            if candidate.is_file():
                return candidate

    # Returning no path when the expected dataset cannot be found automatically
    return None


def configured_value(name: str) -> str:
    """Reading one API credential without showing it in exported application data."""

    # Reading an environment variable first because deployments often store secrets there
    environment_value = os.getenv(name, "").strip()

    # Returning the environment value when it is present and not only blank spaces
    if environment_value:
        return environment_value
    try:
        # Falling back to the private Streamlit secrets file on the local computer
        return str(st.secrets.get(name, "")).strip()
    except (FileNotFoundError, KeyError, AttributeError):
        # Returning a blank value when no private credential has been configured
        return ""


def infer_source_from_filename(filename: str) -> str:
    """Creating a readable job-source name from an uploaded table filename."""

    # Converting the filename to lowercase so matching is not case-sensitive
    lowered = filename.lower()

    # Connecting common filename words with readable source labels for result cards
    sources = {
        "linkedin": "LinkedIn dataset",
        "indeed": "Indeed export",
        "glassdoor": "Glassdoor export",
        "adzuna": "Adzuna export",
        "usajobs": "USAJOBS export",
    }

    # Checking each known marker and returning the first matching source label
    for marker, source in sources.items():
        if marker in lowered:
            return source

    # Treating the original capstone postings.csv file as the LinkedIn research dataset
    if lowered == "postings.csv":
        return "LinkedIn dataset"

    # Building a general label from the filename when no known source word is present
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def read_uploaded_table(filename: str, file_bytes: bytes, row_limit: Optional[int] = None) -> pd.DataFrame:
    """Reading an uploaded CSV or Excel job table from browser-provided bytes."""

    # Reading the extension so pandas can use the correct table reader
    suffix = Path(filename).suffix.lower()

    # Wrapping uploaded bytes in memory instead of saving the private upload to disk
    buffer = io.BytesIO(file_bytes)

    # Loading CSV rows while limiting memory use for large capstone datasets
    if suffix == ".csv":
        try:
            return pd.read_csv(buffer, nrows=row_limit, low_memory=False)
        except UnicodeDecodeError:
            # Moving back to the beginning before retrying an older CSV encoding
            buffer.seek(0)

            # Retrying with Latin-1 because some downloaded datasets are not UTF-8
            return pd.read_csv(buffer, nrows=row_limit, low_memory=False, encoding="latin-1")

    # Loading Excel rows when the uploaded name ends with xlsx or xls
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(buffer, nrows=row_limit)

    # Stopping with a clear message instead of allowing an unknown format to fail later
    raise ValueError(f"Unsupported table type: {suffix}")


def read_uploaded_resume(filename: str, file_bytes: bytes) -> str:
    """Reading resume text locally from an uploaded PDF, DOCX, or TXT file."""

    # Reading the extension to select the correct local resume parser
    suffix = Path(filename).suffix.lower()

    # Keeping the uploaded resume in memory instead of creating a permanent copy
    buffer = io.BytesIO(file_bytes)

    # Decoding a plain-text resume while ignoring uncommon broken characters
    if suffix == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")

    # Extracting text from every page when the uploaded resume is a PDF
    if suffix == ".pdf":
        from pypdf import PdfReader

        # Creating a PDF reader around the in-memory uploaded file
        reader = PdfReader(buffer)

        # Joining page text into one document used by the recommendation model
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    # Extracting paragraph text when the uploaded resume is a Word document
    if suffix == ".docx":
        from docx import Document

        # Opening the Word document directly from the in-memory file
        document = Document(buffer)

        # Joining paragraphs because TF-IDF needs plain text instead of Word formatting
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    # Stopping unsupported resume files with a message the user can understand
    raise ValueError("Supported resume types are PDF, DOCX, and TXT.")


def build_single_resume_corpus(resume_text: str) -> pd.DataFrame:
    """Building a one-row resume table from a personal uploaded resume."""

    # Cleaning the personal resume with the same rules used for dataset resumes
    cleaned_resume = clean_text(resume_text)

    # Rejecting nearly empty text because it cannot produce a useful TF-IDF vector
    if len(cleaned_resume) < 30:
        raise ValueError("The uploaded resume contains too little readable text.")

    # Returning the personal resume in the column structure expected by the model
    return pd.DataFrame(
        {
            "resume_id": [0],
            "resume_text_raw": [resume_text],
            "resume_category": ["Personal upload"],
            "resume_text": [cleaned_resume],
        }
    )


def standardize_local_jobs(raw_jobs: pd.DataFrame, default_source: str) -> pd.DataFrame:
    """Converting a local job table into the same columns used by API results."""

    # Copying the uploaded data so the original DataFrame is not changed unexpectedly
    working = raw_jobs.copy()

    # Adding a source column only when the uploaded table does not already provide one
    if find_column(working, JOB_SOURCE_COLUMNS) is None:
        working["_input_source"] = default_source

    # Reusing the shared cleaning function so local and live jobs follow the same rules
    prepared = prepare_job_data(working)

    # Renaming model columns into the common source schema used before final preparation
    return prepared.rename(
        columns={
            "job_title": "title",
            "job_description_raw": "description",
            "job_skills": "skills_desc",
            "job_location": "location",
            "job_company": "company_name",
            "job_url": "job_posting_url",
            "job_source": "source",
            "job_posted_at": "posted_at",
            "job_expires_at": "expires_at",
        }
    )[
        [
            "job_id",
            "title",
            "description",
            "skills_desc",
            "location",
            "company_name",
            "job_posting_url",
            "source",
            "posted_at",
            "expires_at",
        ]
    ]


def filter_recent_jobs(jobs: pd.DataFrame, maximum_age_days: int) -> pd.DataFrame:
    """Filtering old live jobs while keeping imported research-dataset rows."""

    # Returning unchanged data when no posting-date column is available
    if "posted_at" not in jobs.columns:
        return jobs

    # Converting different API date formats into one timezone-aware datetime series
    posted = pd.to_datetime(jobs["posted_at"], errors="coerce", utc=True, format="mixed")

    # Calculating the oldest acceptable live posting time from the selected age limit
    cutoff = datetime.now(timezone.utc) - timedelta(days=maximum_age_days)

    # Identifying Adzuna and USAJOBS rows because imported datasets may be historical
    live_source = jobs["source"].astype(str).isin({"Adzuna", "USAJOBS"})

    # Keeping imported rows, undated live rows, and dated live rows newer than the cutoff
    eligible = ~live_source | posted.isna() | posted.ge(cutoff)

    # Resetting row numbers after removing jobs so later indexing stays predictable
    return jobs.loc[eligible].reset_index(drop=True)


def format_date(value: object) -> str:
    """Formatting one posting date for showing on a recommendation card."""

    # Returning a friendly message when the source did not provide a date
    if not str(value).strip():
        return "Date not provided"

    # Converting different source date formats into one safe datetime value
    parsed = pd.to_datetime(value, errors="coerce", utc=True)

    # Returning the same friendly message when pandas cannot understand the date
    if pd.isna(parsed):
        return "Date not provided"

    # Showing only the calendar date because the exact API time is not important here
    return parsed.strftime("%Y-%m-%d")


def valid_listing_url(value: object) -> bool:
    """Checking whether a job link starts with a normal HTTP or HTTPS protocol."""

    # Accepting only ordinary web links before creating an application button
    return str(value).strip().lower().startswith(("https://", "http://"))


def shorten_text(value: object, limit: int = 480) -> str:
    """Shortening a long job description while avoiding half-finished words."""

    # Replacing repeated spaces and line breaks with single spaces for a cleaner card
    normalized = " ".join(str(value).split())

    # Returning the full description when it already fits inside the selected limit
    if len(normalized) <= limit:
        return normalized

    # Cutting at the last complete word and adding an ellipsis to show missing text
    return normalized[:limit].rsplit(" ", 1)[0] + "…"


def create_summary(
    settings: ModelSettings,
    model: JobRecommendationModel,
    recommendations: pd.DataFrame,
    eligible_scores: np.ndarray,
    search_settings: dict[str, object],
    source_results: list[SourceResult],
) -> dict[str, object]:
    """Creating a summary of model settings and results for one completed run."""

    # Recording settings and measurements needed for reproducing and explaining the run
    # Excluding API credentials and resume text because they are private information
    return {
        "method": "TF-IDF unigrams and bigrams with cosine similarity",
        "ranking_basis": "Cosine similarity only; skills are explanations and dates are filters",
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "cleaned_resume_count": int(len(model.resume_data)),
        "cleaned_job_count": int(len(model.job_data)),
        "eligible_job_count": int(len(eligible_scores)),
        "vocabulary_size": int(len(model.vectorizer.get_feature_names_out())),
        "median_similarity": float(np.median(eligible_scores)),
        "maximum_similarity": float(np.max(eligible_scores)),
        "top_job_title": str(recommendations.loc[0, "job_title"]),
        "top_job_score": float(recommendations.loc[0, "cosine_similarity"]),
        "search_settings": search_settings,
        "source_results": [
            {
                "source": result.source,
                "job_count": int(len(result.jobs)),
                "fetched_at": result.fetched_at,
                "from_cache": result.from_cache,
            }
            for result in source_results
        ],
        "model_settings": asdict(settings),
        "score_interpretation": "Using scores for ranking rather than treating them as hiring probabilities.",
    }


def render_results(recommendations: pd.DataFrame, summary: dict[str, object]) -> None:
    """Showing summary metrics, downloads, ranked jobs, and matching explanations."""

    # Dividing the page into four equal spaces for the main summary measurements
    metric_columns = st.columns(4)

    # Showing how much job data and vocabulary the latest TF-IDF run actually used
    metric_columns[0].metric("Jobs analyzed", f"{summary['cleaned_job_count']:,}")
    metric_columns[1].metric("Vocabulary", f"{summary['vocabulary_size']:,}")

    # Converting the highest cosine similarity from a decimal into a readable percentage
    metric_columns[2].metric("Top TF-IDF match", f"{100 * summary['top_job_score']:.1f}%")

    # Counting successful sources so the user knows how many data inputs contributed
    metric_columns[3].metric("Sources used", len(summary["source_results"]))

    # Reminding the user that similarity is not predicting an employer's hiring decision
    st.caption("Similarity scores are ranking values, not interview or hiring probabilities.")

    # Copying only the columns required for building the recommendation score chart
    chart = recommendations[["rank", "job_title", "cosine_similarity"]].copy()

    # Combining rank and title into labels that remain understandable on the chart
    chart["job"] = chart.apply(lambda row: f"{int(row['rank'])}. {row['job_title']}", axis=1)

    # Converting decimal similarity values into percentages before plotting them
    chart["match_percent"] = chart["cosine_similarity"] * 100

    # Plotting the same values used for ranking so the chart does not introduce a new score
    st.bar_chart(chart.set_index("job")["match_percent"], color="#3568a8")

    # Creating side-by-side download buttons while leaving extra space on the right
    download_columns = st.columns([1, 1, 2])

    # Converting recommendation rows into CSV bytes for spreadsheet use
    download_columns[0].download_button(
        "Download results",
        data=recommendations.to_csv(index=False).encode("utf-8"),
        file_name="job_recommendations.csv",
        mime="text/csv",
        width="stretch",
    )

    # Converting run settings into JSON bytes for reproducibility and report evidence
    download_columns[1].download_button(
        "Download run details",
        data=json.dumps(summary, indent=2).encode("utf-8"),
        file_name="run_summary.json",
        mime="application/json",
        width="stretch",
    )

    # Starting the detailed result section after the chart and download controls
    st.subheader("Ranked current listings")

    # Repeating one bordered card for every recommendation in final rank order
    for row in recommendations.itertuples(index=False):
        with st.container(border=True):
            # Separating the job heading from the matching score for easier reading
            heading_column, score_column = st.columns([5, 1])
            heading_column.markdown(f"### {int(row.rank)}. {row.job_title}")
            score_column.metric("TF-IDF match", f"{100 * float(row.cosine_similarity):.1f}%")

            # Showing important job identity information in one compact line
            st.caption(
                f"{row.job_company} · {row.job_location or 'Location not provided'} · "
                f"{row.job_source or 'Source not provided'} · Posted: {format_date(row.job_posted_at)}"
            )

            # Showing terms that directly contributed weight to the TF-IDF similarity score
            if row.matched_terms:
                st.write(f"**Contributing terms:** {row.matched_terms}")

            # Showing recognizable skills present in both the resume and the job description
            if row.matched_skills:
                st.write(f"**Skills found in both:** {row.matched_skills}")

            # Showing job skills not detected in the resume as an explanation, not a rejection
            if row.missing_skills:
                st.write(f"**Job skills not found in the resume:** {row.missing_skills}")

            # Showing a shortened job description so many results can fit on one page
            st.write(shorten_text(row.job_description_raw))

            # Creating a link button only when the source supplied a safe web address
            if valid_listing_url(row.job_url):
                st.link_button("Open original listing", str(row.job_url))
            else:
                # Explaining missing links instead of showing a broken application button
                st.caption("No usable application URL was supplied by this source.")


def source_report_table(results: list[SourceResult]) -> pd.DataFrame:
    """Building a small table showing where the current job records came from.

    Showing the retrieval time and cache mode is helping document whether each
    source used a current API response, a one-hour local cache entry, or a local
    table during the recommendation run.
    """

    # Turning each SourceResult object into one simple row for the browser table
    return pd.DataFrame(
        [
            {
                "Source": result.source,
                "Jobs": len(result.jobs),
                "Retrieved UTC": result.fetched_at[:19].replace("T", " "),
                "Mode": "Cache" if result.from_cache else "Live API/local file",
            }
            for result in results
        ]
    )


def prepare_imported_judgments(raw_judgments: pd.DataFrame) -> pd.DataFrame:
    """Preparing a saved evaluation CSV for continuing ratings in the browser.

    Keeping descriptive columns is allowing an evaluator to continue earlier
    work, while converting ranks and ratings is preventing broken spreadsheet
    values from entering the evaluation calculations.
    """

    # Listing the minimum columns needed for identifying jobs and storing ratings
    required_columns = {"query_id", "rank", "job_title", "relevance"}

    # Finding any required column names that are missing from the uploaded CSV
    missing_columns = required_columns.difference(raw_judgments.columns)

    # Stopping early with an exact list when the uploaded file has the wrong structure
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"The evaluation CSV is missing required columns: {missing_text}.")

    # Copying the uploaded table so validation does not modify the original object
    prepared = raw_judgments.copy()

    # Converting rank values into numbers and marking unreadable values as missing
    prepared["rank"] = pd.to_numeric(prepared["rank"], errors="coerce")

    # Rejecting the file when any recommendation row does not have a usable rank
    if prepared["rank"].isna().any():
        raise ValueError("Every uploaded evaluation row requires a numeric rank.")

    # Converting ratings into numbers while keeping blank unfinished ratings as missing
    prepared["relevance"] = pd.to_numeric(prepared["relevance"], errors="coerce")

    # Selecting only completed ratings so blank work can still be loaded and continued
    completed_ratings = prepared["relevance"].dropna()

    # Rejecting decimals and values outside the allowed zero-to-two rating scale
    if not completed_ratings.between(0, 2).all() or not np.equal(completed_ratings, np.floor(completed_ratings)).all():
        raise ValueError("Completed relevance ratings must be whole numbers: 0, 1, or 2.")

    # Using pandas nullable integers so blank ratings and whole-number ratings can coexist
    prepared["relevance"] = prepared["relevance"].astype("Int64")

    # Adding an empty note column when an older evaluation file does not contain one
    if "judgment_note" not in prepared.columns:
        prepared["judgment_note"] = ""

    # Returning the validated table only after all required checks are completed
    return prepared


def replace_evaluation_table(judgments: pd.DataFrame) -> None:
    """Replacing saved browser ratings without keeping an old editor widget.

    Increasing the editor version is making Streamlit build a fresh table when
    a new recommendation run or uploaded CSV replaces earlier ratings.
    """

    # Saving a separate copy because Streamlit reruns the complete script after interactions
    st.session_state["evaluation_judgments"] = judgments.copy()

    # Increasing a version number so the data editor receives a new Streamlit widget key
    st.session_state["evaluation_editor_version"] = int(
        st.session_state.get("evaluation_editor_version", 0)
    ) + 1

    # Removing old metrics because they no longer describe the newly replaced ratings
    st.session_state.pop("evaluation_metrics", None)


def render_evaluation_page() -> None:
    """Showing manual relevance ratings, CSV continuation, and evaluation results.

    Hiding TF-IDF scores while rating is reducing confirmation bias. Showing
    job identity and description fields is giving the evaluator enough evidence
    for an independent decision. Keeping calculations behind a button is making
    sure incomplete ratings are not reported as final evidence.
    """

    # Explaining the zero-to-two scale before showing the editable rating table
    st.subheader("Assigning human relevance ratings")
    st.markdown(
        """
        Evaluating the latest recommendations independently using the uploaded resume:

        - **Selecting 0** when the job is irrelevant to the resume and target role.
        - **Selecting 1** when the job is partly relevant but has important mismatches.
        - **Selecting 2** when the job is highly relevant in role, skills, and experience level.
        - **Ignoring the model's rank** while judging and using the optional note to record a reason.
        """
    )

    # Reminding the evaluator to download ratings because browser session data is temporary
    st.caption("Keeping ratings in the current browser session unless the evaluation CSV is downloaded.")

    # Providing an uploader for continuing a previously downloaded evaluation file
    uploaded_evaluation = st.file_uploader(
        "Continue from a downloaded evaluation CSV",
        type=["csv"],
        key="evaluation_csv_upload",
    )

    # Enabling the load button only after a CSV file has actually been selected
    if st.button(
        "Load evaluation CSV",
        disabled=uploaded_evaluation is None,
        key="load_evaluation_csv",
    ):
        try:
            # Reading uploaded CSV bytes into a DataFrame without saving another local copy
            imported = read_uploaded_table(
                uploaded_evaluation.name,
                uploaded_evaluation.getvalue(),
            )

            # Validating the CSV and replacing the current browser rating state
            replace_evaluation_table(prepare_imported_judgments(imported))

            # Rerunning the page immediately so the new ratings appear in the editor
            st.rerun()
        except (KeyError, ValueError) as error:
            # Showing a readable file-format or validation message inside the page
            st.error(str(error))

    # Stopping page rendering when there are no recommendations or imported ratings yet
    if "evaluation_judgments" not in st.session_state:
        st.info("Running a recommendation on the Find jobs page creates the rating table automatically.")
        return

    # Offering a reset only when recommendation results still exist in browser memory
    if "recommendations" in st.session_state and st.button(
        "Reset ratings from latest recommendations",
        key="reset_evaluation_ratings",
    ):
        # Reading the latest resume label so reset rows remain connected to the same query
        summary = st.session_state.get("run_summary", {})
        query_label = str(summary.get("search_settings", {}).get("resume_query", "Current resume"))

        # Rebuilding blank ratings from the exact latest recommendation rows
        replace_evaluation_table(
            build_judgment_template(
                st.session_state["recommendations"],
                query_id="resume_1",
                query_label=query_label,
            )
        )

        # Rerunning the script so the cleared dropdown values appear immediately
        st.rerun()

    # Copying stored judgments before preparing a safe version for the editable table
    judgments = st.session_state["evaluation_judgments"].copy()

    # Listing columns shown to the evaluator while intentionally leaving out model scores
    display_columns = [
        "rank",
        "job_title",
        "job_company",
        "job_location",
        "job_source",
        "job_description_raw",
        "relevance",
        "judgment_note",
    ]

    # Adding blank display fields when older CSV versions do not include every newer column
    for column in display_columns:
        if column not in judgments.columns:
            judgments[column] = pd.NA if column == "relevance" else ""

    # Converting ratings into nullable whole numbers for the zero-to-two dropdown
    judgments["relevance"] = pd.to_numeric(judgments["relevance"], errors="coerce").astype("Int64")

    # Copying only visible columns so hidden score columns never enter the rating editor
    editor_source = judgments[display_columns].copy()

    # Listing text columns separately because blank CSV values may be read as numeric NaN values
    text_display_columns = [
        "job_title",
        "job_company",
        "job_location",
        "job_source",
        "job_description_raw",
        "judgment_note",
    ]

    # Converting every visible text field into strings to prevent Streamlit type errors
    for column in text_display_columns:
        editor_source[column] = editor_source[column].fillna("").astype(str)

    # Reading the editor version used to create a fresh widget after loading or resetting data
    editor_version = int(st.session_state.get("evaluation_editor_version", 0))

    # Building the interactive table while locking job evidence columns against editing
    edited_display = st.data_editor(
        editor_source,
        hide_index=True,
        num_rows="fixed",
        width="stretch",
        height=min(720, 95 + 36 * max(len(editor_source), 1)),
        disabled=[column for column in display_columns if column not in EVALUATION_EDITABLE_COLUMNS],
        column_config={
            "rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "job_title": st.column_config.TextColumn("Job title", width="medium"),
            "job_company": st.column_config.TextColumn("Company", width="medium"),
            "job_location": st.column_config.TextColumn("Location", width="medium"),
            "job_source": st.column_config.TextColumn("Source", width="small"),
            "job_description_raw": st.column_config.TextColumn("Job description", width="large"),
            "relevance": st.column_config.SelectboxColumn(
                "Relevance",
                help="Selecting 0 for irrelevant, 1 for partly relevant, or 2 for highly relevant.",
                options=[0, 1, 2],
                required=False,
                width="small",
            ),
            "judgment_note": st.column_config.TextColumn(
                "Optional reason",
                help="Recording the main reason for the rating.",
                width="medium",
            ),
        },
        key=f"evaluation_editor_{editor_version}",
    )

    # Copying only the user-controlled rating values back into the complete judgment table
    judgments["relevance"] = pd.to_numeric(edited_display["relevance"], errors="coerce").astype("Int64")

    # Copying optional explanations while turning remaining blank values into empty strings
    judgments["judgment_note"] = edited_display["judgment_note"].fillna("").astype(str)

    # Saving edits in Streamlit session state so ratings survive normal page reruns
    st.session_state["evaluation_judgments"] = judgments

    # Counting only filled ratings instead of incorrectly treating blanks as zero ratings
    completed_count = int(judgments["relevance"].notna().sum())
    total_count = int(len(judgments))

    # Showing exact completion numbers and a visual progress bar for longer evaluations
    st.write(f"Completed ratings: **{completed_count} of {total_count}**")
    st.progress(completed_count / total_count if total_count else 0.0)

    # Creating separate spaces for downloading, calculating, and showing a short instruction
    action_columns = st.columns([1, 1, 2])

    # Allowing ratings to be saved at any time, including before the table is complete
    action_columns[0].download_button(
        "Download evaluation CSV",
        data=judgments.to_csv(index=False).encode("utf-8"),
        file_name="evaluation_candidates.csv",
        mime="text/csv",
        width="stretch",
    )

    # Disabling metric calculation until every row contains a valid human rating
    calculate_metrics = action_columns[1].button(
        "Calculate metrics",
        type="primary",
        disabled=total_count == 0 or completed_count != total_count,
        width="stretch",
    )

    # Explaining why the calculation button may currently be disabled
    action_columns[2].caption("Completing every rating enables the metric calculation.")

    # Calculating metrics only after the evaluator intentionally clicks the enabled button
    if calculate_metrics:
        try:
            # Saving calculated metrics so they remain visible during later Streamlit reruns
            st.session_state["evaluation_metrics"] = evaluate_judgments(judgments)
        except (KeyError, ValueError) as error:
            # Showing a clear validation message instead of ending the entire application
            st.error(str(error))

    # Showing metric cards only after a successful complete evaluation calculation
    if "evaluation_metrics" in st.session_state:
        metrics = st.session_state["evaluation_metrics"]

        # Dividing the results into three equal cards for the required academic metrics
        metric_columns = st.columns(3)
        metric_columns[0].metric("Precision@5", f"{metrics['precision_at_5']:.3f}")
        metric_columns[1].metric("NDCG@10", f"{metrics['ndcg_at_10']:.3f}")
        metric_columns[2].metric("MRR", f"{metrics['mean_reciprocal_rank']:.3f}")

        # Explaining each metric in plain language for using the results in the viva
        st.markdown(
            """
            - **Interpreting Precision@5** as the share of the first five jobs rated 1 or 2.
            - **Interpreting NDCG@10** as how well highly relevant jobs appear near the top.
            - **Interpreting MRR** as how early the first job rated 1 or 2 appears.
            """
        )


# Looking for the original capstone job and resume datasets before building the page
local_job_path = find_local_dataset(("Jobs", "postings.csv"))
local_resume_path = find_local_dataset(("Resume", "Resume", "Resume.csv"))


# Showing the project name and a short honest explanation at the top of the browser
st.title("AI-Based Job Recommendation System")
st.write(
    "Providing privacy-first matching by retrieving recent jobs from Adzuna and USAJOBS, then "
    "ranking them against a resume using TF-IDF and cosine similarity in a local browser application."
)

# Stating the privacy boundary so users know the resume is not being sent to job APIs
st.caption(
    "Processing resume text locally and never sending it to a job API. Supporting LinkedIn and Indeed only through local tables."
)

# Creating three simple tabs for searching, evaluating, and reading instructions
match_tab, evaluation_tab, how_to_use_tab = st.tabs(["Find jobs", "Evaluation", "How to use"])


with st.sidebar:
    # Keeping search controls in the sidebar so they stay available on every tab
    st.header("Search settings")

    # Collecting the job title and location text sent to the selected live API
    keyword = st.text_input("Target job", value="Data Analyst")
    location = st.text_input("Location", value="New York")

    # Selecting the Adzuna country because its API uses a separate country endpoint
    adzuna_country_name = st.selectbox("Adzuna country", list(ADZUNA_COUNTRIES))

    # Limiting the age and number of jobs used during one recommendation run
    maximum_age_days = st.slider("Maximum posting age", 1, 60, 30)
    top_n = st.slider("Recommendations", 5, 30, 10, 5)
    result_limit = st.slider("Jobs per live source", 10, 100, 50, 10)

    # Allowing optional stricter matching when broad API results are not focused enough
    strict_filters = st.checkbox("Applying title/location contains filters", value=False)
    remote_only = st.checkbox("Remote jobs only", value=False)

    # Separating controls from short method and cache reminders
    st.divider()
    st.caption("Ranking: TF-IDF unigrams/bigrams + cosine similarity")
    st.caption("Caching: job results only · Local SQLite · One hour")


# Building the instruction tab first so API credential variables exist before job searching
with how_to_use_tab:
    # Explaining the complete application workflow as short numbered actions
    st.subheader("Running a recommendation")
    st.markdown(
        """
        1. **Adding** at least one live API below, or preparing a local job table.
        2. **Opening** **Find jobs** and selecting Adzuna, USAJOBS, or local job data.
        3. **Uploading** a PDF, DOCX, or TXT resume.
        4. **Entering** the target job, location, and filters in the sidebar.
        5. **Clicking** **Retrieve and rank jobs**.
        6. **Reviewing** matched terms, missing skills, posting dates, and original links.
        7. **Opening** **Evaluation** and assigning 0, 1, or 2 to every recommended job.
        8. **Calculating** Precision@5, NDCG@10, and MRR from the completed human ratings.
        9. **Downloading** recommendation, run-details, or evaluation files when required.
        """
    )

    # Providing official registration steps for the two supported live job sources
    st.subheader("Adding Adzuna and USAJOBS APIs")
    st.markdown(
        """
        - **Registering** for an Adzuna application ID and key at [developer.adzuna.com](https://developer.adzuna.com/).
        - **Requesting** a USAJOBS authorization key at [developer.usajobs.gov](https://developer.usajobs.gov/apirequest/).
        - **Using** the same email address supplied while requesting the USAJOBS key.
        - **Entering** the credentials below for the current session, or adding them to a private local secrets file.
        """
    )

    # Dividing credential inputs into two columns so the form stays compact
    credential_columns = st.columns(2)

    # Reading saved secret values when available while masking Adzuna credentials
    adzuna_id = credential_columns[0].text_input(
        "Adzuna application ID",
        value=configured_value("ADZUNA_APP_ID"),
        type="password",
    )
    adzuna_key = credential_columns[1].text_input(
        "Adzuna application key",
        value=configured_value("ADZUNA_APP_KEY"),
        type="password",
    )

    # Creating a second row for the email and key required by USAJOBS
    usa_columns = st.columns(2)

    # Keeping the USAJOBS email visible because it is an account identifier, not the API key
    usajobs_email = usa_columns[0].text_input(
        "USAJOBS registration email",
        value=configured_value("USAJOBS_EMAIL"),
    )

    # Masking the USAJOBS authorization key so it is not readable over the user's shoulder
    usajobs_key = usa_columns[1].text_input(
        "USAJOBS authorization key",
        value=configured_value("USAJOBS_API_KEY"),
        type="password",
    )

    # Explaining that typed secrets stay out of job caches and downloadable result files
    st.caption("Keeping values entered here in the running Streamlit session and excluding them from cache and exports.")

    # Hiding permanent setup details inside an optional expander to keep the page simple
    with st.expander("Adding credentials permanently on this computer"):
        # Explaining how Streamlit reads a private local secrets.toml file
        st.markdown(
            """
            1. **Creating** a folder named `.streamlit` inside the project folder.
            2. **Copying** `secrets.toml.example` into that folder.
            3. **Renaming** the copied file to `secrets.toml`.
            4. **Replacing** the blank values with the real credentials.
            5. **Restarting** `start_app.bat` after saving the file.
            """
        )

        # Showing an exact secrets-file example without including any real credential
        st.code(
            'ADZUNA_APP_ID = "your_id"\n'
            'ADZUNA_APP_KEY = "your_key"\n'
            'USAJOBS_EMAIL = "your_registration_email"\n'
            'USAJOBS_API_KEY = "your_key"',
            language="toml",
        )

        # Warning against sharing the file because .gitignore is not a complete security system
        st.warning("Keeping `.streamlit/secrets.toml` private and never submitting, emailing, or uploading it.")

    # Listing the main privacy choices in language suitable for the project report
    st.subheader("Protecting privacy")
    st.markdown(
        """
        - **Processing** PDF, DOCX, and TXT resume content inside the local Python process.
        - **Sending** only the job title, location, country, age, and remote filters to the selected job API.
        - **Sending** API credentials only to their official HTTPS endpoints.
        - **Caching** normalized public job results locally for one hour without saving credentials or resume text.
        - **Keeping** uploaded resumes and job tables in memory rather than copying them into the project.
        - **Avoiding** accounts, analytics, tracking, LinkedIn scraping, and Indeed scraping.
        """
    )

    # Explaining the manual human-rating workflow used for academic evaluation
    st.subheader("Completing the evaluation")
    st.markdown(
        """
        1. **Running** a recommendation for one resume on **Find jobs**.
        2. **Opening** **Evaluation** after ranked results appear.
        3. **Reading** each job title, company, location, and description without using the hidden model score.
        4. **Assigning** `0` for irrelevant, `1` for partly relevant, or `2` for highly relevant.
        5. **Adding** an optional short reason when a rating needs explanation.
        6. **Downloading** `evaluation_candidates.csv` to preserve the ratings outside the browser session.
        7. **Calculating** the metrics only after completing every row.
        """
    )

    # Reminding the user that one resume is only a demonstration, not strong general evidence
    st.caption("Using several resumes and independent evaluators produces stronger evidence than evaluating one resume alone.")


# Starting with an empty list of live sources that have complete credential pairs
configured_sources: list[str] = []

# Adding Adzuna only when both its application ID and application key are present
if adzuna_id.strip() and adzuna_key.strip():
    configured_sources.append("Adzuna")

# Adding USAJOBS only when both its registration email and authorization key are present
if usajobs_email.strip() and usajobs_key.strip():
    configured_sources.append("USAJOBS")


with match_tab:
    # Starting the first step of the recommendation workflow with job-source choices
    st.subheader("1. Selecting job sources")

    # Allowing one or both official APIs while preselecting only fully configured sources
    selected_sources = st.multiselect(
        "Official live sources",
        ["Adzuna", "USAJOBS"],
        default=configured_sources,
        help="Adding missing credentials on the How to use page before selecting a live source.",
    )

    # Guiding the user to API setup when no live source has complete credentials
    if not configured_sources:
        st.info("Adding API credentials on the How to use page enables live retrieval. Local job tables can be used without an API.")

    # Accepting optional research datasets without scraping or contacting their original websites
    uploaded_job_files = st.file_uploader(
        "Optional LinkedIn, Indeed, or other job tables",
        type=SUPPORTED_TABLE_TYPES,
        accept_multiple_files=True,
    )

    # Starting with automatic local dataset use disabled until a real file is detected
    use_local_jobs = False

    # Offering the detected Jobs/postings.csv file as an optional recommendation source
    if local_job_path is not None:
        use_local_jobs = st.checkbox(f"Including detected local dataset: {local_job_path}", value=True)
    else:
        # Explaining that live APIs or manual upload still work without the conventional file
        st.caption("No local Jobs/postings.csv was detected. Uploading a table remains optional when using a live API.")

    # Starting the second step by choosing between a personal file and research dataset row
    st.subheader("2. Selecting a resume")
    resume_mode = st.radio(
        "Resume source",
        ["Upload my resume", "Use a resume dataset row"],
        horizontal=True,
    )

    # Creating default values so later processing can safely check both resume modes
    uploaded_resume = None
    uploaded_resume_table = None
    resume_index = 0

    # Showing the personal resume uploader only when that mode is selected
    if resume_mode == "Upload my resume":
        # Restricting uploads to PDF, DOCX, and TXT because local readers exist for them
        uploaded_resume = st.file_uploader("Upload PDF, DOCX, or TXT", type=SUPPORTED_RESUME_TYPES)
    else:
        # Allowing the original resume corpus to support repeatable capstone demonstrations
        uploaded_resume_table = st.file_uploader("Upload resume CSV or Excel", type=SUPPORTED_TABLE_TYPES)

        # Selecting the cleaned dataset row used as the model query for this run
        resume_index = int(st.number_input("Cleaned resume row index", min_value=0, value=0, step=1))

        # Reporting the automatically detected resume file when no manual table is uploaded
        if uploaded_resume_table is None and local_resume_path is not None:
            st.info(f"Using detected resume dataset: {local_resume_path}")

    # Waiting for one deliberate click before calling APIs or fitting the TF-IDF model
    run_button = st.button("Retrieve and rank jobs", type="primary", width="stretch")

    if run_button:
        # Removing earlier run data so old results cannot be confused with the new search
        for state_key in [
            "recommendations",
            "run_summary",
            "source_results",
            "source_errors",
            "evaluation_judgments",
            "evaluation_metrics",
        ]:
            st.session_state.pop(state_key, None)

        # Showing a still information message instead of Streamlit's animated running figure
        processing_notice = st.info("Retrieving jobs, fitting TF-IDF, and calculating matches...")
        try:
            # Grouping processing inside one container without adding another visible page section
            with st.container():
                # Collecting successful sources, readable errors, and job DataFrames separately
                source_results: list[SourceResult] = []
                source_errors: list[str] = []
                frames: list[pd.DataFrame] = []

                # Opening the one-hour local public-job cache used by both API adapters
                cache = ResponseCache(CACHE_PATH)

                # Looping through selected APIs so one failed source does not stop the other
                for source in selected_sources:
                    try:
                        # Calling Adzuna with its required country code and credential pair
                        if source == "Adzuna":
                            result = fetch_adzuna(
                                keyword,
                                location,
                                ADZUNA_COUNTRIES[adzuna_country_name],
                                adzuna_id,
                                adzuna_key,
                                maximum_age_days,
                                result_limit,
                                cache,
                            )
                        else:
                            # Calling USAJOBS with its required email, key, and remote setting
                            result = fetch_usajobs(
                                keyword,
                                location,
                                usajobs_email,
                                usajobs_key,
                                maximum_age_days,
                                result_limit,
                                remote_only,
                                cache,
                            )

                        # Keeping successful source metadata for freshness reporting
                        source_results.append(result)

                        # Keeping normalized job rows for later combination and duplicate removal
                        frames.append(result.jobs)
                    except (JobSourceError, ValueError) as error:
                        # Recording one readable source error while allowing other sources to continue
                        source_errors.append(f"{source}: {error}")

                # Looping through every optional local job table uploaded in the browser
                for uploaded_file in uploaded_job_files or []:
                    # Reading at most thirty thousand rows before applying reproducible sampling
                    raw = read_uploaded_table(uploaded_file.name, uploaded_file.getvalue(), 30000)

                    # Sampling at most ten thousand rows with seed 42 for consistent demonstrations
                    raw = sample_rows(raw, 10000, 42)

                    # Converting different table column names into the common job-source schema
                    standardized = standardize_local_jobs(raw, infer_source_from_filename(uploaded_file.name))
                    frames.append(standardized)

                    # Creating a SourceResult record so local data appears in the freshness table
                    source_results.append(
                        SourceResult(
                            infer_source_from_filename(uploaded_file.name),
                            standardized,
                            datetime.now(timezone.utc).isoformat(),
                            False,
                            "Loading an uploaded table",
                        )
                    )

                # Loading the detected research dataset only when its checkbox remains selected
                if use_local_jobs and local_job_path is not None:
                    # Reading and reproducibly sampling the potentially large local job dataset
                    raw = load_table(local_job_path, row_limit=30000)
                    raw = sample_rows(raw, 10000, 42)

                    # Standardizing local research columns without accessing the LinkedIn website
                    standardized = standardize_local_jobs(raw, "LinkedIn dataset")
                    frames.append(standardized)

                    # Recording the local dataset as a source for transparent run reporting
                    source_results.append(
                        SourceResult(
                            "LinkedIn dataset",
                            standardized,
                            datetime.now(timezone.utc).isoformat(),
                            False,
                            "Loading the detected local table",
                        )
                    )

                # Combining all source frames and removing duplicate job listings in one step
                combined_jobs = combine_job_frames(frames)

                # Applying the selected maximum age only to live Adzuna and USAJOBS records
                combined_jobs = filter_recent_jobs(combined_jobs, maximum_age_days)

                # Applying an extra local remote-text check when remote-only mode is selected
                if remote_only:
                    # Joining title, location, and description because sources label remote jobs differently
                    remote_text = (
                        combined_jobs["title"].astype(str)
                        + " "
                        + combined_jobs["location"].astype(str)
                        + " "
                        + combined_jobs["description"].astype(str)
                    )

                    # Keeping only rows containing the word remote in the combined searchable text
                    combined_jobs = combined_jobs[remote_text.str.contains("remote", case=False, na=False)]

                # Stopping with a useful message when filters remove every available job
                if combined_jobs.empty:
                    raise ValueError("No jobs remained after applying recency and remote filters.")

                # Cleaning and validating the final job table before fitting TF-IDF
                job_data = prepare_job_data(combined_jobs)

                # Starting with no personal resume text until upload mode supplies a valid file
                external_resume_text: Optional[str] = None

                # Reading a personal resume only when personal-upload mode is selected
                if resume_mode == "Upload my resume":
                    # Requiring an uploaded file before starting private local parsing
                    if uploaded_resume is None:
                        raise ValueError("Uploading a PDF, DOCX, or TXT resume is required.")

                    # Extracting resume text locally without sending it to Adzuna or USAJOBS
                    external_resume_text = read_uploaded_resume(uploaded_resume.name, uploaded_resume.getvalue())

                # Giving a manually uploaded resume table priority over an automatically detected table
                if uploaded_resume_table is not None:
                    # Reading, sampling, cleaning, and validating the uploaded resume dataset
                    resume_raw = read_uploaded_table(uploaded_resume_table.name, uploaded_resume_table.getvalue())
                    resume_data = prepare_resume_data(sample_rows(resume_raw, 500, 42))
                elif local_resume_path is not None:
                    # Falling back to the detected capstone resume dataset for dataset-row mode
                    resume_data = prepare_resume_data(sample_rows(load_table(local_resume_path), 500, 42))
                elif external_resume_text is not None:
                    # Building a one-row model corpus when using a personal uploaded resume
                    resume_data = build_single_resume_corpus(external_resume_text)
                else:
                    # Stopping when neither resume mode supplies data that the model can use
                    raise ValueError("Uploading a resume dataset or personal resume is required.")

                # Counting resumes and jobs to choose safe TF-IDF document-frequency limits
                document_count = len(resume_data) + len(job_data)

                # Keeping normal thresholds for real data while relaxing them for tiny test inputs
                settings = ModelSettings(
                    max_resumes=500,
                    max_jobs=len(job_data),
                    top_n=top_n,
                    min_df=2 if document_count >= 5 else 1,
                    max_df=0.95 if document_count >= 5 else 1.0,
                )

                # Creating the recommendation model with the recorded settings for this run
                model = JobRecommendationModel(settings)

                # Fitting one shared vocabulary so resumes and jobs use the same feature meanings
                model.fit(resume_data, job_data)

                # Transforming either the selected dataset row or personal resume into a query vector
                query_vector, query_label = model.create_query_vector(
                    None if resume_mode == "Upload my resume" else resume_index,
                    external_resume_text,
                )

                # Ranking eligible jobs by cosine similarity and collecting explanation fields
                recommendations, eligible_scores, explanation = model.recommend(
                    query_vector,
                    top_n,
                    location_filter=location.strip() if strict_filters else None,
                    title_filter=keyword.strip() if strict_filters else None,
                )

                # Recording non-sensitive search choices used to produce the recommendation list
                search_settings = {
                    "keyword": keyword.strip(),
                    "location": location.strip(),
                    "maximum_age_days": maximum_age_days,
                    "remote_only": remote_only,
                    "strict_filters": strict_filters,
                    "resume_query": query_label,
                }

                # Combining model measurements, settings, and source information into one summary
                summary = create_summary(
                    settings,
                    model,
                    recommendations,
                    eligible_scores,
                    search_settings,
                    source_results,
                )

                # Exporting reproducible results without writing credentials or resume text
                export_results(
                    recommendations,
                    explanation,
                    eligible_scores,
                    OUTPUT_DIRECTORY,
                    summary,
                    create_plot=True,
                )

                # Saving recommendations in browser session state for surviving Streamlit reruns
                st.session_state["recommendations"] = recommendations
                st.session_state["run_summary"] = summary
                st.session_state["source_results"] = source_results
                st.session_state["source_errors"] = source_errors

                # Creating blank human-rating rows from exactly the displayed recommendations
                replace_evaluation_table(
                    build_judgment_template(
                        recommendations,
                        query_id="resume_1",
                        query_label=query_label,
                    )
                )

            # Removing the temporary processing notice after the complete run succeeds
            processing_notice.empty()

            # Confirming completion without using an animated Streamlit status element
            st.success("Retrieving and ranking completed.")
        except (FileNotFoundError, KeyError, ValueError, IndexError, ImportError, JobSourceError) as error:
            # Removing the processing notice before displaying the final readable error message
            processing_notice.empty()

            # Showing expected file, data, model, import, or API failures inside the app
            st.error(str(error))

    # Showing individual source warnings saved from the latest recommendation run
    if "source_errors" in st.session_state:
        for source_error in st.session_state["source_errors"]:
            st.warning(source_error)

    # Showing source counts and freshness only when at least one source succeeded
    if "source_results" in st.session_state and st.session_state["source_results"]:
        st.dataframe(source_report_table(st.session_state["source_results"]), hide_index=True, width="stretch")

    # Showing ranked recommendation cards only when both results and summary are available
    if "recommendations" in st.session_state and "run_summary" in st.session_state:
        render_results(st.session_state["recommendations"], st.session_state["run_summary"])


# Rendering the evaluation tab after defining the complete human-rating function
with evaluation_tab:
    render_evaluation_page()
