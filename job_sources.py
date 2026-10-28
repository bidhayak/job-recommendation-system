"""Retrieving and standardizing live jobs from official public endpoints.

Calling Adzuna and USAJOBS through bounded HTTPS requests, mapping different
response structures into one shared schema, caching only public job records,
and removing repeated listings keeps source handling separate from resume text
and recommendation-model calculations.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


# Defining one shared schema so every API produces the same columns
STANDARD_JOB_COLUMNS = [
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


class JobSourceError(RuntimeError):
    """Reporting a readable live-source failure without exposing credentials."""


@dataclass(frozen=True)
class SourceResult:
    """Storing standardized jobs and retrieval metadata for one source."""

    source: str
    jobs: pd.DataFrame
    fetched_at: str
    from_cache: bool
    note: str


def _clean_html(value: object) -> str:
    """Converting HTML descriptions into plain readable text."""

    # Decoding nested HTML entities before removing markup tags
    text = html.unescape(html.unescape(str(value or "")))
    # Preserving useful spacing where paragraphs, list items, or line breaks end
    text = re.sub(r"<br\s*/?>|</p>|</li>", "\n", text, flags=re.IGNORECASE)
    # Removing remaining HTML elements and collapsing repeated whitespace
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _safe_identifier(value: str, label: str) -> str:
    """Validating a country identifier before forming an endpoint URL."""

    # Removing surrounding spaces before checking the complete identifier
    normalized = value.strip()
    if not normalized or not re.fullmatch(r"[A-Za-z0-9_-]+", normalized):
        raise ValueError(f"{label} must contain only letters, numbers, hyphens, or underscores.")
    return normalized


def _request_json(
    url: str,
    parameters: Optional[dict[str, object]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout_seconds: int = 25,
) -> object:
    """Requesting one HTTPS JSON response with bounded waiting and clear errors."""

    # Rejecting non-HTTPS endpoints before adding any credential-bearing parameters
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS API endpoints are allowed.")
    # Excluding blank optional values while safely URL-encoding the remaining query
    query = urlencode(
        {key: value for key, value in (parameters or {}).items() if value not in (None, "")},
        doseq=True,
    )
    request_url = f"{url}?{query}" if query else url
    # Providing explicit JSON and user-agent headers while allowing API-specific headers
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "Bidhayak-Capstone-Job-Recommender/1.0",
        **(headers or {}),
    }
    # Creating a read-only request and bounding the network waiting time
    request = Request(request_url, headers=request_headers, method="GET")
    try:
        # Decoding the response directly into Python dictionaries and lists
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        if error.code in {401, 403}:
            message = "Authentication was rejected. Checking the API ID, key, and email is required."
        elif error.code == 404:
            message = "The requested company board or API endpoint was not found."
        elif error.code == 429:
            message = "The API rate limit was reached. Waiting and trying again is required."
        else:
            message = f"The API returned HTTP {error.code}."
        raise JobSourceError(message) from error
    except (URLError, TimeoutError) as error:
        raise JobSourceError("The live source could not be reached. Checking the internet connection is required.") from error
    except json.JSONDecodeError as error:
        raise JobSourceError("The live source returned an unreadable response.") from error


class ResponseCache:
    """Caching normalized API records in a small local SQLite database."""

    def __init__(self, database_path: Path) -> None:
        """Creating the cache table when it is not already present."""

        # Keeping the database beside the application instead of using cloud storage
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        # Creating one compact key-value table during the first application run
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    stored_at REAL NOT NULL
                )
                """
            )

    def load(self, cache_key: str, maximum_age_minutes: int) -> Optional[tuple[list[dict], float]]:
        """Loading a fresh cached payload when its age remains acceptable."""

        # Reading one matching entry with a parameterized query
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            row = connection.execute(
                "SELECT payload_json, stored_at FROM api_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        # Rejecting missing or expired values so live retrieval can replace them
        if row is None or (time.time() - float(row[1])) > maximum_age_minutes * 60:
            return None
        return json.loads(str(row[0])), float(row[1])

    def save(self, cache_key: str, records: list[dict]) -> float:
        """Saving one normalized response without storing API credentials."""

        # Recording the save time for applying the automatic one-hour expiry
        stored_at = time.time()
        # Replacing an existing search result without storing API keys or resume text
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute(
                """
                INSERT INTO api_cache(cache_key, payload_json, stored_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    stored_at = excluded.stored_at
                """,
                (cache_key, json.dumps(records), stored_at),
            )
        return stored_at


def _cache_key(source: str, public_parameters: dict[str, object]) -> str:
    """Hashing non-secret search settings into a stable cache key."""

    # Sorting keys before hashing so the same search always produces the same key
    serialized = json.dumps(public_parameters, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{source}:{digest}"


def _timestamp_text(epoch_seconds: float) -> str:
    """Formatting cache and retrieval times as UTC text."""

    return datetime.fromtimestamp(epoch_seconds, timezone.utc).isoformat()


def _using_cache(
    source: str,
    public_parameters: dict[str, object],
    loader: Callable[[], list[dict]],
    cache: Optional[ResponseCache],
    cache_minutes: int,
) -> SourceResult:
    """Loading fresh cached records or calling an endpoint after cache expiry."""

    # Building the cache key only from non-secret search settings
    key = _cache_key(source, public_parameters)
    if cache is not None:
        cached = cache.load(key, cache_minutes)
        if cached is not None:
            records, stored_at = cached
            return SourceResult(
                source=source,
                jobs=pd.DataFrame(records, columns=STANDARD_JOB_COLUMNS),
                fetched_at=_timestamp_text(stored_at),
                from_cache=True,
                note=f"Loading {len(records)} cached job(s)",
            )

    # Calling the official endpoint only when no reusable local result exists
    records = loader()
    stored_at = cache.save(key, records) if cache is not None else time.time()
    return SourceResult(
        source=source,
        jobs=pd.DataFrame(records, columns=STANDARD_JOB_COLUMNS),
        fetched_at=_timestamp_text(stored_at),
        from_cache=False,
        note=f"Retrieving {len(records)} live job(s)",
    )


def fetch_adzuna(
    keyword: str,
    location: str,
    country: str,
    application_id: str,
    application_key: str,
    maximum_age_days: int,
    result_limit: int,
    cache: Optional[ResponseCache] = None,
    cache_minutes: int = 60,
    requester: Callable[..., object] = _request_json,
) -> SourceResult:
    """Retrieving current general-market jobs from the Adzuna search API."""

    # Requiring credentials before constructing a live Adzuna request
    if not application_id.strip() or not application_key.strip():
        raise ValueError("Adzuna requires both an application ID and application key.")
    country_code = _safe_identifier(country.lower(), "Adzuna country code")
    # Excluding credentials from values used for identifying cached searches
    public_parameters = {
        "keyword": keyword.strip(),
        "location": location.strip(),
        "country": country_code,
        "maximum_age_days": maximum_age_days,
        "result_limit": result_limit,
    }

    def loading() -> list[dict]:
        """Loading and normalizing one Adzuna search response."""

        # Preparing a standardized record list across one or more API pages
        records: list[dict] = []
        # Respecting Adzuna's practical page size while supporting the requested limit
        page_size = min(result_limit, 50)
        page_count = max(1, (result_limit + page_size - 1) // page_size)
        for page_number in range(1, page_count + 1):
            # Building the country-specific HTTPS endpoint for the current page
            endpoint = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/{page_number}"
            payload = requester(
                endpoint,
                {
                    "app_id": application_id.strip(),
                    "app_key": application_key.strip(),
                    "results_per_page": page_size,
                    "what": keyword.strip(),
                    "where": location.strip(),
                    "max_days_old": maximum_age_days,
                    "sort_by": "date",
                    "content-type": "application/json",
                },
            )
            # Reading only the documented results array from a valid object response
            results = payload.get("results", []) if isinstance(payload, dict) else []
            for item in results:
                # Expanding optional nested company and location objects safely
                company = item.get("company") or {}
                job_location = item.get("location") or {}
                # Mapping Adzuna fields into the shared job schema
                records.append(
                    {
                        "job_id": str(item.get("id", "")),
                        "title": _clean_html(item.get("title", "")),
                        "description": _clean_html(item.get("description", "")),
                        "skills_desc": "",
                        "location": _clean_html(job_location.get("display_name", "")),
                        "company_name": _clean_html(company.get("display_name", "Not provided")),
                        "job_posting_url": str(item.get("redirect_url", "")),
                        "source": "Adzuna",
                        "posted_at": str(item.get("created", "")),
                        "expires_at": "",
                    }
                )
                # Stopping immediately after collecting the user-requested job count
                if len(records) >= result_limit:
                    return records
            # Stopping pagination when Adzuna returns a partial final page
            if len(results) < page_size:
                break
        return records

    return _using_cache(
        "Adzuna",
        public_parameters,
        loading,
        cache,
        cache_minutes,
    )


def _joining_text(value: object) -> str:
    """Joining nested API text fields into one readable description."""

    # Recursing through lists and dictionaries because USAJOBS returns nested sections
    if isinstance(value, list):
        return " ".join(_joining_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_joining_text(item) for item in value.values())
    return _clean_html(value)


def fetch_usajobs(
    keyword: str,
    location: str,
    user_email: str,
    authorization_key: str,
    maximum_age_days: int,
    result_limit: int,
    remote_only: bool = False,
    cache: Optional[ResponseCache] = None,
    cache_minutes: int = 60,
    requester: Callable[..., object] = _request_json,
) -> SourceResult:
    """Retrieving currently open United States federal jobs from USAJOBS."""

    # Requiring the registered email and key used by USAJOBS request headers
    if not user_email.strip() or not authorization_key.strip():
        raise ValueError("USAJOBS requires the registration email and authorization key.")
    # Excluding authentication values from the local cache identity
    public_parameters = {
        "keyword": keyword.strip(),
        "location": location.strip(),
        "maximum_age_days": maximum_age_days,
        "result_limit": result_limit,
        "remote_only": remote_only,
    }

    def loading() -> list[dict]:
        """Loading and normalizing one USAJOBS search response."""

        # Sending search settings as parameters and credentials as required headers
        payload = requester(
            "https://data.usajobs.gov/api/search",
            {
                "Keyword": keyword.strip(),
                "LocationName": location.strip(),
                "DatePosted": max(0, min(maximum_age_days, 60)),
                "ResultsPerPage": min(result_limit, 500),
                "Page": 1,
                "Fields": "Full",
                "RemoteIndicator": "True" if remote_only else "",
            },
            {
                "Host": "data.usajobs.gov",
                "User-Agent": user_email.strip(),
                "Authorization-Key": authorization_key.strip(),
            },
        )
        # Unwrapping the nested USAJOBS search response defensively
        search_result = payload.get("SearchResult", {}) if isinstance(payload, dict) else {}
        items = search_result.get("SearchResultItems", [])
        records: list[dict] = []
        for item in items[:result_limit]:
            # Extracting the descriptor and detailed text sections for each announcement
            descriptor = item.get("MatchedObjectDescriptor") or {}
            details = (descriptor.get("UserArea") or {}).get("Details") or {}
            # Joining qualification, duty, requirement, education, and evaluation text
            descriptions = [
                descriptor.get("QualificationSummary", ""),
                details.get("JobSummary", ""),
                details.get("MajorDuties", ""),
                details.get("Requirements", ""),
                details.get("Education", ""),
                details.get("Evaluations", ""),
            ]
            # Preferring an application URL while retaining the public announcement URL
            apply_uris = descriptor.get("ApplyURI") or []
            # Mapping USAJOBS fields into the same schema used by Adzuna and local files
            records.append(
                {
                    "job_id": str(descriptor.get("PositionID") or item.get("MatchedObjectId", "")),
                    "title": _clean_html(descriptor.get("PositionTitle", "")),
                    "description": _joining_text(descriptions),
                    "skills_desc": _joining_text(details.get("KeyRequirements", "")),
                    "location": _clean_html(descriptor.get("PositionLocationDisplay", "")),
                    "company_name": _clean_html(descriptor.get("OrganizationName", "United States Government")),
                    "job_posting_url": str(
                        (apply_uris[0] if apply_uris else "") or descriptor.get("PositionURI", "")
                    ),
                    "source": "USAJOBS",
                    "posted_at": str(descriptor.get("PublicationStartDate", "")),
                    "expires_at": str(
                        descriptor.get("ApplicationCloseDate") or descriptor.get("PositionEndDate", "")
                    ),
                }
            )
        return records

    return _using_cache(
        "USAJOBS",
        public_parameters,
        loading,
        cache,
        cache_minutes,
    )


def combine_job_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combining sources while removing exact URLs and near-identical listings."""

    # Ignoring unavailable sources while copying valid frames before modification
    nonempty = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        raise ValueError("No jobs were retrieved. Checking sources, credentials, and search terms is required.")
    # Stacking all source rows before normalizing missing values and duplicates
    combined = pd.concat(nonempty, ignore_index=True, sort=False)

    # Adding missing standard fields before forming duplicate keys
    for column in STANDARD_JOB_COLUMNS:
        if column not in combined.columns:
            combined[column] = ""
    combined = combined.fillna("")
    # Removing exact repeated URLs first because they identify the same public listing
    url_key = combined["job_posting_url"].astype(str).str.strip().str.lower()
    has_url = url_key.ne("")
    with_url = combined[has_url].assign(_url_key=url_key[has_url]).drop_duplicates("_url_key")
    without_url = combined[~has_url]
    combined = pd.concat([with_url.drop(columns="_url_key"), without_url], ignore_index=True)
    # Building a fallback key for listings that have missing or changed URLs
    combined["_text_key"] = (
        combined["title"].astype(str).str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
        + "|"
        + combined["company_name"].astype(str).str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
        + "|"
        + combined["location"].astype(str).str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    )
    combined = combined.drop_duplicates("_text_key").drop(columns="_text_key")
    return combined.reset_index(drop=True)
