from __future__ import annotations

import argparse
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from referee_matcher import (
    collect_text,
    conflict_reasons,
    discover_documents,
    load_conflict_rules,
)
from candidates_loader import load_candidates_table


OPENALEX_AUTHORS_URL = "https://api.openalex.org/authors"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_AUTHOR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/author/search"
SEMANTIC_SCHOLAR_AUTHOR_PAPERS_URL_TMPL = "https://api.semanticscholar.org/graph/v1/author/{author_id}/papers"


ITALIAN_STOPWORDS = {
    "a",
    "ad",
    "al",
    "alla",
    "allo",
    "ai",
    "agli",
    "all",
    "agl",
    "anche",
    "avere",
    "con",
    "come",
    "da",
    "dal",
    "dalla",
    "dallo",
    "dei",
    "del",
    "della",
    "dello",
    "di",
    "e",
    "ed",
    "gli",
    "ha",
    "hanno",
    "i",
    "il",
    "in",
    "io",
    "la",
    "le",
    "lo",
    "ma",
    "mi",
    "nel",
    "nella",
    "nello",
    "non",
    "o",
    "per",
    "piu",
    "quale",
    "quali",
    "quello",
    "quelli",
    "quella",
    "quelle",
    "se",
    "si",
    "sono",
    "su",
    "tra",
    "un",
    "una",
    "uno",
}


@dataclass
class OnlineCandidate:
    candidate_id: str
    name: str
    scholar_query: str
    institution: str
    department: str
    openalex_author_id: str
    pe_areas: set[str]


@dataclass
class OnlineCandidateResult:
    candidate: OnlineCandidate
    ml_score: float
    activity_score: float
    final_score: float
    documents_used: int
    works_total: int
    citations_total: int
    h_index: int | None
    is_conflict: bool
    conflict_reasons: str
    eligible: bool


@dataclass
class MinimalConflictProfile:
    candidate_id: str
    name: str
    institution: str
    department: str
    extracted_authors: set[str]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_pe_value(value: str) -> str:
    return normalize_text(value).upper().replace(" ", "")


def parse_pe_areas(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    parts = re.split(r"[;,|]", raw)
    return {normalize_pe_value(item) for item in parts if item.strip()}


def parse_sources(raw: str) -> list[str]:
    allowed = {"openalex", "semanticscholar"}
    parsed = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [item for item in parsed if item not in allowed]
    if unknown:
        raise ValueError(f"Unsupported sources: {unknown}. Allowed values: {sorted(allowed)}")
    if not parsed:
        raise ValueError("At least one source must be provided in --sources")
    return parsed


def infer_project_pe(project_text: str) -> tuple[list[str], dict[str, int]]:
    matches = re.findall(r"\bPE\s*[-_]?\s*(\d{1,2})\b", project_text, flags=re.IGNORECASE)
    counts: dict[str, int] = {}
    for code in matches:
        token = f"PE{int(code)}"
        counts[token] = counts.get(token, 0) + 1

    if not counts:
        return [], counts

    ordered = sorted(counts.items(), key=lambda item: (-item[1], int(item[0][2:])))
    return [item[0] for item in ordered], counts


def restore_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""

    max_pos = max((max(pos) for pos in inverted_index.values() if pos), default=-1)
    if max_pos < 0:
        return ""

    words = [""] * (max_pos + 1)
    for token, positions in inverted_index.items():
        for pos in positions:
            if 0 <= pos < len(words):
                words[pos] = token

    return normalize_text(" ".join(words))


def name_tokens(name: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z]+", name.lower()) if len(tok) >= 2}


def candidate_name_match_score(candidate_name: str, author_name: str) -> float:
    ct = name_tokens(candidate_name)
    at = name_tokens(author_name)
    if not ct or not at:
        return 0.0
    overlap = len(ct.intersection(at))
    return overlap / max(1, min(len(ct), len(at)))


def load_project_text(project_dir: Path) -> tuple[str, int]:
    docs = discover_documents(project_dir)
    return collect_text(docs)


def load_online_candidates(candidates_csv: Path, project_pes: set[str]) -> list[OnlineCandidate]:
    df = load_candidates_table(candidates_csv, pe_column="pe_areas")

    out: list[OnlineCandidate] = []
    for _, row in df.iterrows():
        pe_areas = parse_pe_areas(str(row.get("pe_areas", "")))
        # Hard pre-filter: only candidates with matching PE are considered for any API call.
        if not pe_areas.intersection(project_pes):
            continue

        candidate_id = str(row["candidate_id"]).strip()
        name = str(row["name"]).strip()
        scholar_query = str(row.get("scholar_query", "")).strip() or name
        institution = str(row.get("institution", "")).strip()
        department = str(row.get("department", "")).strip()
        openalex_author_id = str(row.get("openalex_author_id", "")).strip()
        out.append(
            OnlineCandidate(
                candidate_id=candidate_id,
                name=name,
                scholar_query=scholar_query,
                institution=institution,
                department=department,
                openalex_author_id=openalex_author_id,
                pe_areas=pe_areas,
            )
        )
    return out


def request_json(url: str, params: dict[str, Any], mailto: str, timeout: int = 45) -> dict[str, Any]:
    request_params = dict(params)
    if mailto.strip():
        request_params["mailto"] = mailto.strip()
    response = requests.get(url, params=request_params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def request_semantic_scholar_json(
    url: str,
    params: dict[str, Any],
    api_key: str,
    timeout: int = 45,
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    if api_key.strip():
        headers["x-api-key"] = api_key.strip()

    max_retries = 4
    backoff_seconds = 1.5
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()

        retry_after_raw = response.headers.get("Retry-After", "").strip()
        retry_after = float(retry_after_raw) if retry_after_raw.replace(".", "", 1).isdigit() else 0.0
        wait_seconds = retry_after if retry_after > 0 else backoff_seconds * (2**attempt)
        last_error = requests.HTTPError(
            f"Semantic Scholar rate-limited (429), attempt {attempt + 1}/{max_retries + 1}",
            response=response,
        )
        if attempt < max_retries:
            print(f"[WARN] Semantic Scholar 429, retry in {wait_seconds:.1f}s")
            time.sleep(wait_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Semantic Scholar request failed unexpectedly")


def resolve_openalex_author_id(candidate: OnlineCandidate, mailto: str) -> tuple[str, str, int | None]:
    if candidate.openalex_author_id:
        return candidate.openalex_author_id, candidate.institution, None

    params = {
        "search": candidate.scholar_query,
        "per-page": 15,
        "sort": "works_count:desc",
    }
    payload = request_json(OPENALEX_AUTHORS_URL, params=params, mailto=mailto)
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"No OpenAlex author found for candidate {candidate.name}")

    best = None
    best_score = -1.0
    for item in results:
        display_name = str(item.get("display_name", ""))
        score = candidate_name_match_score(candidate.name, display_name)
        if score > best_score:
            best = item
            best_score = score

    if best is None:
        raise ValueError(f"Cannot resolve OpenAlex author for candidate {candidate.name}")

    author_id = str(best.get("id", "")).strip()
    if not author_id:
        raise ValueError(f"Resolved author has no id for candidate {candidate.name}")

    resolved_institution = candidate.institution
    institutions = best.get("last_known_institutions") or []
    if not resolved_institution and institutions:
        display_name = institutions[0].get("display_name")
        if display_name:
            resolved_institution = str(display_name)

    h_index = None
    summary_stats = best.get("summary_stats") or {}
    if isinstance(summary_stats, dict):
        h_idx = summary_stats.get("h_index")
        if isinstance(h_idx, int):
            h_index = h_idx

    return author_id, resolved_institution, h_index


def resolve_semantic_scholar_author_id(candidate: OnlineCandidate, s2_api_key: str) -> tuple[str | None, int | None]:
    params = {
        "query": candidate.scholar_query,
        "limit": 20,
        "fields": "name,paperCount,hIndex",
    }
    payload = request_semantic_scholar_json(
        SEMANTIC_SCHOLAR_AUTHOR_SEARCH_URL,
        params=params,
        api_key=s2_api_key,
    )
    results = payload.get("data", [])
    if not results:
        return None, None

    best = None
    best_score = -1.0
    for item in results:
        display_name = str(item.get("name", ""))
        score = candidate_name_match_score(candidate.name, display_name)
        if score > best_score:
            best = item
            best_score = score

    if best is None:
        return None, None

    author_id = str(best.get("authorId", "") or "").strip()
    h_idx = best.get("hIndex")
    h_index = int(h_idx) if isinstance(h_idx, int) else None
    if not author_id:
        return None, h_index
    return author_id, h_index


def language_stopwords(language: str) -> str | list[str] | None:
    if language == "en":
        return "english"
    if language == "it":
        return list(ITALIAN_STOPWORDS)
    return None


def stream_openalex_works(
    author_id: str,
    candidate_name: str,
    vectorizer: HashingVectorizer,
    mailto: str,
    from_year: int | None,
    max_works: int | None,
    sleep_seconds: float,
) -> tuple[csr_matrix, int, int, set[str], int]:
    cursor = "*"
    per_page = 200

    aggregate_vec: csr_matrix | None = None
    works_total = 0
    citations_total = 0
    coauthors: set[str] = set()
    docs_used = 0

    while True:
        params: dict[str, Any] = {
            "filter": f"author.id:{author_id}",
            "per-page": per_page,
            "cursor": cursor,
        }
        if from_year is not None:
            params["filter"] = f"author.id:{author_id},from_publication_date:{from_year}-01-01"

        payload = request_json(OPENALEX_WORKS_URL, params=params, mailto=mailto)
        results = payload.get("results", [])
        if not results:
            break

        for work in results:
            title = normalize_text(str(work.get("title", "")))
            abstract = restore_abstract(work.get("abstract_inverted_index"))
            text = normalize_text(f"{title} {abstract}")

            if text:
                vec = vectorizer.transform([text])
                aggregate_vec = vec if aggregate_vec is None else (aggregate_vec + vec)
                docs_used += 1

            works_total += 1
            citations_total += int(work.get("cited_by_count", 0) or 0)

            for auth in work.get("authorships", []):
                display_name = str((auth.get("author") or {}).get("display_name", "")).strip()
                if not display_name:
                    continue
                if candidate_name_match_score(candidate_name, display_name) >= 0.8:
                    continue
                coauthors.add(display_name)

            if max_works is not None and works_total >= max_works:
                break

        if max_works is not None and works_total >= max_works:
            break

        meta = payload.get("meta") or {}
        next_cursor = str(meta.get("next_cursor", "") or "").strip()
        if not next_cursor or next_cursor == cursor:
            break

        cursor = next_cursor
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if aggregate_vec is None:
        aggregate_vec = vectorizer.transform([""])

    return aggregate_vec, works_total, citations_total, coauthors, docs_used


def stream_semantic_scholar_papers(
    author_id: str,
    candidate_name: str,
    vectorizer: HashingVectorizer,
    s2_api_key: str,
    from_year: int | None,
    max_works: int | None,
    sleep_seconds: float,
) -> tuple[csr_matrix, int, int, set[str], int]:
    limit = 100
    offset = 0

    aggregate_vec: csr_matrix | None = None
    works_total = 0
    citations_total = 0
    coauthors: set[str] = set()
    docs_used = 0

    while True:
        params: dict[str, Any] = {
            "offset": offset,
            "limit": limit,
            "fields": "title,abstract,year,citationCount,authors",
        }
        url = SEMANTIC_SCHOLAR_AUTHOR_PAPERS_URL_TMPL.format(author_id=author_id)
        payload = request_semantic_scholar_json(url, params=params, api_key=s2_api_key)
        data = payload.get("data", [])
        if not data:
            break

        for item in data:
            year = item.get("year")
            if from_year is not None and isinstance(year, int) and year < from_year:
                continue

            title = normalize_text(str(item.get("title", "")))
            abstract = normalize_text(str(item.get("abstract", "")))
            text = normalize_text(f"{title} {abstract}")
            if text:
                vec = vectorizer.transform([text])
                aggregate_vec = vec if aggregate_vec is None else (aggregate_vec + vec)
                docs_used += 1

            works_total += 1
            citations_total += int(item.get("citationCount", 0) or 0)

            for author in item.get("authors", []):
                display_name = str(author.get("name", "") or "").strip()
                if not display_name:
                    continue
                if candidate_name_match_score(candidate_name, display_name) >= 0.8:
                    continue
                coauthors.add(display_name)

            if max_works is not None and works_total >= max_works:
                break

        if max_works is not None and works_total >= max_works:
            break

        if len(data) < limit:
            break

        offset += limit
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if aggregate_vec is None:
        aggregate_vec = vectorizer.transform([""])

    return aggregate_vec, works_total, citations_total, coauthors, docs_used


def compute_activity_score(works_total: int, citations_total: int, h_index: int | None) -> float:
    works_part = math.log1p(max(0, works_total))
    citations_part = math.log1p(max(0, citations_total))
    h_part = math.log1p(max(0, h_index or 0))
    return 0.35 * works_part + 0.45 * citations_part + 0.20 * h_part


def write_markdown_report(output_path: Path, ranking: pd.DataFrame, project_doc_count: int) -> None:
    eligible = ranking[ranking["eligible"] == True]
    if eligible.empty:
        raise ValueError("No eligible candidate after conflict filtering")

    best = eligible.iloc[0]
    lines = [
        "# Online Referee Ranking (No Local Publications Download)",
        "",
        f"Project documents analyzed: **{project_doc_count}**",
        "",
        "## Recommended Referee",
        "",
        f"- ID: **{best['candidate_id']}**",
        f"- Name: **{best['name']}**",
        f"- Final score: **{best['final_score']:.4f}**",
        f"- ML compatibility score: **{best['ml_score']:.4f}**",
        f"- Works analyzed online: **{int(best['works_total'])}**",
        "",
        "## Full Ranking",
        "",
        ranking.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Online referee matching that analyzes all candidate activity via OpenAlex API in streaming mode, "
            "without saving publications locally."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--project-dir", type=Path, default=None)
    parser.add_argument("--candidates-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--conflicts-file", type=Path, default=None)
    parser.add_argument("--conflict-mode", type=str, default="exclude", choices=["exclude", "flag"])
    parser.add_argument("--language", type=str, default="en", choices=["en", "it", "none"])
    parser.add_argument("--from-year", type=int, default=None)
    parser.add_argument(
        "--max-works",
        type=int,
        default=100,
        help="Maximum works per source per candidate (default: 100). Ignored when --full-scan is enabled.",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Enable explicit full scan without work limit per candidate/source.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument(
        "--sources",
        type=str,
        default="openalex,semanticscholar",
        help="Comma-separated sources to use for matching: openalex, semanticscholar",
    )
    parser.add_argument(
        "--openalex-mailto",
        type=str,
        default="",
        help="Contact email to include in OpenAlex requests for better rate-limit handling",
    )
    parser.add_argument(
        "--activity-weight",
        type=float,
        default=0.20,
        help="Weight for global scientific activity score in final ranking",
    )
    parser.add_argument(
        "--pe-overlap-weight",
        type=float,
        default=0.03,
        help="Weight for PE overlap score in final ranking (kept much smaller than activity weight)",
    )
    parser.add_argument(
        "--s2-api-key",
        type=str,
        default="",
        help="Semantic Scholar API key (optional). If omitted, environment variable S2_API_KEY is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = parse_sources(args.sources)
    s2_api_key = args.s2_api_key.strip() or os.getenv("S2_API_KEY", "").strip()
    effective_max_works = None if args.full_scan else max(1, int(args.max_works))

    base_dir = args.base_dir.resolve()
    project_dir = (args.project_dir or (base_dir / "data" / "project")).resolve()
    candidates_csv = (args.candidates_csv or (base_dir / "data" / "candidates.csv")).resolve()
    output_dir = (args.output_dir or (base_dir / "output")).resolve()
    conflicts_file = (args.conflicts_file or (base_dir / "data" / "project" / "conflicts.yaml")).resolve()

    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")
    if not candidates_csv.exists():
        raise FileNotFoundError(f"Candidates CSV not found: {candidates_csv}")

    project_text_raw, project_doc_count = load_project_text(project_dir)
    if not project_text_raw.strip():
        raise ValueError("No usable text found in project documents")

    project_text = project_text_raw

    stop_words = language_stopwords(args.language)
    vectorizer = HashingVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words=stop_words,
        ngram_range=(1, 2),
        n_features=2**20,
        alternate_sign=False,
        norm=None,
    )

    project_vec = vectorizer.transform([project_text])
    project_vec = normalize(project_vec, norm="l2")

    inferred_pes, pe_counts = infer_project_pe(project_text_raw)
    if not inferred_pes:
        raise ValueError(
            "Unable to infer project PE from project documents. "
            "Add a PE code in project text (for example 'PE4')."
        )
    project_pes = set(inferred_pes)
    project_pe_primary = inferred_pes[0]
    ordered_counts = ", ".join(
        f"{key}:{value}" for key, value in sorted(pe_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    print(f"[INFO] Project PE auto-detected: {','.join(inferred_pes)} (counts: {ordered_counts})")

    candidates = load_online_candidates(candidates_csv, project_pes=project_pes)
    print(f"[INFO] Candidates after PE pre-filter ({','.join(inferred_pes)}): {len(candidates)}")
    if not candidates:
        raise ValueError(f"No candidates found with PE overlap in {','.join(inferred_pes)}")

    if args.full_scan:
        print("[INFO] Work scan mode: full-scan (no max limit per candidate/source)")
    else:
        print(f"[INFO] Work scan mode: capped (max_works={effective_max_works} per candidate/source)")

    rules = load_conflict_rules(conflicts_file)

    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        print(
            f"[INFO] Analyzing candidate {idx}/{len(candidates)}: "
            f"{candidate.candidate_id} | {candidate.name}"
        )
        resolved_institution = candidate.institution
        resolved_h_index: int | None = None

        aggregate_vec: csr_matrix | None = None
        works_total = 0
        citations_total = 0
        coauthors: set[str] = set()
        docs_used = 0

        works_openalex = 0
        works_semanticscholar = 0
        citations_openalex = 0
        citations_semanticscholar = 0

        openalex_author_id = ""
        semantic_author_id = ""

        if "openalex" in sources:
            try:
                author_id, resolved_institution, resolved_h_index = resolve_openalex_author_id(
                    candidate,
                    mailto=args.openalex_mailto,
                )
                openalex_author_id = author_id
                oa_vec, oa_works, oa_citations, oa_coauthors, oa_docs = stream_openalex_works(
                    author_id=author_id,
                    candidate_name=candidate.name,
                    vectorizer=vectorizer,
                    mailto=args.openalex_mailto,
                    from_year=args.from_year,
                    max_works=effective_max_works,
                    sleep_seconds=args.sleep_seconds,
                )
                aggregate_vec = oa_vec if aggregate_vec is None else (aggregate_vec + oa_vec)
                works_total += oa_works
                citations_total += oa_citations
                coauthors.update(oa_coauthors)
                docs_used += oa_docs
                works_openalex = oa_works
                citations_openalex = oa_citations
            except ValueError:
                print(
                    f"[WARN] OpenAlex author not found for candidate "
                    f"{candidate.candidate_id} ({candidate.name})."
                )

        if "semanticscholar" in sources:
            try:
                s2_author_id, s2_h_index = resolve_semantic_scholar_author_id(candidate, s2_api_key=s2_api_key)
                if s2_author_id:
                    semantic_author_id = s2_author_id
                    s2_vec, s2_works, s2_citations, s2_coauthors, s2_docs = stream_semantic_scholar_papers(
                        author_id=s2_author_id,
                        candidate_name=candidate.name,
                        vectorizer=vectorizer,
                        s2_api_key=s2_api_key,
                        from_year=args.from_year,
                        max_works=effective_max_works,
                        sleep_seconds=args.sleep_seconds,
                    )
                    aggregate_vec = s2_vec if aggregate_vec is None else (aggregate_vec + s2_vec)
                    works_total += s2_works
                    citations_total += s2_citations
                    coauthors.update(s2_coauthors)
                    docs_used += s2_docs
                    works_semanticscholar = s2_works
                    citations_semanticscholar = s2_citations
                    if resolved_h_index is None and s2_h_index is not None:
                        resolved_h_index = s2_h_index
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else "?"
                print(
                    f"[WARN] Semantic Scholar skipped for candidate {candidate.candidate_id} "
                    f"({candidate.name}) due to HTTP {status_code}"
                )

        if aggregate_vec is None:
            print(
                f"[WARN] Candidate skipped (no source data available): "
                f"{candidate.candidate_id} | {candidate.name}"
            )
            continue

        aggregate_vec = normalize(aggregate_vec, norm="l2")
        ml_score = float(cosine_similarity(aggregate_vec, project_vec)[0, 0])

        profile = MinimalConflictProfile(
            candidate_id=candidate.candidate_id,
            name=candidate.name,
            institution=resolved_institution,
            department=candidate.department,
            extracted_authors=coauthors,
        )
        reasons = conflict_reasons(profile, rules)
        is_conflict = len(reasons) > 0
        eligible = (not is_conflict) if args.conflict_mode == "exclude" else True

        activity_score = compute_activity_score(works_total, citations_total, resolved_h_index)

        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "name": candidate.name,
                "openalex_author_id": openalex_author_id,
                "semantic_scholar_author_id": semantic_author_id,
                "institution": resolved_institution,
                "department": candidate.department,
                "project_pe": project_pe_primary,
                "project_pes": ";".join(inferred_pes),
                "candidate_pe_areas": ";".join(sorted(candidate.pe_areas)),
                "pe_overlap_count": len(candidate.pe_areas.intersection(project_pes)),
                "sources_used": ",".join(sources),
                "ml_score": ml_score,
                "activity_score_raw": activity_score,
                "documents_used": docs_used,
                "works_total": works_total,
                "citations_total": citations_total,
                "works_openalex": works_openalex,
                "works_semanticscholar": works_semanticscholar,
                "citations_openalex": citations_openalex,
                "citations_semanticscholar": citations_semanticscholar,
                "h_index": resolved_h_index,
                "is_conflict": is_conflict,
                "eligible": eligible,
                "conflict_reasons": " | ".join(reasons),
            }
        )
        print(
            f"[INFO] Candidate done: {candidate.candidate_id} | works_total={works_total}, "
            f"citations_total={citations_total}, ml_score={ml_score:.4f}, eligible={eligible}"
        )

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        raise ValueError("No candidates processed")

    min_a = float(ranking["activity_score_raw"].min())
    max_a = float(ranking["activity_score_raw"].max())
    if max_a > min_a:
        ranking["activity_score"] = (ranking["activity_score_raw"] - min_a) / (max_a - min_a)
    else:
        ranking["activity_score"] = 0.0

    denom = max(1, len(project_pes))
    ranking["pe_overlap_score"] = ranking["pe_overlap_count"] / denom

    activity_weight = min(max(args.activity_weight, 0.0), 1.0)
    requested_pe_overlap_weight = min(max(args.pe_overlap_weight, 0.0), 1.0)
    # Keep PE overlap contribution much smaller than activity contribution.
    pe_overlap_weight_cap = activity_weight * 0.35
    pe_overlap_weight = min(requested_pe_overlap_weight, pe_overlap_weight_cap)
    if pe_overlap_weight < requested_pe_overlap_weight:
        print(
            "[INFO] pe-overlap-weight reduced to preserve relative importance: "
            f"requested={requested_pe_overlap_weight:.4f}, applied={pe_overlap_weight:.4f}, "
            f"activity_weight={activity_weight:.4f}"
        )
    base_ml_weight = max(0.0, 1.0 - activity_weight - pe_overlap_weight)
    ranking["final_score"] = (
        base_ml_weight * ranking["ml_score"]
        + activity_weight * ranking["activity_score"]
        + pe_overlap_weight * ranking["pe_overlap_score"]
    )

    ranking = ranking.sort_values(by=["eligible", "final_score"], ascending=[False, False]).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking = ranking[
        [
            "rank",
            "candidate_id",
            "name",
            "openalex_author_id",
            "semantic_scholar_author_id",
            "institution",
            "department",
            "project_pe",
            "project_pes",
            "candidate_pe_areas",
            "pe_overlap_count",
            "pe_overlap_score",
            "sources_used",
            "final_score",
            "ml_score",
            "activity_score",
            "works_total",
            "citations_total",
            "works_openalex",
            "works_semanticscholar",
            "citations_openalex",
            "citations_semanticscholar",
            "h_index",
            "documents_used",
            "is_conflict",
            "eligible",
            "conflict_reasons",
        ]
    ]

    eligible = ranking[ranking["eligible"] == True]
    if eligible.empty:
        raise ValueError("No eligible candidate after conflict filtering")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "online_referee_ranking.csv"
    md_path = output_dir / "online_referee_ranking.md"

    ranking.to_csv(csv_path, index=False)
    write_markdown_report(md_path, ranking, project_doc_count)

    winner = eligible.iloc[0]
    print("=== Recommended Referee (Online Streaming) ===")
    print(f"ID: {winner['candidate_id']}")
    print(f"Name: {winner['name']}")
    print(f"Final score: {winner['final_score']:.4f}")
    print(f"ML score: {winner['ml_score']:.4f}")
    print(f"Online works analyzed: {int(winner['works_total'])}")
    print(f"Candidates in conflict: {int(ranking['is_conflict'].sum())}")
    print()
    print(f"Ranking CSV: {csv_path}")
    print(f"Report Markdown: {md_path}")


if __name__ == "__main__":
    main()
