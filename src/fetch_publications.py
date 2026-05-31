from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from candidates_loader import load_candidates_table


OPENALEX_BASE_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")[:100] or "paper"


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


def parse_sources(raw: str) -> list[str]:
    allowed = {"openalex", "semanticscholar"}
    parsed = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [item for item in parsed if item not in allowed]
    if unknown:
        raise ValueError(f"Sorgenti non supportate: {unknown}. Valori ammessi: {sorted(allowed)}")
    if not parsed:
        raise ValueError("Devi specificare almeno una sorgente in --sources")
    return parsed


def looks_like_candidate_author(candidate_name: str, authorships: list[dict[str, Any]]) -> bool:
    candidate_tokens = {t for t in re.findall(r"[a-z]+", candidate_name.lower()) if len(t) >= 3}
    if not candidate_tokens:
        return True

    for author_block in authorships:
        display_name = str(author_block.get("author", {}).get("display_name", "")).lower()
        author_tokens = set(re.findall(r"[a-z]+", display_name))
        overlap = candidate_tokens.intersection(author_tokens)
        if len(overlap) >= min(2, len(candidate_tokens)):
            return True
    return False


def looks_like_candidate_author_names(candidate_name: str, author_names: list[str]) -> bool:
    candidate_tokens = {t for t in re.findall(r"[a-z]+", candidate_name.lower()) if len(t) >= 3}
    if not candidate_tokens:
        return True

    for author_name in author_names:
        author_tokens = set(re.findall(r"[a-z]+", author_name.lower()))
        overlap = candidate_tokens.intersection(author_tokens)
        if len(overlap) >= min(2, len(candidate_tokens)):
            return True
    return False


def fetch_openalex_works(name: str, max_papers: int, from_year: int | None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "search": name,
        "per-page": min(max(max_papers * 2, 10), 100),
        "sort": "cited_by_count:desc",
    }

    if from_year is not None:
        params["filter"] = f"from_publication_date:{from_year}-01-01"

    response = requests.get(OPENALEX_BASE_URL, params=params, timeout=45)
    response.raise_for_status()

    raw_results = response.json().get("results", [])
    filtered: list[dict[str, Any]] = []
    for item in raw_results:
        authorships = item.get("authorships", [])
        if looks_like_candidate_author(name, authorships):
            filtered.append(item)
        if len(filtered) >= max_papers:
            break

    return filtered


def fetch_semantic_scholar_works(name: str, max_papers: int, from_year: int | None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "query": name,
        "limit": min(max(max_papers * 2, 10), 100),
        "fields": "title,year,abstract,authors,citationCount,venue,externalIds,url",
    }

    headers: dict[str, str] = {}
    api_key = os.getenv("S2_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key

    response = requests.get(SEMANTIC_SCHOLAR_BASE_URL, params=params, headers=headers, timeout=45)
    response.raise_for_status()

    raw_results = response.json().get("data", [])
    filtered: list[dict[str, Any]] = []
    for item in raw_results:
        year = item.get("year")
        if from_year is not None and isinstance(year, int) and year < from_year:
            continue

        author_names = [str(a.get("name", "")) for a in item.get("authors", []) if a.get("name")]
        if looks_like_candidate_author_names(name, author_names):
            filtered.append(item)
        if len(filtered) >= max_papers:
            break

    return filtered


def normalize_openalex_paper(paper: dict[str, Any]) -> dict[str, Any]:
    authors = []
    for author_block in paper.get("authorships", []):
        name = author_block.get("author", {}).get("display_name")
        if name:
            authors.append(str(name))

    venue = ""
    primary_location = paper.get("primary_location") or {}
    source = primary_location.get("source") or {}
    if source.get("display_name"):
        venue = str(source["display_name"])

    return {
        "source": "openalex",
        "title": normalize_text(str(paper.get("title", ""))),
        "year": paper.get("publication_year"),
        "doi": str(paper.get("doi", "") or "").replace("https://doi.org/", ""),
        "citations": int(paper.get("cited_by_count", 0) or 0),
        "venue": venue,
        "authors": authors,
        "abstract": restore_abstract(paper.get("abstract_inverted_index")),
        "url": str((paper.get("primary_location") or {}).get("landing_page_url", "") or ""),
    }


def normalize_semantic_scholar_paper(paper: dict[str, Any]) -> dict[str, Any]:
    external_ids = paper.get("externalIds") or {}
    doi = str(external_ids.get("DOI", "") or "")
    authors = [str(a.get("name", "")) for a in paper.get("authors", []) if a.get("name")]
    return {
        "source": "semanticscholar",
        "title": normalize_text(str(paper.get("title", ""))),
        "year": paper.get("year"),
        "doi": doi,
        "citations": int(paper.get("citationCount", 0) or 0),
        "venue": str(paper.get("venue", "") or ""),
        "authors": authors,
        "abstract": normalize_text(str(paper.get("abstract", "") or "")),
        "url": str(paper.get("url", "") or ""),
    }


def dedupe_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for paper in papers:
        title = normalize_text(str(paper.get("title", ""))).lower()
        year = str(paper.get("year", "") or "")
        doi = normalize_text(str(paper.get("doi", ""))).lower()
        key = f"doi:{doi}" if doi else f"title_year:{title}::{year}"
        if not title:
            continue

        existing = merged.get(key)
        if existing is None or int(paper.get("citations", 0) or 0) > int(existing.get("citations", 0) or 0):
            merged[key] = paper

    out = list(merged.values())
    out.sort(key=lambda x: int(x.get("citations", 0) or 0), reverse=True)
    return out


def write_publication_text(destination: Path, paper: dict[str, Any]) -> None:
    title = normalize_text(str(paper.get("title", "")))
    year = paper.get("year")
    doi = paper.get("doi", "")
    citation_count = paper.get("citations", 0)
    abstract = normalize_text(str(paper.get("abstract", "")))
    authors = [str(a) for a in paper.get("authors", [])]
    venue = str(paper.get("venue", ""))
    source = str(paper.get("source", ""))
    url = str(paper.get("url", ""))

    parts = [
        f"Source: {source}",
        f"Title: {title}",
        f"Year: {year}",
        f"DOI: {doi}",
        f"Citations: {citation_count}",
        f"Venue: {venue}",
        f"URL: {url}",
        f"Authors: {', '.join(authors)}",
        "",
        "Abstract:",
        abstract,
    ]

    destination.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scarica articoli da OpenAlex e/o Semantic Scholar per i candidati e popola data/candidates/<candidate_id>/auto. "
            "I file risultanti sono poi usati dal motore ML di referee_matcher.py."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Cartella base del progetto ref_finder",
    )
    parser.add_argument(
        "--candidates-csv",
        type=Path,
        default=None,
        help="CSV candidati (default: <base-dir>/data/candidates.csv)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=10,
        help="Numero massimo articoli per candidato",
    )
    parser.add_argument(
        "--max-total-papers",
        type=int,
        default=200,
        help="Numero massimo totale articoli da salvare su tutto il run",
    )
    parser.add_argument(
        "--from-year",
        type=int,
        default=None,
        help="Anno minimo di pubblicazione (es. 2018)",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.4,
        help="Pausa tra richieste API per ridurre rate limiting",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="openalex,semanticscholar",
        help="Sorgenti bibliografiche separate da virgola: openalex, semanticscholar",
    )
    parser.add_argument(
        "--min-citations",
        type=int,
        default=0,
        help="Scarta paper con meno citazioni di questa soglia",
    )
    parser.add_argument(
        "--write-mode",
        type=str,
        default="single-profile",
        choices=["single-profile", "per-paper"],
        help="single-profile: 1 file per candidato, per-paper: 1 file per articolo",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = args.base_dir.resolve()
    candidates_csv = (args.candidates_csv or (base_dir / "data" / "candidates.csv")).resolve()

    if not candidates_csv.exists():
        raise FileNotFoundError(f"CSV candidati non trovato: {candidates_csv}")

    df = load_candidates_table(candidates_csv)

    sources = parse_sources(args.sources)
    remaining_budget = max(0, int(args.max_total_papers))

    total_files = 0
    for _, row in df.iterrows():
        if remaining_budget <= 0:
            print("\n[INFO] Budget totale articoli esaurito, stop raccolta.")
            break

        candidate_id = str(row["candidate_id"]).strip()
        name = str(row["name"]).strip()
        query = str(row.get("scholar_query", "")).strip() or name

        print(f"\n[INFO] Download articoli per {name} (query: {query}, sources: {','.join(sources)})")

        unified: list[dict[str, Any]] = []
        if "openalex" in sources:
            openalex_papers = fetch_openalex_works(query, args.max_papers, args.from_year)
            unified.extend(normalize_openalex_paper(p) for p in openalex_papers)
        if "semanticscholar" in sources:
            s2_papers = fetch_semantic_scholar_works(query, args.max_papers, args.from_year)
            unified.extend(normalize_semantic_scholar_paper(p) for p in s2_papers)

        papers = [p for p in dedupe_papers(unified) if int(p.get("citations", 0) or 0) >= args.min_citations]

        allowed_for_candidate = min(args.max_papers, remaining_budget)
        papers = papers[:allowed_for_candidate]

        destination = base_dir / "data" / "candidates" / candidate_id / "auto"
        destination.mkdir(parents=True, exist_ok=True)

        saved = 0
        if args.write_mode == "single-profile":
            profile_path = destination / "_profile_corpus.txt"
            chunks: list[str] = []
            for idx, paper in enumerate(papers, start=1):
                title = normalize_text(str(paper.get("title", "")))
                if not title:
                    continue
                header = f"\n\n===== PAPER {idx:03d} =====\n"
                chunks.append(header)
                body_parts = [
                    f"Source: {paper.get('source', '')}",
                    f"Title: {title}",
                    f"Year: {paper.get('year', '')}",
                    f"DOI: {paper.get('doi', '')}",
                    f"Citations: {paper.get('citations', 0)}",
                    f"Venue: {paper.get('venue', '')}",
                    f"URL: {paper.get('url', '')}",
                    f"Authors: {', '.join(str(a) for a in paper.get('authors', []))}",
                    "",
                    "Abstract:",
                    str(paper.get("abstract", "") or ""),
                ]
                chunks.append("\n".join(body_parts))
                saved += 1

            profile_path.write_text("\n".join(chunks).strip() + "\n", encoding="utf-8")
        else:
            for idx, paper in enumerate(papers, start=1):
                title = normalize_text(str(paper.get("title", "")))
                if not title:
                    continue
                filename = f"{idx:03d}_{slugify(title)}.txt"
                path = destination / filename
                write_publication_text(path, paper)
                saved += 1

        total_files += saved
        remaining_budget -= saved
        print(f"[INFO] Salvati {saved} articoli in {destination}")
        time.sleep(max(0.0, args.sleep_seconds))

    print("\n[OK] Raccolta completata")
    print(f"File articoli creati: {total_files}")
    print("Ora puoi lanciare: python src/referee_matcher.py")


if __name__ == "__main__":
    main()
