from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


ALIASES = {
    "candidate_id": ["candidate_id", "Referee ID", "referee_id", "id"],
    "name": ["name", "full_name", "Full Name"],
    "first_name": ["first_name", "Nome", "First Name"],
    "last_name": ["last_name", "Cognome", "Last Name", "Surname"],
    "scholar_query": ["scholar_query", "Scholar Query"],
    "institution": ["institution", "Ente nome", "Affiliation", "Institution"],
    "department": ["department", "Ente dipartimento", "Department", "Affiliation Department"],
    "openalex_author_id": ["openalex_author_id", "OpenAlex Author ID"],
    "docs_glob": ["docs_glob", "Docs Glob"],
    "notes": ["notes", "Notes"],
}


def _as_str(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _pick(row: pd.Series, names: list[str]) -> str:
    for name in names:
        if name in row.index:
            value = _as_str(row.get(name, ""))
            if value:
                return value
    return ""


def _read_candidates_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Candidates table not found: {path}")

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    errors: list[str] = []

    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
            if len(df.columns) > 1:
                return df
        except Exception:
            errors.append(f"pandas autodetect failed ({enc})")

    for enc in encodings:
        try:
            return pd.read_csv(path, sep=";", encoding=enc)
        except Exception:
            errors.append(f"pandas semicolon failed ({enc})")

    # Last resort: plain csv parser with explicit delimiters.
    for enc in encodings:
        for delimiter in [";", ",", "\t"]:
            try:
                with path.open("r", encoding=enc, errors="ignore", newline="") as handle:
                    reader = csv.DictReader(handle, delimiter=delimiter)
                    fieldnames = reader.fieldnames or []
                    rows = list(reader)
                if rows and len(fieldnames) > 1:
                    return pd.DataFrame(rows)
            except Exception:
                errors.append(f"csv fallback failed ({enc}, delimiter={delimiter})")

    short_errors = "; ".join(errors[:6])
    raise ValueError(f"Unable to read candidates table: {path}. Attempts: {short_errors}")


def _extract_pe_areas(row: pd.Series, pe_column: str) -> str:
    values: list[str] = []

    if pe_column in row.index:
        values.extend(re.split(r"[;,|]", _as_str(row.get(pe_column, ""))))

    panel_columns = [col for col in row.index if str(col).strip().lower().startswith("panel id")]
    for col in panel_columns:
        values.extend(re.split(r"[;,|]", _as_str(row.get(col, ""))))

    codes = []
    for value in values:
        token = _as_str(value).upper().replace(" ", "")
        if re.fullmatch(r"(PE|LS)\d+", token):
            codes.append(token)

    deduped = sorted(set(codes))
    return ";".join(deduped)


def _extract_notes(row: pd.Series) -> str:
    explicit = _pick(row, ALIASES["notes"])
    if explicit:
        return explicit

    keywords = []
    for idx in range(1, 6):
        key = f"Keyword {idx}"
        if key in row.index:
            value = _as_str(row.get(key, ""))
            if value:
                keywords.append(value)
    return "; ".join(keywords)


def load_candidates_table(path: Path, pe_column: str = "pe_areas") -> pd.DataFrame:
    raw = _read_candidates_dataframe(path)

    rows: list[dict[str, str]] = []
    for _, row in raw.iterrows():
        candidate_id = _pick(row, ALIASES["candidate_id"])
        first_name = _pick(row, ALIASES["first_name"])
        last_name = _pick(row, ALIASES["last_name"])
        full_name = _pick(row, ALIASES["name"]) or f"{first_name} {last_name}".strip()

        if not candidate_id or not full_name:
            continue

        scholar_query = _pick(row, ALIASES["scholar_query"]) or full_name
        institution = _pick(row, ALIASES["institution"])
        department = _pick(row, ALIASES["department"])
        openalex_author_id = _pick(row, ALIASES["openalex_author_id"])
        docs_glob = _pick(row, ALIASES["docs_glob"])
        notes = _extract_notes(row)
        pe_areas = _extract_pe_areas(row, pe_column=pe_column)

        rows.append(
            {
                "candidate_id": candidate_id,
                "name": full_name,
                "scholar_query": scholar_query,
                "institution": institution,
                "department": department,
                "openalex_author_id": openalex_author_id,
                "docs_glob": docs_glob,
                "notes": notes,
                "pe_areas": pe_areas,
            }
        )

    out = pd.DataFrame(rows)
    required = {"candidate_id", "name"}
    if out.empty or not required.issubset(out.columns):
        raise ValueError("Candidates table missing required information for candidate_id/name")
    return out
