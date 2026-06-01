from __future__ import annotations

import argparse
import difflib
import glob
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from candidates_loader import load_candidates_table


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
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

LANGUAGE_STOPWORDS: dict[str, list[str] | str | None] = {
    "en": "english",
    "it": list(ITALIAN_STOPWORDS),
    "none": None,
}


@dataclass
class CandidateProfile:
    candidate_id: str
    name: str
    text: str
    n_docs: int
    institution: str
    department: str
    extracted_authors: set[str]


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    try:
        reader = PdfReader(str(path))
    except Exception:
        return ""

    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def read_docx_file(path: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        return ""

    try:
        document = Document(str(path))
    except Exception:
        return ""

    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return ""
    if suffix in {".txt", ".md"}:
        return read_text_file(path)
    if suffix == ".pdf":
        return read_pdf_file(path)
    if suffix == ".docx":
        return read_docx_file(path)
    return ""


def discover_documents(root: Path) -> list[Path]:
    docs: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        docs.extend(root.rglob(f"*{ext}"))
    return sorted(set(docs))


def collect_text(paths: Iterable[Path]) -> tuple[str, int]:
    chunks: list[str] = []
    count = 0
    for path in paths:
        text = load_document(path)
        if text.strip():
            chunks.append(text)
            count += 1
    return "\n\n".join(chunks), count


def extract_author_names(text: str) -> set[str]:
    found: set[str] = set()
    for line in text.splitlines():
        if not line.lower().startswith("authors:"):
            continue
        raw = line.split(":", 1)[1] if ":" in line else ""
        names = [n.strip() for n in raw.split(",") if n.strip()]
        found.update(names)
    return found


def name_tokens(name: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z]+", name.lower()) if len(tok) >= 2}


DEPARTMENT_GENERIC_TOKENS = {
    "department",
    "departamento",
    "dipartimento",
    "division",
    "school",
    "faculty",
    "instituto",
    "institut",
    "institute",
    "sciences",
    "science",
    "physics",
    "fisica",
    "chimica",
    "chemistry",
    "engineering",
    "ingegneria",
    "university",
    "universita",
    "studies",
    "degli",
    "della",
    "delle",
    "di",
    "of",
    "and",
}


def normalize_org_text(value: str) -> str:
    lowered = value.lower().strip()
    lowered = unicodedata.normalize("NFKD", lowered)
    lowered = "".join(ch for ch in lowered if not unicodedata.combining(ch))
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def department_tokens(value: str) -> set[str]:
    normalized = normalize_org_text(value)
    tokens = {tok for tok in normalized.split(" ") if len(tok) >= 3}
    return {tok for tok in tokens if tok not in DEPARTMENT_GENERIC_TOKENS}


def departments_match(candidate_department: str, project_department: str) -> bool:
    cand_norm = normalize_org_text(candidate_department)
    proj_norm = normalize_org_text(project_department)
    if not cand_norm or not proj_norm:
        return False

    cand_tokens = department_tokens(candidate_department)
    proj_tokens = department_tokens(project_department)

    if cand_tokens and proj_tokens:
        overlap = len(cand_tokens.intersection(proj_tokens))
        if overlap >= min(2, len(cand_tokens), len(proj_tokens)):
            return True
        if overlap >= 1 and (len(cand_tokens) <= 2 or len(proj_tokens) <= 2):
            return True

    similarity = difflib.SequenceMatcher(a=cand_norm, b=proj_norm).ratio()
    if similarity >= 0.86:
        return True

    return False


def similar_person_name(a: str, b: str) -> bool:
    ta = name_tokens(a)
    tb = name_tokens(b)
    if not ta or not tb:
        return False
    overlap = ta.intersection(tb)
    return len(overlap) >= min(2, len(ta), len(tb))


def load_conflict_rules(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {
            "project_team": set(),
            "project_institutions": set(),
            "project_departments": set(),
            "excluded_candidate_ids": set(),
            "excluded_names": set(),
        }

    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def as_set(key: str) -> set[str]:
        values = content.get(key, [])
        if not isinstance(values, list):
            return set()
        return {str(v).strip() for v in values if str(v).strip()}

    return {
        "project_team": as_set("project_team"),
        "project_institutions": as_set("project_institutions"),
        "project_departments": as_set("project_departments"),
        "excluded_candidate_ids": as_set("excluded_candidate_ids"),
        "excluded_names": as_set("excluded_names"),
    }


def conflict_reasons(profile: CandidateProfile, rules: dict[str, set[str]]) -> list[str]:
    reasons: list[str] = []

    if profile.candidate_id in rules["excluded_candidate_ids"]:
        reasons.append("candidate_id in blacklist")

    for forbidden_name in rules["excluded_names"]:
        if similar_person_name(profile.name, forbidden_name):
            reasons.append(f"nome in blacklist: {forbidden_name}")
            break

    lowered_institution = profile.institution.lower().strip()
    if lowered_institution:
        for project_inst in rules["project_institutions"]:
            if project_inst.lower() in lowered_institution or lowered_institution in project_inst.lower():
                reasons.append(f"stesso ente: {profile.institution}")
                break

    lowered_department = profile.department.lower().strip()
    if lowered_department:
        for project_dep in rules["project_departments"]:
            if departments_match(profile.department, project_dep):
                reasons.append(f"stesso dipartimento: {profile.department}")
                break

    project_team = rules["project_team"]
    observed_names = set(profile.extracted_authors)
    for person in project_team:
        if any(similar_person_name(person, observed) for observed in observed_names):
            reasons.append(f"coautore nel team progetto: {person}")
            break

    return reasons


def get_candidate_docs(base_dir: Path, row: pd.Series) -> list[Path]:
    docs_glob = str(row.get("docs_glob", "")).strip()
    candidate_id = str(row["candidate_id"]).strip()

    if docs_glob:
        matches = [Path(p) for p in glob.glob(str(base_dir / docs_glob), recursive=True)]
        return [p for p in matches if p.is_file()]

    candidate_dir = base_dir / "data" / "candidates" / candidate_id
    if not candidate_dir.exists():
        return []
    return discover_documents(candidate_dir)


def load_candidates(base_dir: Path, candidates_csv: Path) -> list[CandidateProfile]:
    df = load_candidates_table(candidates_csv)

    profiles: list[CandidateProfile] = []
    for _, row in df.iterrows():
        docs = get_candidate_docs(base_dir, row)
        docs_text, n_docs = collect_text(docs)

        notes = str(row.get("notes", "")).strip()
        merged = f"{notes}\n\n{docs_text}".strip()
        institution = str(row.get("institution", "")).strip()
        department = str(row.get("department", "")).strip()
        extracted_authors = extract_author_names(docs_text)

        profiles.append(
            CandidateProfile(
                candidate_id=str(row["candidate_id"]).strip(),
                name=str(row["name"]).strip(),
                text=merged,
                n_docs=n_docs,
                institution=institution,
                department=department,
                extracted_authors=extracted_authors,
            )
        )

    return profiles


def load_project_text(project_dir: Path) -> tuple[str, int]:
    docs = discover_documents(project_dir)
    return collect_text(docs)


def top_shared_terms(
    project_vector,
    candidate_vector,
    feature_names: np.ndarray,
    top_n: int,
) -> str:
    shared = candidate_vector.multiply(project_vector)
    if shared.nnz == 0:
        return ""

    order = np.argsort(shared.data)[::-1][:top_n]
    terms = [feature_names[shared.indices[i]] for i in order]
    return ", ".join(terms)


def rank_candidates(
    project_text: str,
    candidates: list[CandidateProfile],
    top_terms: int,
    language: str,
    conflict_rules: dict[str, set[str]],
    conflict_mode: str,
) -> pd.DataFrame:
    valid_candidates = [c for c in candidates if c.text.strip()]
    if not valid_candidates:
        raise ValueError("Nessun candidato con testo utile trovato.")

    corpus = [project_text] + [c.text for c in valid_candidates]
    stop_words = LANGUAGE_STOPWORDS[language]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=50000,
        stop_words=stop_words,
    )
    matrix = vectorizer.fit_transform(corpus)

    project_vector = matrix[0]
    candidate_vectors = matrix[1:]

    similarities = cosine_similarity(candidate_vectors, project_vector).ravel()
    feature_names = vectorizer.get_feature_names_out()

    rows = []
    for idx, candidate in enumerate(valid_candidates):
        explanation = top_shared_terms(
            project_vector=project_vector,
            candidate_vector=candidate_vectors[idx],
            feature_names=feature_names,
            top_n=top_terms,
        )
        reasons = conflict_reasons(candidate, conflict_rules)
        is_conflict = len(reasons) > 0
        eligible = (not is_conflict) if conflict_mode == "exclude" else True
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "name": candidate.name,
                "ml_score": float(similarities[idx]),
                "documents_used": candidate.n_docs,
                "shared_terms": explanation,
                "is_conflict": is_conflict,
                "conflict_reasons": " | ".join(reasons),
                "eligible": eligible,
            }
        )

    result = pd.DataFrame(rows)
    result = result.sort_values(by=["eligible", "ml_score"], ascending=[False, False]).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result[
        [
            "rank",
            "candidate_id",
            "name",
            "ml_score",
            "documents_used",
            "is_conflict",
            "eligible",
            "conflict_reasons",
            "shared_terms",
        ]
    ]


def write_markdown_report(output_path: Path, ranking: pd.DataFrame, project_doc_count: int) -> None:
    eligible = ranking[ranking["eligible"] == True]
    if eligible.empty:
        raise ValueError("Nessun candidato eleggibile dopo il filtro conflitti.")
    best = eligible.iloc[0]

    lines = [
        "# Ranking referee scientifici",
        "",
        f"Documenti progetto analizzati: **{project_doc_count}**",
        "",
        "## Miglior referee suggerito",
        "",
        f"- ID: **{best['candidate_id']}**",
        f"- Nome: **{best['name']}**",
        f"- Punteggio ML (compatibilita' scientifica): **{best['ml_score']:.4f}**",
        f"- Termini condivisi principali: {best['shared_terms'] or 'n/d'}",
        "",
        "## Classifica completa",
        "",
        ranking.to_markdown(index=False),
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seleziona il referee scientifico ottimale usando un modello ML di similarita' testuale "
            "basato su TF-IDF + cosine similarity."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Cartella base del progetto ref_finder",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Cartella dei documenti progetto (default: <base-dir>/data/project)",
    )
    parser.add_argument(
        "--candidates-csv",
        type=Path,
        default=None,
        help="CSV candidati (default: <base-dir>/data/candidates.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Cartella output (default: <base-dir>/output)",
    )
    parser.add_argument(
        "--top-terms",
        type=int,
        default=12,
        help="Numero di termini condivisi da mostrare per candidato",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "it", "none"],
        help="Lingua corpus per stopwords TF-IDF: en (default), it, none",
    )
    parser.add_argument(
        "--conflicts-file",
        type=Path,
        default=None,
        help="File YAML regole conflitto (default: <base-dir>/data/project/conflicts.yaml)",
    )
    parser.add_argument(
        "--conflict-mode",
        type=str,
        default="exclude",
        choices=["exclude", "flag"],
        help="exclude: rimuove i candidati in conflitto dalla scelta; flag: li segnala soltanto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = args.base_dir.resolve()
    project_dir = (args.project_dir or (base_dir / "data" / "project")).resolve()
    candidates_csv = (args.candidates_csv or (base_dir / "data" / "candidates.csv")).resolve()
    output_dir = (args.output_dir or (base_dir / "output")).resolve()
    conflicts_file = (args.conflicts_file or (base_dir / "data" / "project" / "conflicts.yaml")).resolve()

    if not project_dir.exists():
        raise FileNotFoundError(f"Cartella progetto non trovata: {project_dir}")
    if not candidates_csv.exists():
        raise FileNotFoundError(f"CSV candidati non trovato: {candidates_csv}")

    project_text, project_doc_count = load_project_text(project_dir)
    if not project_text.strip():
        raise ValueError("Nessun testo utile trovato nei documenti del progetto.")

    candidates = load_candidates(base_dir=base_dir, candidates_csv=candidates_csv)
    rules = load_conflict_rules(conflicts_file)
    ranking = rank_candidates(
        project_text=project_text,
        candidates=candidates,
        top_terms=args.top_terms,
        language=args.language,
        conflict_rules=rules,
        conflict_mode=args.conflict_mode,
    )

    eligible = ranking[ranking["eligible"] == True]
    if eligible.empty:
        raise ValueError(
            "Nessun candidato eleggibile dopo il filtro conflitti. "
            "Controlla data/project/conflicts.yaml o usa --conflict-mode flag."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "referee_ranking.csv"
    md_path = output_dir / "referee_ranking.md"

    ranking.to_csv(csv_path, index=False)
    write_markdown_report(md_path, ranking, project_doc_count)

    winner = eligible.iloc[0]
    print("=== Referee consigliato (ML) ===")
    print(f"ID: {winner['candidate_id']}")
    print(f"Nome: {winner['name']}")
    print(f"Punteggio: {winner['ml_score']:.4f}")
    print(f"Candidati in conflitto: {int(ranking['is_conflict'].sum())}")
    print(f"Termini condivisi: {winner['shared_terms'] or 'n/d'}")
    print()
    print(f"Ranking CSV: {csv_path}")
    print(f"Report Markdown: {md_path}")


if __name__ == "__main__":
    main()
