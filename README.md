# Ref Finder

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Shell-Bash](https://img.shields.io/badge/shell-bash-informational)
![Online Sources](https://img.shields.io/badge/sources-OpenAlex%20%2B%20Semantic%20Scholar-success)
![Status](https://img.shields.io/badge/status-active-brightgreen)

## TL;DR

Ref Finder ranks scientific reviewer candidates against project documents, then enforces conflict-of-interest rules.

- Best for: reviewer recommendation with transparent scoring and COI controls
- Input: project docs + candidate table
- Output: ranked CSV/Markdown reports with debug evidence columns

## Quickstart (Copy/Paste)

```bash
cd ref_finder/finder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/online_referee_matcher.py \
  --language en \
  --sources openalex,semanticscholar \
  --conflict-mode exclude \
  --openalex-mailto your-email@example.org
```

Ref Finder recommends the best scientific reviewer from a candidate list by matching project content against candidate publication profiles, then applying conflict-of-interest (COI) filters.

The project includes two ranking paths:
- offline: local documents and TF-IDF similarity
- online streaming: OpenAlex and Semantic Scholar metadata, with optional sentence-transformer embeddings

## Highlights

- scientific relevance ranking from project text and candidate publication signals
- online streaming mode with no local publication dump required
- automatic PE code detection from project documents (for example PE4, PE5)
- hard PE pre-filter before API calls
- COI filtering for:
  - explicit candidate/name exclusions
  - same institution
  - same department
  - co-authorship with project team
- hybrid scoring in online mode:
  - ml_score (profile-to-project match)
  - focus_score (top-k most relevant selected works)
  - final_score blend
- built-in debug columns to inspect why candidates rank high

## How Ranking Works

### Online mode scoring

For each candidate:
- collect works from selected sources
- rank works by project affinity
- keep selected works and compute:
  - ml_score
  - focus_score
- combine scores:

final_score = (1 - focus_weight) * ml_score + focus_weight * focus_score

### Embeddings behavior

Embeddings are integrated in online mode only when all these are true:
- sentence-transformers is installed
- embedding_weight > 0
- the model loads successfully

At startup, check logs:
- Similarity mode: hybrid -> embeddings active
- Similarity mode: lexical-only -> embeddings not contributing

To explicitly disable embeddings:
- set --embedding-weight 0

To force CPU execution for embeddings:
- set --embedding-device cpu

## Repository Layout

- src/referee_matcher.py: offline ranking engine
- src/online_referee_matcher.py: online streaming ranking engine
- src/fetch_publications.py: optional local publication fetcher
- src/candidates_loader.py: canonical + legacy candidates CSV normalization
- data/project/: project documents and conflicts file
- data/template.csv: candidate table template
- output/: generated rankings

## Requirements

- Python 3.10+
- bash shell
- internet access for online mode

## Installation

```bash
cd ref_finder/finder
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional one-shot initializer:

```bash
cd ref_finder/finder
chmod +x ./scripts/init-local.sh
./scripts/init-local.sh
```

Useful init flags:

```bash
./scripts/init-local.sh --help
./scripts/init-local.sh --recreate-venv
./scripts/init-local.sh --skip-install
```

## Quick Start (Online Streaming)

```bash
cd ref_finder/finder
source .venv/bin/activate
python3 src/online_referee_matcher.py \
  --language en \
  --sources openalex,semanticscholar \
  --conflict-mode exclude \
  --openalex-mailto your-email@example.org
```

Run with legacy candidates file:

```bash
python3 src/online_referee_matcher.py \
  --candidates-csv ./data/candidates_latest.csv \
  --language en \
  --sources openalex,semanticscholar \
  --conflict-mode exclude \
  --openalex-mailto your-email@example.org
```

## Input Data

### Project documents

Place project files under data/project/ using supported extensions:
- .txt
- .md
- .pdf
- .docx

Include PE codes in project text if you want automatic PE filtering.

### Candidates table

Use data/candidates.csv (or a legacy-compatible file).

Required columns:
- candidate_id
- name

Optional columns:
- docs_glob
- notes
- scholar_query
- openalex_author_id
- pe_areas
- institution
- department

Legacy aliases are supported for department, including Ente dipartimento and Department.

## Conflict Rules

Configure COI in data/project/conflicts.yaml.

Create a starter file from bash:

```bash
cat > data/project/conflicts.yaml <<'YAML'
excluded_candidate_ids: []
excluded_names: []
project_institutions: []
project_departments: []
project_team: []
YAML
```

Department COI matching is normalized and fuzzy-aware (accents/symbols/token overlap + similarity).

## Output Files

Online mode writes:
- output/online_referee_ranking.csv
- output/online_referee_ranking.md

Offline mode writes:
- output/referee_ranking.csv
- output/referee_ranking.md

Important online output columns:
- final_score
- ml_score
- focus_score
- debug_focus_top_works
- is_conflict
- eligible
- conflict_reasons

debug_focus_top_works format:
- source|affinity|year|title entries joined by ||

## Key CLI Options (Online)

- --sources openalex,semanticscholar
- --openalex-mailto <email>
- --openalex-max-retries 10
- --openalex-backoff-seconds 3.0
- --openalex-min-interval-seconds 1.2
- --openalex-jitter-seconds 0.3
- --openalex-max-wait-seconds 90
- --s2-api-key <key> (or use S2_API_KEY)
- --embedding-model allenai-specter
- --embedding-weight 0.60
- --embedding-device auto|cpu|cuda|mps
- --focus-weight 0.35
- --focus-top-k 8
- --max-works 100
- --full-scan
- --debug-top-n 5
- --resume
- --checkpoint-file online_referee_checkpoint.json
- --conflict-mode exclude|flag

## Checkpoint And Restart (Online)

The online matcher now supports checkpointing to safely restart long runs.

- default checkpoint path: output/online_referee_checkpoint.json
- checkpoint is always written inside output/
- checkpoint is saved after each processed candidate
- both processed and skipped candidates are tracked

First run:

```bash
python3 src/online_referee_matcher.py \
  --language en \
  --sources openalex,semanticscholar \
  --conflict-mode exclude \
  --openalex-mailto your-email@example.org
```

Restart from checkpoint:

```bash
python3 src/online_referee_matcher.py \
  --language en \
  --sources openalex,semanticscholar \
  --conflict-mode exclude \
  --openalex-mailto your-email@example.org \
  --resume
```

Use a custom checkpoint file:

```bash
python3 src/online_referee_matcher.py \
  --resume \
  --checkpoint-file my_run_checkpoint.json
```

Note:
- resume validates run configuration against the saved checkpoint
- if configuration differs, the script stops with a signature mismatch error
- checkpoints from older script versions are accepted when core run settings match

OpenAlex rate-limit note:
- set `--openalex-mailto` to reduce 429 risk
- matcher now applies polite throttling + jitter by default for OpenAlex
- very large `Retry-After` values are capped by `--openalex-max-wait-seconds`

## Offline Workflow (Optional)

Fetch local candidate corpora:

```bash
python3 src/fetch_publications.py \
  --sources openalex,semanticscholar \
  --max-papers 25 \
  --from-year 2018
```

Run offline ranking:

```bash
python3 src/referee_matcher.py \
  --base-dir . \
  --project-dir ./data/project \
  --candidates-csv ./data/candidates.csv \
  --output-dir ./output \
  --language en \
  --conflicts-file ./data/project/conflicts.yaml \
  --conflict-mode exclude
```

## Debugging Hangs and Slow Runs

Disable embeddings entirely:

```bash
python3 src/online_referee_matcher.py --embedding-weight 0
```

Force embeddings on CPU:

```bash
python3 src/online_referee_matcher.py --embedding-device cpu
```

Run in background and stream logs:

```bash
nohup python3 src/online_referee_matcher.py --embedding-device cpu > run.log 2>&1 &
tail -f run.log
```

## Current Limits

- ranking is primarily text/semantic matching plus rule-based COI filtering
- online mode uses title + abstract metadata, not full-text PDFs
- result quality depends on OpenAlex and Semantic Scholar metadata coverage
- method is unsupervised (no historical assignment labels used)

## License

See repository license files for terms and notices.
