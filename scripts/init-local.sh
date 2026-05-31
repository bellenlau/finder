#!/usr/bin/env bash
set -euo pipefail

RECREATE_VENV=0
SKIP_INSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate-venv)
      RECREATE_VENV=1
      shift
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    -h|--help)
      echo "Uso: ./scripts/init-local.sh [--recreate-venv] [--skip-install]"
      exit 0
      ;;
    *)
      echo "Argomento non riconosciuto: $1" >&2
      echo "Uso: ./scripts/init-local.sh [--recreate-venv] [--skip-install]" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "[INFO] Repo root: $REPO_ROOT"

PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python non trovato nel PATH. Installa Python 3.10+ e riprova." >&2
  exit 1
fi

DATA_DIR="$REPO_ROOT/data"
PROJECT_DIR="$DATA_DIR/project"
CANDIDATES_DIR="$DATA_DIR/candidates"
OUTPUT_DIR="$REPO_ROOT/output"
TEMPLATE_CSV="$DATA_DIR/template.csv"
CANDIDATES_CSV="$DATA_DIR/candidates.csv"
CONFLICTS_YAML="$PROJECT_DIR/conflicts.yaml"
PROJECT_DESCRIPTION="$PROJECT_DIR/project_description.txt"
VENV_DIR="$REPO_ROOT/.venv"
VENV_PYTHON_LINUX="$VENV_DIR/bin/python"
VENV_PYTHON_WINDOWS="$VENV_DIR/Scripts/python.exe"
VENV_PYTHON=""

mkdir -p "$DATA_DIR" "$PROJECT_DIR" "$CANDIDATES_DIR" "$OUTPUT_DIR"

if [[ ! -f "$TEMPLATE_CSV" ]]; then
  echo "Template non trovato: $TEMPLATE_CSV" >&2
  exit 1
fi

if [[ ! -f "$CANDIDATES_CSV" ]]; then
  cp "$TEMPLATE_CSV" "$CANDIDATES_CSV"
  echo "[OK] Creato file candidati da template: data/candidates.csv"
else
  echo "[INFO] File gia presente, non sovrascrivo: data/candidates.csv"
fi

if [[ ! -f "$CONFLICTS_YAML" ]]; then
  cat > "$CONFLICTS_YAML" <<'YAML'
excluded_candidate_ids: []
excluded_names: []
project_institutions: []
project_departments: []
project_team: []
YAML
  echo "[OK] Creato template conflitti: data/project/conflicts.yaml"
else
  echo "[INFO] File conflitti gia presente, non sovrascrivo: data/project/conflicts.yaml"
fi

if [[ ! -f "$PROJECT_DESCRIPTION" ]]; then
  cat > "$PROJECT_DESCRIPTION" <<'TXT'
Incolla qui la descrizione del progetto.

Suggerimento: includi il codice PE (esempio: PE4) per abilitare il filtro PE automatico.
TXT
  echo "[OK] Creato file progetto: data/project/project_description.txt"
else
  echo "[INFO] File progetto gia presente, non sovrascrivo: data/project/project_description.txt"
fi

if [[ "$RECREATE_VENV" -eq 1 && -d "$VENV_DIR" ]]; then
  rm -rf "$VENV_DIR"
  echo "[INFO] Rimossa virtualenv esistente"
fi

if [[ -x "$VENV_PYTHON_LINUX" ]]; then
  VENV_PYTHON="$VENV_PYTHON_LINUX"
elif [[ -f "$VENV_PYTHON_WINDOWS" ]]; then
  VENV_PYTHON="$VENV_PYTHON_WINDOWS"
fi

if [[ -z "$VENV_PYTHON" ]]; then
  echo "[INFO] Creo virtualenv..."
  "$PYTHON_BIN" -m venv .venv
  if [[ -x "$VENV_PYTHON_LINUX" ]]; then
    VENV_PYTHON="$VENV_PYTHON_LINUX"
  elif [[ -f "$VENV_PYTHON_WINDOWS" ]]; then
    VENV_PYTHON="$VENV_PYTHON_WINDOWS"
  else
    echo "Impossibile trovare l'interprete della virtualenv appena creata." >&2
    exit 1
  fi
fi

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  echo "[INFO] Aggiorno pip e installo dipendenze..."
  "$VENV_PYTHON" -m pip install --upgrade pip
  "$VENV_PYTHON" -m pip install -r requirements.txt
else
  echo "[INFO] Installazione dipendenze saltata (--skip-install)"
fi

echo
echo "[DONE] Inizializzazione completata"
echo "Prossimi passi:"
if [[ "$VENV_PYTHON" == *"/bin/python" ]]; then
  echo "1) Attiva venv: source ./.venv/bin/activate"
else
  echo "1) Venv Windows rilevata da WSL: usa direttamente $VENV_PYTHON"
fi
echo "2) Compila data/candidates.csv e documenti in data/project/"
echo "3) Esegui: python src/online_referee_matcher.py --language en --sources openalex,semanticscholar --conflict-mode exclude --openalex-mailto your-email@example.org"
