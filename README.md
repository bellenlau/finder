# Ref Finder ML

Strumento per selezionare il referee scientifico ottimale da una lista di candidati, usando compatibilita' scientifica calcolata con machine learning sui testi:
- articoli scientifici dei candidati
- documenti interni del progetto

Il modello usa similarita' semantica ibrida (lessicale + embeddings scientifici) sul corpus documentale.
Default lingua analisi: inglese (`--language en`).

## Policy backup del repo

Questo repository contiene solo file backupabili (codice, configurazioni e template).

- input reali: non versionati
- output generati: non versionati
- template versionato: `data/template.csv`

## Modalita' scalabile senza download pubblicazioni

Se non vuoi scaricare articoli in locale, usa il motore online streaming:

```bash
python src/online_referee_matcher.py --language en --sources openalex,semanticscholar --conflict-mode exclude --openalex-mailto your-email@example.org
```

In questo caso la PE viene inferita automaticamente dai documenti progetto (pattern tipo `PE4`, `PE5`, ...).
Se nel progetto compaiono piu' PE, vengono considerate tutte.

Per usare direttamente la tabella originale `candidates_latest.csv` (separatore `;`, colonne italiane):

```bash
python src/online_referee_matcher.py --candidates-csv ./data/candidates_latest.csv --language en --sources openalex,semanticscholar --conflict-mode exclude --openalex-mailto your-email@example.org
```

Caratteristiche:
- analizza il contenuto scientifico dei candidati via sorgenti multiple (OpenAlex + Semantic Scholar)
- applica il filtro PE all'inizio della pipeline (prima di qualsiasi chiamata API)
- analizza solo candidati con almeno una PE in comune col progetto (PE inferite automaticamente dal testo progetto)
- espande automaticamente i termini scientifici (sinonimi/concetti correlati) per migliorare il matching di sotto-argomenti
- legge i metadati bibliografici online (titolo + abstract, non full-text PDF) e seleziona i lavori piu' affini all'argomento del progetto
- supporta embeddings scientifici (`sentence-transformers`, default modello `allenai-specter`) con fallback lessicale
- non richiede alcun parametro PE da CLI
- legge in automatico sia schema canonico sia schema `candidates_latest` (`Referee ID`, `Nome`, `Cognome`, `Panel ID1..5`)
- non salva pubblicazioni su disco
- mantiene solo output finali di ranking in `output/`
- applica comunque il filtro conflitti di interesse

Output modalita' online:
- `output/online_referee_ranking.csv`
- `output/online_referee_ranking.md`

Note:
- puoi fornire `openalex_author_id` nel CSV per evitare errori di disambiguazione autore
- puoi usare `--sources` per selezionare le sorgenti (`openalex`, `semanticscholar`)
- per Semantic Scholar puoi passare `--s2-api-key` oppure usare la variabile ambiente `S2_API_KEY`
- il ranking finale online combina `ml_score` (profilo candidato vs progetto) e `focus_score` (media top-k paper piu' affini)
- puoi regolare il blend con `--focus-weight` (default `0.35`) e `--focus-top-k` (default `8`)
- formula punteggio finale: `final_score = (1 - focus_weight) * ml_score + focus_weight * focus_score`
- puoi configurare embeddings con `--embedding-model` (default `allenai-specter`)
- puoi bilanciare matching lessicale/embedding con `--embedding-weight` (default `0.60`)
- puoi forzare il device embeddings con `--embedding-device` (`auto|cpu|cuda|mps`)
- se il caricamento embeddings si blocca, usa `--embedding-device cpu` o disattiva embeddings con `--embedding-weight 0` (in questo caso il modello embeddings non viene caricato)
- default `--max-works` e' `100` per sorgente/candidato
- full scan solo esplicito con `--full-scan`
- puoi regolare dettagli debug con `--debug-top-n` (default `5`)
- il log runtime riporta quale candidato e' in analisi e lo stato di completamento

Nel CSV online e' presente la colonna `debug_focus_top_works` con i principali lavori che hanno contribuito al `focus_score`.
Formato: `source|affinity|year|title` separati da ` || `.
Sono inoltre presenti le colonne `final_score`, `ml_score` e `focus_score` per audit completo del ranking.

## Struttura

- `data/template.csv`: template CSV da usare come base per i dati reali
- `src/referee_matcher.py`: motore di ranking ML
- `output/`: ranking generato (locale, non versionato)

## Formato `data/candidates.csv`

Colonne obbligatorie:
- `candidate_id`
- `name`

Colonne opzionali:
- `docs_glob`: pattern glob relativo alla root del progetto (es. `data/candidates/cand_001/**/*`)
- `notes`: testo libero con parole chiave o info aggiuntive
- `scholar_query`: stringa query autore per il recupero automatico da rete
- `openalex_author_id`: ID autore OpenAlex (consigliato per analisi online completa e robusta)
- `pe_areas`: aree PE del candidato, separate da `;` o `,` (es. `PE4;PE5`)
- `institution`: ente del candidato (usato dal filtro conflitti)
- `department`: dipartimento del candidato (usato dal filtro conflitti su `project_departments`)

Note compatibilita' CSV legacy:
- in `candidates_latest.csv` vengono gestiti anche alias come `Ente dipartimento` / `Department` per popolare automaticamente `department`

Esempio:

```csv
candidate_id,name,docs_glob,notes,scholar_query
cand_001,Dr. Alice Rossi,data/candidates/cand_001/**/*,"DFT, materiali","Alice Rossi computational materials"
```

## Raccolta automatica articoli da rete

Per popolare automaticamente il database pubblicazioni dei candidati puoi usare lo script:

```bash
python src/fetch_publications.py --max-papers 25 --from-year 2018 --sources openalex,semanticscholar
```

Modalita' leggera consigliata (pochi download):

```bash
python src/fetch_publications.py --sources openalex,semanticscholar --max-papers 6 --max-total-papers 80 --from-year 2021 --min-citations 10 --write-mode single-profile
```

Con `--write-mode single-profile` viene scritto un solo file `_profile_corpus.txt` per candidato (invece di un file per paper).

Cosa fa:
- interroga OpenAlex e Semantic Scholar (fonti bibliografiche aperte, compatibili con workflow "stile Google Scholar")
- salva i paper in `data/candidates/<candidate_id>/auto/*.txt`
- ogni file contiene titolo, autori, venue, DOI, citazioni e abstract

Se hai una API key Semantic Scholar, puoi impostarla per aumentare l'affidabilita' delle richieste:

```bash
export S2_API_KEY="<tua_api_key>"
```

Poi esegui il ranking ML:

```bash
python src/referee_matcher.py
```

## Filtro conflitti di interesse

Il ranking applica automaticamente un filtro conflitti usando `data/project/conflicts.yaml`.

Regole supportate:
- `excluded_candidate_ids`: esclusione diretta per ID
- `excluded_names`: esclusione diretta per nominativo
- `project_institutions`: conflitto se il candidato risulta dello stesso ente
- `project_departments`: conflitto se il candidato risulta dello stesso dipartimento del PI/team
- `project_team`: conflitto se compaiono coautori nel team progetto (da righe `Authors:` nei documenti)

Nota sul match dipartimento:
- il confronto e' normalizzato (accenti/simboli/spazi)
- usa overlap token + fuzzy similarity per gestire varianti lessicali (es. IT/EN)

Modalita':
- `--conflict-mode exclude` (default): candidati in conflitto non eleggibili
- `--conflict-mode flag`: conflitti solo segnalati, ma candidati ancora selezionabili

## Setup

```bash
cd ref_finder/finder
python -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
```

In alternativa, inizializzazione automatica per nuovo utilizzo:

```bash
cd ref_finder/finder
chmod +x ./scripts/init-local.sh
./scripts/init-local.sh
```

Opzioni utili script Bash:
- `--skip-install`: prepara solo struttura file/cartelle senza installare dipendenze
- `--recreate-venv`: ricrea la virtualenv da zero

## Esecuzione

```bash
python src/referee_matcher.py
```

Output generati:
- `output/referee_ranking.csv`
- `output/referee_ranking.md`

## Parametri utili

```bash
python src/referee_matcher.py \
  --base-dir . \
  --project-dir ./data/project \
  --candidates-csv ./data/candidates.csv \
  --output-dir ./output \
  --top-terms 12 \
  --language en \
  --conflicts-file ./data/project/conflicts.yaml \
  --conflict-mode exclude
```

## Come personalizzarlo sul tuo caso reale

1. Sostituisci i file in `data/project/` con i tuoi documenti reali.
2. Inserisci i candidati reali in `data/candidates.csv`.
3. Crea una cartella per ogni candidato in `data/candidates/<candidate_id>/` e carica gli articoli/documenti.
4. Esegui lo script e usa il primo in classifica come referee consigliato.

Workflow consigliato completo:

1. Aggiorna `data/candidates.csv` con candidati e `scholar_query`.
2. Esegui `python src/fetch_publications.py --sources openalex,semanticscholar` per creare il database articoli da rete.
3. Aggiungi eventuali documenti interni in `data/candidates/<candidate_id>/`.
4. Definisci il file `data/project/conflicts.yaml` con le tue regole COI.
5. Esegui `python src/referee_matcher.py` per ottenere il referee ottimale.

## Limiti del modello attuale

- Il ranking resta principalmente testuale-semantico; il filtro conflitti e' rule-based.
- Non usa supervisione con etichette storiche (e' un modello unsupervised).
- Se vuoi, nel passo successivo possiamo aggiungere:
  - embeddings transformer (es. SciBERT/SentenceTransformers)
  - rilevazione conflitti temporale piu' rigorosa (es. co-autori negli ultimi N anni via metadati strutturati)
  - calibrazione pesi per documenti interni vs pubblicazioni
