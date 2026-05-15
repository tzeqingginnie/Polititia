# Politica NLP Project

Static dashboard and data-analysis pipeline for French parliamentary speech
transcripts from the Assemblee nationale open-data Syceron XML archive.

## Data Source

Raw XML is not committed. Download it from the official Assemblee nationale
open-data endpoint:

https://data.assemblee-nationale.fr/travaux-parlementaires/debats

The downloader writes the archive and extracted XML under `data/raw/`, which is
ignored by git.

## Rebuild Pipeline

Run from the repository root.

```bash
python3 scripts/download_assemblee_data.py
python3 extract_speeches.py "data/raw/xml/compteRendu/*.xml" "extracted_texts/project_full"
python3 analyze_project_ngrams.py \
  --speaker-dir "extracted_texts/project_full/by_speaker" \
  --out-dir "analysis_outputs/plain_project_content_stable" \
  --token-mode surface_content \
  --ngram-sizes 1 2 3 4 \
  --top-k 25 \
  --min-distinctive-count 50
python3 dashboard/build_dashboard_data.py
```

Then open `dashboard/index.html`.

Optional per-speaker distribution export:

```bash
python3 ngram_distribution.py "extracted_texts/project_full/by_speaker" \
  --out "analysis_outputs/ngram_distribution_project_full_surface.csv" \
  --ngram-sizes 1 2 3 4 \
  --top-k 20 \
  --token-mode surface
```

## Ignored Outputs

- `data/raw/`
- `extracted_texts/`
- `analysis_outputs/`
- `dashboard/data/`
