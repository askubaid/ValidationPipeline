# Validation Pipeline

This is a dataset validation pipeline for English–Portuguese parallel corpora. It is part of our semester project at the Institute of Space Technology, Islamabad. The team members are Ubaid ur Rehman, Hafsa Ayub, Hajra Shahid, and Iqra Younas. The project is supervised by Dr. Madiha Tahir.

This repository provides a two-layer validation workflow:

- **Layer 1** filters and categorizes sentence pairs using heuristics and language identification.
- **Layer 2** scores clean sentence pairs using COMET quality estimation and separates them into quality buckets.
- **Merge step** combines selected layer2 outputs into final `learner.en` and `learner.pt` files.

## Layer 1 checks

The first layer performs heuristic validation on aligned source and target sentences, including:

- `length_ratio` — flags sentence pairs with a target/source length ratio outside the expected bounds (0.6–1.5).
- `duplicates` — detects duplicate source sentences.
- `identical` — flags pairs where source and target are identical and longer than two words.
- `LangID` — checks that the source sentence is in the source language and the target sentence is in the target language using FastText.
- `markdown` — flags markdown or HTML-like formatting in the target sentence.
- `control_char` — detects control characters or replacement characters in the target sentence.

Clean sentence pairs are saved separately, and flagged examples are written to category-specific files in `layer1/`.

## Layer 2 processing

The second layer takes the clean sentence pairs from Layer 1 and scores them using COMET quality estimation:

- It loads a COMET-QE model (`Unbabel/wmt22-cometkiwi-da`).
- It scores each cleaned sentence pair and assigns a quality bucket.
- Sentence pairs are written to `*_excellent`, `*_good`, or `*_mediocre` files based on the score threshold.
  - `excellent`: score >= 0.8
  - `good`: 0.7 <= score < 0.8
  - `mediocre`: score < 0.7

## Repository structure

- `config.py` — dataset configuration and language settings.
- `layer1/m1_heuristics.py` — heuristic validation and cleanup.
- `layer2/m2_semantics.py` — semantic quality scoring with COMET.
- `Output/merge.py` — merge selected layer2 bundles into dataset files.
- `learner*.en` / `learner*.pt` — source and target bilingual files used by the pipeline.
- `layer1/` — outputs from the first filtering layer.
- `layer2/` — outputs from the second semantic scoring layer.

## Prerequisites

- Python 3.8+
- Internet access for model downloads:
  - FastText language identification model (`lid.176.bin`)
  - COMET-QE model from Hugging Face

## Installation

1. Create and activate a Python virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Set the source and target filenames in `config.py` if needed.
2. Run Layer 1 validation:

```bash
python layer1\m1_heuristics.py
```

This generates:

- `layer1/learner_single____clean.en`
- `layer1/learner_single____clean.pt`
- `layer1/layer1_results.csv`
- category-specific flagged files in `layer1/`

3. Run Layer 2 semantic scoring:

```bash
python layer2\m2_semantics.py
```

This generates quality buckets in `layer2/`:

- `*_excellent.en` / `*_excellent.pt`
- `*_good.en` / `*_good.pt`
- `*_mediocre.en` / `*_mediocre.pt`

4. Merge selected layer2 outputs:

```bash
python Output\merge.py
```

The merge script writes to `learner.en` and `learner.pt` by default.

## Notes

- `layer1/m1_heuristics.py` downloads the FastText language ID model automatically if missing.
- `layer2/m2_semantics.py` downloads the COMET model automatically if not already present in the local cache.
- The pipeline assumes aligned source/target sentence pairs with one sentence per line.

## Customization

- Edit `config.py` to change `SRC_FILE_NAME`, `TGT_FILE_NAME`, `SRC_LANG`, and `TGT_LANG`.
- Adjust the merge file names in `Output/merge.py` to change which quality buckets are combined.

## License
- This project is open-source. A gift from the Institute of Space Technology, Islamabad. Feel free to use, modify, and distribute.

## Contact & Support
- For issues, questions, or contributions, please get in touch: askubaid@gmail.com 