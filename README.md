# Predicting the Political Leaning of the Media Outlet from Article Text

A reproducible NLP framework for article-level political classification using classical machine learning, linguistic feature engineering, and semantic embeddings.

The project supports:
- article-level classification
- grouped/source-aware evaluation
- external full-text datasets
- reproducible NLP experimentation
- modular feature engineering pipelines

---

# Features

Implemented feature extraction methods include:

- TF-IDF lexical features
- TF-IDF + NMF topic representations
- Linguistic/POS features
- Readability metrics
- Sentiment analysis
- Sentence embeddings
- Chunked document embeddings

Implemented classifiers:
- Logistic Regression
- Linear SVC

---

# Repository Structure

```text
├── data/
│   ├── left/
│       ├── alter-net/
│          └── alternet_final.json
│       ├── counter-punch/
│           └── counterpunch_final.json
│       ├── daily-beast/
│           └── dailybeast_final.json
│       ├── mother-jones/
│           └── motherjones_final.json
│       └── salon/
│           └── salon_final.json
│   └── right/
│       ├── american-conservative/
│           └── americanconservative_final.json
│       ├── american-spectator/
│           └── americanspectator_final.json
│       ├── daily-caller/
│           └── dailycaller_final.json
│       ├── fox-news/
│           └── foxnews_final.json
│       └── the-free-beacon/
│           └── thefreebeacon_final.json
├── scrapers/
│   ├── left/
│       ├── alter-net/
│          └── alter-net-scraper.ipynb
│       ├── counter-punch/
│           └── counter-punch-scraper.ipynb
│       ├── daily-beast/
│           └── daily-beast-scraper.ipynb
│       ├── mother-jones/
│           └── mother-jones-scraper.ipynb
│       └── salon/
│           └── salon-scraper.ipynb
│   └── right/
│       ├── american-conservative/
│           └── american-conservative-scraper.ipynb
│       ├── american-spectator/
│           └── american-spectator-scraper.ipynb
│       ├── daily-caller/
│           └── daily-caller-scraper.ipynb
│       ├── fox-news/
│           └── scraper-fox-news.ipynb
│       └── the-free-beacon/
│           └── free-beacon-scraper.ipynb
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_article_classification.ipynb
│
├── outputs/
│   └── tables/
│
├── results/
│   └── tables/
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── cross_validation.py
│   ├── run_experiments.py
│   └── utils.py
│
├── main.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

# Data Collection Utilities

The repository also includes archival scraping notebooks used during the original dataset construction process.

These notebooks were used to collect article data from multiple news outlets across the political spectrum.

Example outlet categories include:
- left-leaning sources
- right-leaning sources

The scraping notebooks are preserved for:
- methodological transparency
- reproducibility documentation
- future dataset reconstruction
- educational purposes

These scrapers are not required for running the main experimentation pipeline.

Note:
- website structures may have changed since the original data collection
- some scrapers may no longer function without modification
- no copyrighted article bodies are redistributed in this repository

---

# Dataset Notes

The repository does not include copyrighted article bodies.

Included dataset files contain only non-copyrighted structural metadata such as:
- article IDs
- titles
- URLs
- publication dates
- source identifiers
- article statistics

These files are provided solely to preserve:
- dataset structure
- preprocessing compatibility
- experiment reproducibility
- repository organization

The full experimentation pipeline is designed to operate on externally supplied full-text datasets.

---

# External Full-Text Datasets

The framework is designed to support externally supplied full-text datasets.

The included examples use the AG News dataset as a publicly available demonstration corpus.

Other compatible datasets can also be used if they provide:
- article text
- classification labels
- optional source/group metadata

---

# Installation

## 1. Clone Repository

```bash
git clone <repository-url>
cd <repository-name>
```

---

## 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Download spaCy Model

```bash
python3 -m spacy download en_core_web_sm
```

---

# Running Experiments

## Run Main Pipeline

The primary experiment workflow can be executed using:

```bash
python3 -m main
```

Equivalent alternatives may include:

```bash
python -m main
```

or:

```bash
py -m main
```

depending on your Python environment.

---

# What `main.py` Does

The pipeline:
- loads and prepares the AG News dataset
- converts labels into a binary classification task
- performs train/test splitting
- runs feature extraction experiments
- evaluates classifiers using cross-validation
- performs final held-out testing
- exports experiment results

---

# Default Experiments

Enabled by default:

- TF-IDF
- TF-IDF + NMF

These configurations are lightweight and suitable for standard execution environments.

---

# Optional Advanced Experiments

The following feature pipelines are implemented but commented out by default:

- Full Linguistic Features
- Sentence Embeddings
- Chunked Embeddings

These experiments are computationally intensive and may significantly increase runtime.

They can be enabled manually inside:

```text
main.py
```

by uncommenting the corresponding experiment definitions.

---

# Running the Notebook Workflow

The notebook workflow is located in:

```text
notebooks/02_article_classification.ipynb
```

The notebook demonstrates:
- dataset loading
- preprocessing
- feature extraction
- model evaluation
- result visualization

---

# Outputs

Generated experiment results are saved to:

```text
outputs/tables/
```

Example outputs:
- `cv_summary.csv`
- `test_results.csv`

---

# Cross-Validation

The framework supports:
- standard stratified cross-validation
- grouped/source-aware cross-validation

Grouped evaluation helps reduce source leakage by ensuring articles from the same outlet are not shared between training and evaluation folds.

---

# Feature Engineering Overview

## TF-IDF

Sparse lexical unigram/bigram features.

---

## TF-IDF + NMF

Combines lexical TF-IDF features with topic representations extracted using Non-negative Matrix Factorization.

---

## Full Linguistic Features

Combines:
- POS tag distributions
- sentiment analysis
- readability metrics

---

## Sentence Embeddings

Dense semantic document embeddings generated using:

```text
all-MiniLM-L6-v2
```

---

## Chunked Embeddings

Long documents are divided into chunks and embedded separately before averaging.

---

# Reproducibility

The project uses:
- fixed random seeds
- deterministic train/test splitting
- centralized model configuration

to improve reproducibility across runs.

---

# Supported Use Cases

The framework is designed for:
- political article classification
- ideological framing analysis
- article-level text classification
- NLP experimentation
- feature engineering comparison

---

# Future Extensions

Potential future improvements include:
- multi-class classification
- transformer fine-tuning
- hierarchical classification
- multilingual support
- source-level domain adaptation
- embedding caching
- experiment tracking pipelines

---

# Author

- Edgar Kolian (https://github.com/kolianedgar)

---

# License

This repository is provided for research and educational purposes.

Users are responsible for ensuring compliance with:
- dataset licenses
- redistribution restrictions
- copyright requirements
when supplying external corpora.
