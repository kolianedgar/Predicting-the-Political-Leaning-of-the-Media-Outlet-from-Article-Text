"""
Main experiment execution script for article-level classification.

This script serves as the primary entry point for running
reproducible political article classification experiments.

The pipeline:
- loads and prepares external full-text datasets
- performs train/test splitting
- executes multiple feature extraction pipelines
- evaluates classical machine learning classifiers
- aggregates cross-validation and test metrics
- exports experiment results

Implemented feature configurations include:
- TF-IDF lexical features
- TF-IDF + NMF topic features
- linguistic/readability features
- sentence embeddings
- chunked document embeddings

To reduce runtime overhead and improve accessibility
on limited hardware, computationally intensive
feature pipelines are commented out by default.
These advanced experiments remain fully implemented
and can be enabled for extended benchmarking.

The default workflow uses the AG News dataset as a
publicly available example corpus for demonstrating
the full NLP experimentation pipeline.

Exported outputs include:
- cross-validation summaries
- held-out test metrics

Results are saved to:
    outputs/tables/

This script is intended as a lightweight,
reproducible baseline workflow for:
- article-level classification
- feature engineering comparison
- external dataset experimentation
- future extensibility
"""

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.features import (
    build_tfidf,
    build_tfidf_nmf,
    build_full,
    build_embeddings,
    build_chunked
)

from src.run_experiments import run_experiments
from src.utils import results_to_df


# ======================
# LOAD + PREPARE AG NEWS
# ======================

def load_agnews_binary():
    """
    Load and prepare the AG News dataset for
    binary article classification experiments.

    The original AG News dataset contains four classes:
    - 0: World
    - 1: Sports
    - 2: Business
    - 3: Sci/Tech

    For simplified binary classification:
    - World + Sports are mapped to class 0
    - Business + Sci/Tech are mapped to class 1

    The returned dataframe is standardized to match
    the project's preprocessing and experimentation
    pipeline requirements.

    Returns
    -------
    pandas.DataFrame
        Standardized dataframe containing:
        - text : article text
        - label : binary class label
        - source : dataset/source identifier
    """
    dataset = load_dataset("ag_news")

    df = dataset["train"].to_pandas()

    # Convert to binary:
    # 0 = World + Sports
    # 1 = Business + Sci/Tech
    df["label"] = df["label"].apply(lambda x: 0 if x < 2 else 1)

    # Keep only required columns
    df = df.rename(columns={"text": "text"})
    df["source"] = "AGNews"

    return df


# ======================
# MAIN PIPELINE
# ======================

def main():
    """
    Execute the complete article-level classification pipeline.

    The workflow performs:
    - dataset loading and preparation
    - stratified train/test splitting
    - feature extraction experiment configuration
    - cross-validation evaluation
    - held-out test evaluation
    - experiment result aggregation
    - CSV result export

    The pipeline evaluates multiple feature engineering
    strategies using classical machine learning models,
    including:
    - Logistic Regression
    - Linear SVC

    By default, computationally intensive experiments
    (e.g. transformer embeddings and linguistic pipelines)
    are commented out to avoid excessive runtime overhead
    during standard execution.

    Outputs
    -------
    CSV result tables saved to:
        outputs/tables/

    Generated files:
    - cv_summary.csv
    - test_results.csv
    """

    # ======================
    # LOAD DATA
    # ======================
    df = load_agnews_binary()

    texts = df["text"].values
    y = df["label"].values
    groups = df["source"].values  # single group (not used)

    # ======================
    # TRAIN / TEST SPLIT
    # ======================
    texts_train, texts_test, y_train, y_test, groups_train, _ = train_test_split(
        texts,
        y,
        groups,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ======================
    # DEFINE EXPERIMENTS
    # ======================
    experiments = [
        ("TF-IDF", build_tfidf),
        ("TF-IDF + NMF", build_tfidf_nmf),
        # ("Full Linguistic", build_full),
        # ("Embeddings", build_embeddings),
        # ("Chunked Embeddings", build_chunked),
    ]

    # ======================
    # RUN EXPERIMENTS
    # ======================
    cv_folds, cv_summary, test_res, cms = run_experiments(
        experiments,
        texts_train,
        y_train,
        texts_test,
        y_test,
        grouped=False
    )

    # ======================
    # SAVE RESULTS
    # ======================
    df_folds, df_summary, df_test = results_to_df(
        cv_folds,
        cv_summary,
        test_res
    )

    df_summary.to_csv("results/tables/cv_summary.csv", index=False)
    df_test.to_csv("results/tables/test_results.csv", index=False)

    print("\nDone. Results saved in results/")


if __name__ == "__main__":
    main()