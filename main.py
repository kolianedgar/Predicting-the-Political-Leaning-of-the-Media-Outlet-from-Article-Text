# main.py

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.features import (
    build_tfidf,
    build_tfidf_nmf,
    build_full,
    build_embeddings,
    build_chunked
)

from src.experiments import run_experiments
from src.utils import results_to_df


# ======================
# LOAD + PREPARE AG NEWS
# ======================

def load_agnews_binary():

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
        ("Full Linguistic", build_full),
        ("Embeddings", build_embeddings),
        ("Chunked Embeddings", build_chunked),
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

    df_summary.to_csv("outputs/tables/cv_summary.csv", index=False)
    df_test.to_csv("outputs/tables/test_results.csv", index=False)

    print("\nDone. Results saved in outputs/")


if __name__ == "__main__":
    main()