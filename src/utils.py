import numpy as np
import pandas as pd

def safe_counts(y):
    counts = np.bincount(y, minlength=2)
    return counts[0], counts[1]

def format_confusion_matrix(cm):

    return pd.DataFrame(
        cm,
        index=["Actual Left", "Actual Right"],
        columns=["Pred Left", "Pred Right"]
    )

def aggregate_confusion_matrices(cms):
    return np.sum(cms, axis=0)

def results_to_df(cv_folds, summaries, test_results):

    df_folds = pd.DataFrame(cv_folds)
    df_summary = pd.DataFrame(summaries)
    df_test = pd.DataFrame(test_results)

    return df_folds, df_summary, df_test

def print_summary(summary):
    print(f"\nModel: {summary['Model']}")
    print(f"LR:  {summary['LR Mean']:.3f} ± {summary['LR Std']:.3f}")
    print(f"SVC: {summary['SVC Mean']:.3f} ± {summary['SVC Std']:.3f}")