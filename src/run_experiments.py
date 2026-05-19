"""
Experiment runner utilities for political article classification.

This module coordinates:
- cross-validation experiments
- final held-out test evaluation
- result aggregation
- confusion matrix collection

The experiment runner is designed to support:
- standard article-level evaluation
- grouped/source-aware evaluation
- multiple feature extraction pipelines

Experiments are typically launched from:
    02_article_classification.ipynb
"""

from src.cross_validation import run_cv, run_test


def run_experiments(
    experiments,
    texts_train,
    y_train,
    texts_test,
    y_test,
    groups_train=None,
    grouped=False
):
    """
    Run all configured feature extraction experiments.

    For each experiment:
    - perform cross-validation
    - evaluate on held-out test data
    - collect confusion matrices
    - aggregate metrics

    Parameters
    ----------
    experiments : list
        List of tuples:
            (experiment_name, feature_builder)

    texts_train : array-like
        Training article texts.

    y_train : array-like
        Training labels.

    texts_test : array-like
        Held-out test article texts.

    y_test : array-like
        Held-out test labels.

    groups_train : array-like, optional
        Source/outlet group labels used for grouped CV.

    grouped : bool, default=False
        Whether to use grouped/source-aware cross-validation.

    Returns
    -------
    tuple
        (
            cv_all_folds,
            cv_summaries,
            test_results,
            all_conf_matrices
        )

    cv_all_folds : list
        Fold-level cross-validation results.

    cv_summaries : list
        Aggregate experiment summaries.

    test_results : list
        Final held-out test metrics.

    all_conf_matrices : dict
        Stored confusion matrices for all experiments.
    """

    cv_all_folds = []
    cv_summaries = []
    test_results = []

    all_conf_matrices = {}

    for name, builder in experiments:

        print("\n==============================")
        print(f"RUNNING EXPERIMENT: {name}")
        print("==============================")

        # ======================
        # CROSS-VALIDATION
        # ======================

        fold_res, summary, cms_lr, cms_svc = run_cv(
            builder=builder,
            name=name,
            texts=texts_train,
            y=y_train,
            groups=groups_train,
            grouped=grouped
        )

        cv_all_folds.extend(fold_res)

        cv_summaries.append(summary)

        # ======================
        # FINAL TEST EVALUATION
        # ======================

        test_metrics, cm_lr, cm_svc = run_test(
            builder=builder,
            name=name,
            X_train_text=texts_train,
            X_test_text=texts_test,
            y_train=y_train,
            y_test=y_test
        )

        test_results.append(test_metrics)

        # ======================
        # STORE CONFUSION MATRICES
        # ======================

        all_conf_matrices[name] = {

            "CV_LR": cms_lr,
            "CV_SVC": cms_svc,

            "TEST_LR": cm_lr,
            "TEST_SVC": cm_svc
        }

    return (
        cv_all_folds,
        cv_summaries,
        test_results,
        all_conf_matrices
    )