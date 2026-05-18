"""
Cross-validation and evaluation utilities for political article classification.

This module provides:
- standard stratified cross-validation
- grouped/source-aware cross-validation
- final held-out test evaluation
- confusion matrix collection
- fold-level metric tracking

Grouped cross-validation is used to reduce outlet/source leakage
by ensuring articles from the same source are not shared across
training and evaluation folds.

Supported classifiers are defined centrally in:
    src.models.get_models()
"""

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from src.models import get_models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from src.utils import safe_counts

def run_cv(
        builder,
        name,
        texts,
        y,
        groups=None,
        grouped=False
    ):

    """
    Run cross-validation experiments using the provided feature builder.

    Supports:
    - standard stratified cross-validation
    - grouped/source-aware cross-validation

    Parameters
    ----------
    builder : callable
        Feature builder function that returns:
            (X_train, X_test)

    name : str
        Name of the feature configuration or experiment.

    texts : array-like
        Input article texts.

    y : array-like
        Binary classification labels.

    groups : array-like, optional
        Source/outlet group labels used for grouped evaluation.

    grouped : bool, default=False
        Whether to use source-aware grouped cross-validation.

    Returns
    -------
    tuple
        (
            fold_results,
            summary,
            cms_lr,
            cms_svc
        )

    fold_results : list
        Per-fold evaluation statistics.

    summary : dict
        Aggregate mean/std accuracy metrics.

    cms_lr : list
        Logistic Regression confusion matrices.

    cms_svc : list
        Linear SVC confusion matrices.
    """

    if grouped:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        split_iter = splitter.split(texts, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split_iter = splitter.split(texts, y)

    fold_results = []
    cms_lr, cms_svc = [], []

    print(f"\n===== CV: {name} =====")

    for fold, (train_idx, test_idx) in enumerate(split_iter):

        X_train_text = texts[train_idx]
        X_test_text  = texts[test_idx]

        y_tr = y[train_idx]
        y_te = y[test_idx]

        # --- features ---
        X_train, X_test = builder(X_train_text, X_test_text)

        # --- models ---
        models = get_models()
        lr = models["LR"]
        svc = models["SVC"]

        lr.fit(X_train, y_tr)
        svc.fit(X_train, y_tr)

        y_pred_lr = lr.predict(X_test)
        y_pred_svc = svc.predict(X_test)

        # --- metrics ---
        acc_lr = accuracy_score(y_te, y_pred_lr)
        acc_svc = accuracy_score(y_te, y_pred_svc)

        cm_lr = confusion_matrix(y_te, y_pred_lr, labels=[0, 1])
        cm_svc = confusion_matrix(y_te, y_pred_svc, labels=[0, 1])

        cms_lr.append(cm_lr)
        cms_svc.append(cm_svc)

        train_left, train_right = safe_counts(y_tr)
        test_left, test_right = safe_counts(y_te)

        result = {
            "Model": name,
            "Fold": fold + 1,
            "LR Accuracy": acc_lr,
            "SVC Accuracy": acc_svc,
            "Train Size": len(train_idx),
            "Test Size": len(test_idx),
            "Train Left": train_left,
            "Train Right": train_right,
            "Test Left": test_left,
            "Test Right": test_right
        }

        if grouped:
            result["Train Sources"] = ", ".join(np.unique(groups[train_idx]))
            result["Test Sources"] = ", ".join(np.unique(groups[test_idx]))

        fold_results.append(result)

        print(f"Fold {fold+1}: LR={acc_lr:.3f}, SVC={acc_svc:.3f}")

    # --- aggregate ---
    lr_scores = [f["LR Accuracy"] for f in fold_results]
    svc_scores = [f["SVC Accuracy"] for f in fold_results]

    summary = {
        "Model": name,
        "LR Mean": np.mean(lr_scores),
        "LR Std": np.std(lr_scores),
        "SVC Mean": np.mean(svc_scores),
        "SVC Std": np.std(svc_scores)
    }

    return fold_results, summary, cms_lr, cms_svc

def run_test(
        builder,
        name,
        X_train_text,
        X_test_text,
        y_train,
        y_test
    ):

    """
    Run final held-out test evaluation.

    This function:
    - builds train/test features
    - trains all configured models
    - evaluates final test performance
    - prints classification reports
    - returns confusion matrices and metrics

    Parameters
    ----------
    builder : callable
        Feature builder function.

    name : str
        Experiment or feature configuration name.

    X_train_text : array-like
        Training article texts.

    X_test_text : array-like
        Test article texts.

    y_train : array-like
        Training labels.

    y_test : array-like
        Test labels.

    Returns
    -------
    tuple
        (
            metrics,
            cm_lr,
            cm_svc
        )

    metrics : dict
        Final test accuracy metrics.

    cm_lr : numpy.ndarray
        Logistic Regression confusion matrix.

    cm_svc : numpy.ndarray
        Linear SVC confusion matrix.
    """

    print(f"\n===== FINAL TEST: {name} =====")

    X_train, X_test = builder(X_train_text, X_test_text)

    models = get_models()
    lr = models["LR"]
    svc = models["SVC"]

    lr.fit(X_train, y_train)
    svc.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    y_pred_svc = svc.predict(X_test)

    cm_lr = confusion_matrix(y_test, y_pred_lr, labels=[0, 1])
    cm_svc = confusion_matrix(y_test, y_pred_svc, labels=[0, 1])

    print("\nLR:")
    print(classification_report(y_test, y_pred_lr))

    print("\nSVC:")
    print(classification_report(y_test, y_pred_svc))

    return {
        "Model": name,
        "LR Test Acc": accuracy_score(y_test, y_pred_lr),
        "SVC Test Acc": accuracy_score(y_test, y_pred_svc)
    }, cm_lr, cm_svc