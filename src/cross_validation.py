from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from src.models import get_models
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from src.utils import safe_counts

def run_cv(builder, name, texts, y, groups=None, grouped=False):

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

from sklearn.metrics import classification_report

def run_test(builder, name, X_train_text, X_test_text, y_train, y_test):

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