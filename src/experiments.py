from src.cross_validation import *

def run_experiments(
    experiments,
    texts_train,
    y_train,
    texts_test,
    y_test,
    groups_train=None,
    grouped=False
):

    cv_all_folds = []
    cv_summaries = []
    test_results = []
    all_conf_matrices = {}

    for name, builder in experiments:

        print(f"\n==============================")
        print(f"RUNNING: {name}")
        print(f"==============================")

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
        # FINAL TEST
        # ======================
        test_res = run_test(
            builder=builder,
            name=name,
            X_train_text=texts_train,
            X_test_text=texts_test,
            y_train=y_train,
            y_test=y_test
        )

        test_results.append(test_res)

        # ======================
        # STORE CONFUSION MATRICES
        # ======================
        all_conf_matrices[name] = {
            "CV_LR": cms_lr,
            "CV_SVC": cms_svc
        }

    return cv_all_folds, cv_summaries, test_results, all_conf_matrices