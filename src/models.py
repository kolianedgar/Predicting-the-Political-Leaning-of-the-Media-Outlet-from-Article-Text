"""
Model definitions for political article classification experiments.

This module centralizes all classifier configurations used
throughout cross-validation and final evaluation workflows.

Current models:
- Logistic Regression
- Linear Support Vector Classifier

All models use a shared random seed to improve reproducibility.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# =========================
# CONFIGURATION
# =========================

SEED = 42


# =========================
# MODEL FACTORIES
# =========================

def get_models():
    """
    Return initialized classification models.

    Returns
    -------
    dict
        Dictionary mapping model names to sklearn estimators.
    """

    return {

        "LR": LogisticRegression(
            max_iter=2000,
            C=1.0,
            random_state=SEED,
            n_jobs=-1
        ),

        "SVC": LinearSVC(
            C=0.5,
            max_iter=20000,
            random_state=SEED,
            dual=False
        )
    }


# =========================
# MODEL REGISTRY
# =========================

AVAILABLE_MODELS = [
    "LR",
    "SVC"
]