"""Los seis modelos comparados y sus espacios de busqueda, en scikit-learn y en PySpark.

Se definen UNA vez aqui para que los dos entornos usen espacios de busqueda equivalentes.
Claves cortas: LR, DT, RF, GB, SVM, NB.
"""
from __future__ import annotations

import numpy as np

from . import lc_utils as lc

CLAVES = ["LR", "DT", "RF", "GB", "SVM", "NB"]
NOMBRES = {"LR": "Regresión logística", "DT": "Árbol de decisión", "RF": "Bosque aleatorio",
           "GB": "Gradient boosting", "SVM": "SVM lineal", "NB": "Naive Bayes"}
REG_PARAMS = [1e-6, 1e-5, 1e-4]           # regParam de PySpark (LR y LinearSVC)
DEPTHS = [5, 10, 15]
RF_TREES = [10, 50, 100]
GB_ITERS = [50, 100]
GB_DEPTHS = [3, 5]
USA_ESCALADO = {"LR": True, "SVM": True}   # el resto trabaja con las variables sin escalar


def c_from_reg(reg_param: float, n_train: int) -> float:
    """Equivalencia aproximada entre C (scikit-learn) y regParam (Spark): C = 1 / (regParam * n)."""
    return 1.0 / (reg_param * n_train)


def sk_specs(n_train: int, gb_hist: bool = False) -> dict:
    """{clave: (estimador, malla, tipo_de_puntuacion)}. `score`: 'proba' o 'decision'."""
    from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier

    cs = [c_from_reg(r, n_train) for r in REG_PARAMS]
    gb = (HistGradientBoostingClassifier(learning_rate=0.1, random_state=lc.SEED),
          {"max_iter": GB_ITERS, "max_depth": GB_DEPTHS}) if gb_hist else \
         (GradientBoostingClassifier(learning_rate=0.1, random_state=lc.SEED),
          {"n_estimators": GB_ITERS, "max_depth": GB_DEPTHS})
    return {
        "LR": (LogisticRegression(penalty="l2", max_iter=100, random_state=lc.SEED), {"C": cs}, "proba"),
        "DT": (DecisionTreeClassifier(random_state=lc.SEED), {"max_depth": DEPTHS}, "proba"),
        "RF": (RandomForestClassifier(random_state=lc.SEED, n_jobs=1),
               {"n_estimators": RF_TREES, "max_depth": DEPTHS}, "proba"),
        "GB": (*gb, "proba"),
        "SVM": (LinearSVC(loss="hinge", penalty="l2", random_state=lc.SEED), {"C": cs}, "decision"),
        "NB": (GaussianNB(), {}, "proba"),
    }


def score_sklearn(est, X, tipo: str) -> np.ndarray:
    """Puntuacion continua de la clase positiva (probabilidad o valor de decision)."""
    return est.predict_proba(X)[:, 1] if tipo == "proba" else est.decision_function(X)
