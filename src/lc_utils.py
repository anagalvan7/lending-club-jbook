"""Utilidades del estudio (Lending Club): carga, variable objetivo, features y metricas.

Principio central: solo se usan variables que EXISTEN al momento de otorgar el prestamo.
Columnas como `total_pymnt`, `recoveries` o `last_pymnt_d` se generan DESPUES del desenlace
(pagado / incumplido) y, si se usaran, el modelo "adivinaria" el objetivo (fuga de datos).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

SEED = 42
# Ruta del CSV: variable de entorno LC_DATA o data/accepted_2007_to_2018Q4.csv (ver README).
DATA_PATH = Path(os.environ.get("LC_DATA", Path(__file__).resolve().parents[1] / "data"
                                / "accepted_2007_to_2018Q4.csv"))

RESOLVED = ["Fully Paid", "Charged Off"]

NUMERIC = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "fico_range_high",
           "emp_length", "open_acc", "revol_bal", "revol_util", "total_acc", "pub_rec",
           "delinq_2yrs", "inq_last_6mths", "mort_acc", "credit_history_years", "term_months"]
CATEGORICAL = ["purpose", "home_ownership", "addr_state", "verification_status"]
FEATURES = NUMERIC + CATEGORICAL

# Columnas crudas que hay que leer del CSV para construir las features.
RAW_COLUMNS = ["id", "loan_status", "issue_d", "term", "loan_amnt", "int_rate", "installment",
               "annual_inc", "dti", "fico_range_high", "emp_length", "open_acc", "revol_bal",
               "revol_util", "total_acc", "pub_rec", "delinq_2yrs", "inq_last_6mths", "mort_acc",
               "earliest_cr_line"] + CATEGORICAL

# Columnas generadas DESPUES de otorgar el prestamo: NO se usan como predictores.
POST_ORIGINATION = ["out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
                    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
                    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
                    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low"]

MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _year_month(s: pd.Series) -> pd.Series:
    """'Aug-2003' -> 2003 + 7/12 (fraccion de anio). Independiente del idioma del sistema."""
    year = pd.to_numeric(s.str[-4:], errors="coerce")
    month = s.str[:3].map(MONTHS)
    return year + (month - 1) / 12.0


def parse_emp_length(s: pd.Series) -> pd.Series:
    """'10+ years' -> 10 ; '< 1 year' -> 0 ; '3 years' -> 3 ; vacio -> NaN."""
    num = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
    return num.mask(s.str.startswith("<", na=False), 0.0)


def load_raw(usecols=None, path: Path = DATA_PATH, **kw) -> pd.DataFrame:
    """Lee el CSV COMPLETO (sin muestreo). `usecols` solo elige columnas, no filas."""
    return pd.read_csv(path, usecols=usecols, low_memory=False, **kw)


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prestamos ya resueltos + variable objetivo + features. No aprende nada de los datos."""
    df = df[df["loan_status"].isin(RESOLVED)].copy()
    out = pd.DataFrame({"id": df["id"].astype(str)})
    out["default"] = (df["loan_status"] == "Charged Off").astype("int8")
    out["issue_year"] = pd.to_numeric(df["issue_d"].str[-4:], errors="coerce").astype("int16")
    out["term_months"] = pd.to_numeric(df["term"].str.extract(r"(\d+)")[0], errors="coerce")
    out["emp_length"] = parse_emp_length(df["emp_length"])
    out["credit_history_years"] = _year_month(df["issue_d"]) - _year_month(df["earliest_cr_line"])
    for c in ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "fico_range_high",
              "open_acc", "revol_bal", "revol_util", "total_acc", "pub_rec", "delinq_2yrs",
              "inq_last_6mths", "mort_acc"]:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL:
        out[c] = df[c].fillna("Desconocido").astype(str)
    return out.reset_index(drop=True)


def make_split(m: pd.DataFrame, test_size: float = 0.2, seed: int = SEED, n_folds: int = 3) -> pd.DataFrame:
    """Particion COMUN 80/20 estratificada por `default` + pliegues fijos de validacion cruzada.

    Devuelve un DataFrame con:
      * `id`     : identificador unico del prestamo
      * `split`  : 'train' o 'test'  (la asignacion que leen scikit-learn Y PySpark)
      * `fold`   : pliegue 0..n_folds-1 de las filas de entrenamiento (estratificado por `default`); -1 en test.
                   scikit-learn y el CrossValidator de Spark (foldCol) usan LOS MISMOS pliegues.
      * `is_test`: version booleana de `split`
      * `rank`   : orden aleatorio fijo de las filas de entrenamiento (0..n_train-1); sirve para los
                   subconjuntos anidados del experimento de escala (mismo orden en ambos motores).
    Es deterministica: llamarla dos veces con los mismos datos da exactamente lo mismo.
    """
    from sklearn.model_selection import StratifiedKFold

    idx = np.arange(len(m))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, stratify=m["default"],
                                           random_state=seed)
    rank = np.full(len(m), -1, dtype=np.int64)
    rank[idx_train] = np.random.default_rng(seed).permutation(len(idx_train))
    fold = np.full(len(m), -1, dtype=np.int64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf.split(idx_train, m["default"].to_numpy()[idx_train])):
        fold[idx_train[va]] = k
    split = pd.DataFrame({"id": m["id"].to_numpy(), "split": "train", "fold": fold, "is_test": False, "rank": rank})
    split.loc[idx_test, "split"] = "test"
    split.loc[idx_test, "is_test"] = True
    return split


def make_preprocessor(min_frequency: float = 0.01) -> ColumnTransformer:
    """Preprocesamiento para arboles. Sin escalado (los arboles no lo necesitan).

    * Numericas: los faltantes se rellenan con la constante -1 (las variables validas son >= 0;
      la unica excepcion observada es dti = -1, que tambien es un dato invalido y por lo tanto
      se trata igual que un faltante). Es una transformacion SIN parametros aprendidos => no
      puede filtrar informacion de validacion hacia entrenamiento, ni siquiera si se ajusta
      antes de la validacion cruzada (que es lo que ocurre al usar GridSearchCV sin Pipeline).
    * Categoricas: One-Hot; las categorias con menos del 1 % del entrenamiento se agrupan
      en 'infrecuente'. Categorias nuevas en test tambien caen en 'infrecuente'.
    """
    return ColumnTransformer(
        [("num", SimpleImputer(strategy="constant", fill_value=-1.0), NUMERIC),
         ("cat", OneHotEncoder(sparse_output=False, dtype=np.float32, min_frequency=min_frequency,
                               handle_unknown="infrequent_if_exist"), CATEGORICAL)],
        verbose_feature_names_out=False)


# ---------------------------------------------------------------- metricas

def threshold_free(y, p) -> dict:
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p)}


def threshold_metrics(y, p, thr: float) -> dict:
    pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"umbral": float(thr), "accuracy": accuracy_score(y, pred),
            "balanced_acc": balanced_accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "especificidad": tn / (tn + fp), "f1": f1_score(y, pred, zero_division=0),
            "mcc": matthews_corrcoef(y, pred), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "TP": int(tp)}


def best_f1_threshold(y, p, n_grid: int = 199) -> float:
    """Umbral que maximiza F1 sobre una malla fina (se usa con puntuaciones fuera de pliegue).

    Para probabilidades la malla va de 0.02 a 0.98; para valores de decision (p. ej. LinearSVC),
    que no estan en [0, 1], la malla son cuantiles de las propias puntuaciones.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if p.min() >= 0.0 and p.max() <= 1.0:
        grid = np.linspace(0.02, 0.98, n_grid)
    else:
        grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, n_grid)))
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tp_cum = np.cumsum(ys)
    fp_cum = np.cumsum(1 - ys)
    pos = ys.sum()
    idx = np.searchsorted(-ps, -grid, side="right")  # cuantos casos tienen p >= umbral
    tp = np.where(idx > 0, tp_cum[np.maximum(idx - 1, 0)], 0)
    fp = np.where(idx > 0, fp_cum[np.maximum(idx - 1, 0)], 0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + (pos - tp), 1)
    return float(grid[int(np.argmax(f1))])


def bootstrap_auc_ci(y, p, n_boot: int = 200, alpha: float = 0.05, seed: int = SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    vals = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def delong_paired(y_true, proba1, proba2) -> dict:
    """DeLong PAREADO (con covarianza) para dos AUC medidas sobre las MISMAS observaciones."""
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-y, kind="stable")
    m = int(y.sum())
    n = len(y) - m
    preds = np.vstack([np.asarray(proba1)[order], np.asarray(proba2)[order]])
    tx = np.vstack([stats.rankdata(r[:m]) for r in preds])
    ty = np.vstack([stats.rankdata(r[m:]) for r in preds])
    tz = np.vstack([stats.rankdata(r) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    delta = float(aucs[0] - aucs[1])
    z = delta / np.sqrt(var) if var > 0 else (0.0 if delta == 0 else np.inf)
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "delta": delta, "z": float(z),
            "p_value": float(2 * stats.norm.sf(abs(z))),
            "corr_auc": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))}


class LimeAdapter:
    """Conecta LIME con CUALQUIER modelo que reciba un DataFrame de variables crudas.

    LIME perturba la instancia en el espacio de las variables ORIGINALES (no en el espacio
    One-Hot), por eso las categoricas se codifican como enteros y, antes de llamar al modelo, se
    decodifican a texto. `predict_df` es una funcion  DataFrame -> probabilidades de la clase 1;
    puede envolver un modelo de scikit-learn o de Spark (ver capitulos 4 y 3).
    """

    def __init__(self, ref: pd.DataFrame, predict_df, extra: pd.DataFrame | None = None):
        from lime.lime_tabular import LimeTabularExplainer

        base = pd.concat([ref[FEATURES]] + ([extra[FEATURES]] if extra is not None else []))
        self.levels = {c: sorted(base[c].astype(str).unique()) for c in CATEGORICAL}
        self.predict_df = predict_df
        self.cat_idx = [FEATURES.index(c) for c in CATEGORICAL]
        self.explainer = LimeTabularExplainer(
            self.encode(ref), feature_names=FEATURES, categorical_features=self.cat_idx,
            categorical_names={FEATURES.index(c): self.levels[c] for c in CATEGORICAL},
            class_names=["pagado", "incumplido"], discretize_continuous=True, mode="classification",
            random_state=SEED)

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        out = df[FEATURES].copy()
        for c in NUMERIC:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(-1.0)      # mismo centinela que el modelo
        for c in CATEGORICAL:
            out[c] = out[c].astype(str).map({v: i for i, v in enumerate(self.levels[c])})
        return out.to_numpy(dtype=float)

    def decode(self, X: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(X, columns=FEATURES)
        for c in CATEGORICAL:
            lv = np.array(self.levels[c], dtype=object)
            df[c] = lv[np.clip(np.rint(df[c].to_numpy()).astype(int), 0, len(lv) - 1)]
        return df

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = np.asarray(self.predict_df(self.decode(X)), dtype=float)
        return np.column_stack([1.0 - p1, p1])

    def explain(self, row: pd.DataFrame, num_features: int = 10, num_samples: int = 5000):
        return self.explainer.explain_instance(self.encode(row)[0], self.predict_proba, labels=(1,),
                                               num_features=num_features, num_samples=num_samples)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """V de Cramer (0 = sin asociacion, 1 = asociacion perfecta), con correccion de sesgo."""
    tab = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    r, k = tab.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(min(kc - 1, rc - 1), 1e-12)))
