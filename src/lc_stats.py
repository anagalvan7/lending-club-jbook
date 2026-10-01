"""Comparacion estadistica de clasificadores sobre las MISMAS observaciones de prueba.

* `delong_fast`     : DeLong rapido (Sun y Xu, 2014), O(n log n), con covarianza entre las AUC.
* `delong_direct`   : version directa O(m*n), solo para VALIDAR la rapida con pocos datos.
* `holm`            : correccion de Holm por comparaciones multiples.
* `mcnemar_test`    : McNemar (con correccion de continuidad; exacta si b + c < 25).
* `paired_bootstrap`: bootstrap pareado (mismos indices para ambos modelos) de dAUC, dAUC-PR y dF1.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def _components(y, scores):
    """AUC y componentes estructurales V10 (positivos) y V01 (negativos) de cada modelo.

    `scores` es (k, N). Las filas se reordenan con los positivos primero.
    """
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")
    m = int(y.sum())
    n = len(y) - m
    preds = np.atleast_2d(np.asarray(scores, dtype=float))[:, order]
    tx = np.vstack([stats.rankdata(r[:m]) for r in preds])        # rangos entre positivos
    ty = np.vstack([stats.rankdata(r[m:]) for r in preds])        # rangos entre negativos
    tz = np.vstack([stats.rankdata(r) for r in preds])            # rangos entre todos
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    return aucs, v01, v10, m, n


def delong_fast(y, s1, s2, alpha: float = 0.05) -> dict:
    """DeLong pareado rapido. Devuelve AUC con IC, dAUC con IC, z y p bilateral."""
    aucs, v01, v10, m, n = _components(y, np.vstack([s1, s2]))
    cov = np.cov(v01) / m + np.cov(v10) / n
    var_d = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    delta = float(aucs[0] - aucs[1])
    se = float(np.sqrt(max(var_d, 0.0)))
    z = delta / se if se > 0 else (0.0 if delta == 0 else np.inf)
    q = stats.norm.ppf(1 - alpha / 2)
    se1, se2 = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]),
            "auc1_lo": float(max(aucs[0] - q * se1, 0)), "auc1_hi": float(min(aucs[0] + q * se1, 1)),
            "auc2_lo": float(max(aucs[1] - q * se2, 0)), "auc2_hi": float(min(aucs[1] + q * se2, 1)),
            "delta": delta, "delta_lo": delta - q * se, "delta_hi": delta + q * se, "se": se,
            "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z))),
            "corr_auc": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])) if cov[0, 0] * cov[1, 1] > 0 else float("nan")}


def delong_direct(y, s1, s2) -> dict:
    """DeLong 'de libro' con O(m*n) (solo para validar con pocos datos)."""
    y = np.asarray(y).astype(int)
    pos, neg = np.asarray([s1, s2], float)[:, y == 1], np.asarray([s1, s2], float)[:, y == 0]
    v10, v01 = [], []
    for k in range(2):
        d = pos[k][:, None] - neg[k][None, :]
        psi = (d > 0) + 0.5 * (d == 0)
        v10.append(psi.mean(axis=1))
        v01.append(psi.mean(axis=0))
    m, n = pos.shape[1], neg.shape[1]
    aucs = np.array([v.mean() for v in v10])
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    cov = s10 / m + s01 / n
    var_d = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    delta = float(aucs[0] - aucs[1])
    z = delta / np.sqrt(var_d)
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "delta": delta, "z": float(z),
            "p_value": float(2 * stats.norm.sf(abs(z)))}


def holm(pvals) -> np.ndarray:
    """Valores p ajustados por Holm (control de la tasa de error por familia)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def mcnemar_test(y, pred1, pred2) -> dict:
    """McNemar sobre aciertos/errores de dos clasificadores en las mismas observaciones."""
    y = np.asarray(y).astype(int)
    ok1, ok2 = np.asarray(pred1) == y, np.asarray(pred2) == y
    b = int(np.sum(ok1 & ~ok2))       # acierta el 1 y falla el 2
    c = int(np.sum(~ok1 & ok2))       # falla el 1 y acierta el 2
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0, "metodo": "sin discordantes"}
    if b + c < 25:
        p = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
        return {"b": b, "c": c, "chi2": float("nan"), "p_value": p, "metodo": "exacta (binomial)"}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "chi2": float(chi2), "p_value": float(stats.chi2.sf(chi2, 1)),
            "metodo": "chi-cuadrado con correccion de continuidad"}


def _boot_chunk(y, s1, s2, p1, p2, seeds):
    n = len(y)
    out = []
    for sd in seeds:
        w = np.bincount(np.random.default_rng(sd).integers(0, n, n), minlength=n)     # remuestreo con reemplazo
        out.append((roc_auc_score(y, s1, sample_weight=w) - roc_auc_score(y, s2, sample_weight=w),
                    average_precision_score(y, s1, sample_weight=w) - average_precision_score(y, s2, sample_weight=w),
                    f1_score(y, p1, sample_weight=w) - f1_score(y, p2, sample_weight=w)))
    return out


def paired_bootstrap(y, s1, s2, thr1, thr2, B: int = 2000, seed: int = 42, n_jobs: int = 6, alpha: float = 0.05) -> dict:
    """Bootstrap pareado: los MISMOS indices remuestreados para ambos modelos.

    El remuestreo con reemplazo se implementa con pesos enteros (conteos de cada fila), lo que es
    exactamente equivalente a remuestrear las filas. Devuelve la diferencia media, el IC percentil
    del 95 % y un valor p bilateral (2 * min(P(d* <= 0), P(d* >= 0))) de cada metrica.
    """
    y = np.asarray(y).astype(int)
    s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
    p1, p2 = (s1 >= thr1).astype(int), (s2 >= thr2).astype(int)
    seeds = np.random.SeedSequence(seed).generate_state(B)
    chunks = np.array_split(seeds, n_jobs * 4)
    res = Parallel(n_jobs=n_jobs)(delayed(_boot_chunk)(y, s1, s2, p1, p2, c) for c in chunks)
    d = np.array([r for ch in res for r in ch])                     # (B, 3)
    out = {}
    for k, nombre in enumerate(("auc", "auc_pr", "f1")):
        x = d[:, k]
        out[nombre] = {"media": float(x.mean()), "lo": float(np.percentile(x, 100 * alpha / 2)),
                       "hi": float(np.percentile(x, 100 * (1 - alpha / 2))),
                       "p_value": float(min(1.0, 2 * min((x <= 0).mean(), (x >= 0).mean()) + 1.0 / (B + 1))),
                       "obs": float([roc_auc_score(y, s1) - roc_auc_score(y, s2),
                                     average_precision_score(y, s1) - average_precision_score(y, s2),
                                     f1_score(y, p1) - f1_score(y, p2)][k])}
    out["B"] = B
    return out
