"""Los seis modelos de PySpark: espacios de busqueda, entrenamiento con CrossValidator y evaluacion.

Todo lo que toca el driver ocurre DESPUES del entrenamiento y son solo las columnas id, default y la
puntuacion del conjunto de prueba; el tiempo de esa transferencia se mide por separado.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import lc_models as lm
from . import lc_utils as lc

MAXMEM = 64            # maxMemoryInMB de los arboles (ver capitulo 3: limita la memoria de histogramas por pasada)
MAXBINS = 32           # valor por defecto de Spark: discretiza cada variable continua en 32 intervalos
FOLDS = 3


def spark_specs(maxmem: int = MAXMEM) -> dict:
    """{clave: (estimador, malla, columna de puntuacion)}; `score_col` es 'probability' o 'rawPrediction'."""
    from pyspark.ml.classification import (DecisionTreeClassifier, GBTClassifier, LinearSVC, LogisticRegression,
                                           NaiveBayes, RandomForestClassifier)
    from pyspark.ml.tuning import ParamGridBuilder

    kw = dict(labelCol="default", seed=lc.SEED)
    lr = LogisticRegression(labelCol="default", featuresCol="features_sc", elasticNetParam=0.0, maxIter=100,
                            standardization=False)
    dt = DecisionTreeClassifier(featuresCol="features", maxBins=MAXBINS, maxMemoryInMB=maxmem, **kw)
    rf = RandomForestClassifier(featuresCol="features", maxBins=MAXBINS, maxMemoryInMB=maxmem, **kw)
    gb = GBTClassifier(featuresCol="features", stepSize=0.1, maxBins=MAXBINS, maxMemoryInMB=maxmem, **kw)
    svm = LinearSVC(labelCol="default", featuresCol="features_sc", maxIter=100, standardization=False)
    nb = NaiveBayes(labelCol="default", featuresCol="features", modelType="gaussian")
    return {
        "LR": (lr, ParamGridBuilder().addGrid(lr.regParam, lm.REG_PARAMS).build(), "probability"),
        "DT": (dt, ParamGridBuilder().addGrid(dt.maxDepth, lm.DEPTHS).build(), "probability"),
        "RF": (rf, ParamGridBuilder().addGrid(rf.numTrees, lm.RF_TREES).addGrid(rf.maxDepth, lm.DEPTHS).build(), "probability"),
        "GB": (gb, ParamGridBuilder().addGrid(gb.maxIter, lm.GB_ITERS).addGrid(gb.maxDepth, lm.GB_DEPTHS).build(), "probability"),
        "SVM": (svm, ParamGridBuilder().addGrid(svm.regParam, lm.REG_PARAMS).build(), "rawPrediction"),
        "NB": (nb, ParamGridBuilder().build(), "probability"),     # sin busqueda: una sola combinacion (la de por defecto)
    }


def score_column(score_col: str):
    """Puntuacion de la clase positiva (segundo elemento del vector), con vector_to_array."""
    from pyspark.ml.functions import vector_to_array
    return vector_to_array(score_col)[1].alias("score")


def pred_con(modelo, df, n: int = 3):
    """Aplica un modelo con pocas tareas simultaneas: Spark copia el modelo a cada tarea (un bosque de 100
    arboles y profundidad 15 ocupa > 1 GB por copia) y con ~12 tareas a la vez agotaba el heap de 8 GB."""
    return modelo.transform(df.coalesce(n))


def _param_names(grid) -> list[str]:
    return sorted({p.name for pm in grid for p in pm})


def _best_params(model, names: list[str]) -> dict:
    out = {}
    for n in names:
        v = model.getOrDefault(n) if n != "numTrees" else model.getNumTrees
        out[n] = float(v) if isinstance(v, float) else int(v)
    return out


def oof_threshold(estimator, train_f, score_col: str, folds: int = FOLDS) -> tuple[float, float]:
    """Umbral que maximiza F1 con puntuaciones FUERA DE PLIEGUE del entrenamiento.

    Para cada pliegue se ajusta el estimador (ya con los mejores hiperparametros) sin ese pliegue y se
    puntua el pliegue que no vio. Los verdaderos/falsos positivos para una malla de umbrales se
    calculan con agregados de Spark y se SUMAN entre pliegues: no se trae ninguna fila al driver.
    Es un bucle sobre pliegues (no una busqueda de hiperparametros). Devuelve (umbral, F1 OOF).
    """
    from pyspark.sql import functions as F

    grilla, tot = None, {}
    for i in range(folds):
        modelo_i = estimator.fit(train_f.filter(F.col("fold") != i))
        pred_i = pred_con(modelo_i, train_f.filter(F.col("fold") == i)).select("default", score_column(score_col))
        if grilla is None:
            if score_col == "probability":
                grilla = [round(float(x), 4) for x in np.linspace(0.02, 0.98, 99)]
            else:                                                   # valores de decision: cuantiles de las puntuaciones
                grilla = sorted(set(pred_i.approxQuantile("score", [float(q) for q in np.linspace(0.02, 0.98, 99)], 1e-3)))
            aggs = []
            for j, t in enumerate(grilla):
                pos_pred = F.col("score") >= t
                aggs += [F.sum(F.when(pos_pred & (F.col("default") == 1), 1).otherwise(0)).alias(f"tp_{j}"),
                         F.sum(F.when(pos_pred & (F.col("default") == 0), 1).otherwise(0)).alias(f"fp_{j}")]
        fila = pred_i.agg(F.sum("default").alias("pos"), *aggs).first().asDict()
        for k, v in fila.items():
            tot[k] = tot.get(k, 0) + (v or 0)
        del modelo_i, pred_i
        gc.collect()
    pos = tot["pos"]
    f1 = np.array([2 * tot[f"tp_{j}"] / max(2 * tot[f"tp_{j}"] + tot[f"fp_{j}"] + (pos - tot[f"tp_{j}"]), 1)
                   for j in range(len(grilla))])
    return float(grilla[int(np.argmax(f1))]), float(f1.max())


def run_model(spark, clave: str, train_f, test_f, out_dir: Path, maxmem: int = MAXMEM, keep_model: bool = False,
              verbose: bool = True):
    """Entrena UN modelo con ParamGridBuilder + CrossValidator (3 pliegues fijos, foldCol) y lo evalua.

    Guarda `sp_modelo_<clave>.json`, `sp_cv_<clave>.csv` y `sp_scores_test_<clave>.parquet` en `out_dir`.
    Devuelve (dict de resultados, modelo si keep_model). Si el JSON ya existe, lo carga (reanudacion).
    """
    from pyspark import StorageLevel
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator
    from pyspark.sql import functions as F

    f_json = out_dir / f"sp_modelo_{clave}.json"
    if f_json.exists():
        if verbose:
            print(f"[{clave}] resultados previos encontrados: se cargan (no se reentrena)")
        return json.loads(f_json.read_text(encoding="utf-8")), None

    est, grid, score_col = spark_specs(maxmem)[clave]
    ev = BinaryClassificationEvaluator(labelCol="default", rawPredictionCol=score_col, metricName="areaUnderROC")
    ev_pr = BinaryClassificationEvaluator(labelCol="default", rawPredictionCol=score_col, metricName="areaUnderPR")
    r = {"clave": clave, "modelo": lm.NOMBRES[clave], "score_col": score_col, "numFolds": FOLDS, "maxMemoryInMB": maxmem,
         "maxBins": MAXBINS if clave in ("DT", "RF", "GB") else None}
    names = _param_names(grid)

    # ---- entrenamiento con validacion cruzada (incluye el reajuste final con todo el entrenamiento)
    t0 = time.time()
    cv = CrossValidator(estimator=est, estimatorParamMaps=grid, evaluator=ev, numFolds=FOLDS, foldCol="fold",
                        seed=lc.SEED, parallelism=1)
    cvm = cv.fit(train_f)
    r["t_entrenamiento_con_cv_s"] = time.time() - t0
    best = cvm.bestModel
    r["mejores_hiperparametros"] = _best_params(best, names) if names else {}
    r["auc_cv"] = float(max(cvm.avgMetrics))
    r["auc_cv_sd"] = float(cvm.stdMetrics[int(np.argmax(cvm.avgMetrics))]) if hasattr(cvm, "stdMetrics") else None
    cvres = pd.DataFrame([{**{p.name: v for p, v in pm.items()}, "mean_test_score": m_}
                          for pm, m_ in zip(grid, cvm.avgMetrics)])
    cvres.to_csv(out_dir / f"sp_cv_{clave}.csv", index=False)
    if verbose:
        print(f"[{clave}] mejores {r['mejores_hiperparametros']} · AUC CV {r['auc_cv']:.4f} · "
              f"entrenamiento+CV {r['t_entrenamiento_con_cv_s']:.0f} s", flush=True)

    # ---- prediccion sobre el test (una sola vez); la puntuacion queda en cache
    t0 = time.time()
    pred = (pred_con(best, test_f).select("id", "default", score_column(score_col),
                                          F.col(score_col).alias(score_col)).persist(StorageLevel.MEMORY_AND_DISK))
    n_pred = pred.count()
    r["t_prediccion_s"] = time.time() - t0
    r["auc_test"] = float(ev.evaluate(pred))
    r["pr_auc_test"] = float(ev_pr.evaluate(pred))

    # ---- transferencia al driver de id, default y puntuacion (solo del TEST, despues de entrenar)
    t0 = time.time()
    sc = pred.select("id", "default", "score").toPandas()
    r["t_transferencia_s"] = time.time() - t0
    sc.rename(columns={"default": "y"}).to_parquet(out_dir / f"sp_scores_test_{clave}.parquet", index=False)
    pred.unpersist()
    assert len(sc) == n_pred

    # ---- umbral con puntuaciones fuera de pliegue del entrenamiento (sin usar el test)
    t0 = time.time()
    est_best = est.copy({p: v for p, v in _extract_map(best, est, names).items()})
    r["umbral_f1"], r["f1_oof"] = oof_threshold(est_best, train_f, score_col)
    r["t_oof_umbral_s"] = time.time() - t0
    r["umbral_defecto"] = 0.5 if score_col == "probability" else 0.0
    if verbose:
        print(f"[{clave}] AUC test {r['auc_test']:.4f} · predicción {r['t_prediccion_s']:.1f} s · transferencia "
              f"{r['t_transferencia_s']:.1f} s · umbral OOF {r['umbral_f1']:.3f} ({r['t_oof_umbral_s']:.0f} s)", flush=True)
    f_json.write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")
    gc.collect()
    return r, (best if keep_model else None)


def _extract_map(best, est, names: list[str]) -> dict:
    """ParamMap del estimador con los mejores hiperparametros encontrados por el CrossValidator."""
    out = {}
    for n in names:
        p = est.getParam(n)
        out[p] = best.getNumTrees if n == "numTrees" else best.getOrDefault(n)
    return out


def refit_best(clave: str, res: dict, train_f, maxmem: int = MAXMEM):
    """Reajusta un modelo con sus mejores hiperparametros y todo el entrenamiento (para LIME, si el
    objeto del modelo ya no esta en memoria tras una reanudacion)."""
    est, _, _ = spark_specs(maxmem)[clave]
    for n, v in res["mejores_hiperparametros"].items():
        est = est.copy({est.getParam(n): (int(v) if float(v).is_integer() and n != "regParam" else v)})
    return est.fit(train_f)
