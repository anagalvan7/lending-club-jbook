"""Preparacion de datos con PySpark (capitulos 3 y 3b). Misma logica que `lc_utils.build_model_frame`.

Se importa `pyspark` dentro de las funciones para poder usar el resto del paquete sin Spark instalado.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from . import lc_utils as lc

NUM_RAW = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "fico_range_high", "open_acc",
           "revol_bal", "revol_util", "total_acc", "pub_rec", "delinq_2yrs", "inq_last_6mths", "mort_acc"]


def configure_java() -> None:
    """Spark 3.5 necesita Java 8/11/17 (no funciona con Java 25): se usa el JDK 17 del entorno conda."""
    if "JAVA_HOME" not in os.environ:
        jvm = Path(sys.prefix) / "Library" / "lib" / "jvm"
        if jvm.exists():
            os.environ["JAVA_HOME"] = str(jvm)
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def start_session(arrow: bool = False):
    """Sesion con la configuracion de paralelismo y memoria del capitulo 3 (+ barra de progreso apagada)."""
    configure_java()
    from pyspark.sql import SparkSession

    b = (SparkSession.builder.appName("LendingClub_Optimized")
         .config("spark.sql.shuffle.partitions", "400")
         .config("spark.default.parallelism", "400")
         .config("spark.executor.memory", "8g")
         .config("spark.driver.memory", "8g")
         .config("spark.memory.fraction", 0.8)
         .config("spark.memory.storageFraction", 0.3)
         .config("spark.ui.showConsoleProgress", "false"))
    if arrow:
        b = b.config("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def build_features(sdf):
    """Prestamos con desenlace conocido + variables (mismas reglas que pandas)."""
    from pyspark.sql import functions as F

    months = F.create_map([F.lit(x) for kv in lc.MONTHS.items() for x in kv])

    def year_month(col):
        return F.substring(col, -4, 4).cast("double") + (months[F.substring(col, 1, 3)].cast("double") - 1) / 12.0

    d = sdf.filter(F.col("loan_status").isin(lc.RESOLVED))
    return d.select(
        F.col("id").cast("string").alias("id"),
        (F.col("loan_status") == "Charged Off").cast("int").alias("default"),
        F.substring("issue_d", -4, 4).cast("int").alias("issue_year"),
        F.regexp_extract("term", r"(\d+)", 1).cast("double").alias("term_months"),
        F.when(F.col("emp_length").isNull(), F.lit(None).cast("double"))
         .when(F.col("emp_length").startswith("<"), F.lit(0.0))
         .otherwise(F.regexp_extract("emp_length", r"(\d+)", 1).cast("double")).alias("emp_length"),
        (year_month(F.col("issue_d")) - year_month(F.col("earliest_cr_line"))).alias("credit_history_years"),
        *[F.col(c).cast("double").alias(c) for c in NUM_RAW],
        *[F.coalesce(F.col(c), F.lit("Desconocido")).alias(c) for c in lc.CATEGORICAL])


def prepare(spark, data_dir: Path):
    """Lee el CSV COMPLETO, une con la particion comun (id, split, fold), ajusta los transformadores
    SOLO con el entrenamiento y devuelve (train_f, test_f, prep_model, keep, to_model_input, feat, train_raw).

    `train_f` y `test_f` quedan en cache con las columnas id, default, issue_year, term_months, rank,
    features (sin escalar) y features_sc (escaladas con StandardScaler: media 0, desviacion 1);
    `train_f` incluye ademas `fold` (0..2), el pliegue fijo que usa el CrossValidator (foldCol).
    """
    from pyspark import StorageLevel
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
    from pyspark.sql import functions as F

    raw = spark.read.csv(str(lc.DATA_PATH), header=True, inferSchema=False, escape='"')
    split = spark.read.parquet(str(data_dir / "split_ids.parquet"))
    feat = build_features(raw).join(F.broadcast(split), on="id", how="inner")
    train_raw = feat.filter(F.col("split") == "train")
    n_train = train_raw.count()
    keep = {c: sorted(r[c] for r in train_raw.groupBy(c).count().collect() if r["count"] >= 0.01 * n_train)
            for c in lc.CATEGORICAL}

    def to_model_input(sdf):
        out = sdf.na.fill(-1.0, subset=lc.NUMERIC)
        for c in lc.CATEGORICAL:
            out = out.withColumn(c, F.when(F.col(c).isin(keep[c]), F.col(c)).otherwise(F.lit("infrecuente")))
        return out

    idx_out = [f"{c}_idx" for c in lc.CATEGORICAL]
    ohe_out = [f"{c}_ohe" for c in lc.CATEGORICAL]
    stages = [StringIndexer(inputCols=lc.CATEGORICAL, outputCols=idx_out, handleInvalid="keep"),
              OneHotEncoder(inputCols=idx_out, outputCols=ohe_out, dropLast=False, handleInvalid="keep"),
              VectorAssembler(inputCols=lc.NUMERIC + ohe_out, outputCol="features", handleInvalid="keep"),
              StandardScaler(inputCol="features", outputCol="features_sc", withMean=True, withStd=True)]
    prep_model = Pipeline(stages=stages).fit(to_model_input(train_raw))          # SOLO con el entrenamiento
    cols = ["id", "default", "issue_year", "term_months", "rank", F.col("fold").cast("int").alias("fold"),
            "features", "features_sc"]
    train_f = prep_model.transform(to_model_input(train_raw)).select(*cols).persist(StorageLevel.MEMORY_AND_DISK)
    test_f = prep_model.transform(to_model_input(feat.filter(F.col("split") == "test"))).select(*cols).persist(
        StorageLevel.MEMORY_AND_DISK)
    train_f.count()
    test_f.count()
    return train_f, test_f, prep_model, keep, to_model_input, feat, train_raw
