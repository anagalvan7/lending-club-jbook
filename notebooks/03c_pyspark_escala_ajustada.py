# %% [markdown]
# # 3c · PySpark con el paralelismo ajustado al equipo (análisis de sensibilidad)
#
# En el capítulo 3b el experimento de escala mostró que, con las **400 particiones de la sesión**, Spark
# gasta ≈ 160 s de costo fijo aunque se entrene con solo 10 000 filas, y que en la prueba de particiones
# (3b.4) usar ≈ 12 particiones (los hilos del equipo utilizado) fue ≈ 9.4 veces más rápido. Para que la comparación con scikit-learn no dependa
# únicamente de una configuración de Spark poco adecuada para el equipo utilizado, aquí se repite el
# experimento de escala con el DataFrame de entrenamiento reagrupado en **12 particiones**.
#
# ```{important}
# Esta variante **no sustituye** la configuración del capítulo 3: los resultados de los capítulos 3 y 5
# usan las 400 particiones. Sirve para responder a la pregunta
# "¿desde qué volumen conviene Spark?": la respuesta depende de cómo esté configurado, y una máquina
# de 12 hilos no es un clúster.
# ```

# %%
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

sys.path.insert(0, str(Path.cwd().parent))
from src import lc_spark as lcs  # noqa: E402
from src import lc_utils as lc  # noqa: E402

lcs.configure_java()
import pyspark  # noqa: E402
from pyspark import StorageLevel  # noqa: E402
from pyspark.ml.classification import RandomForestClassifier  # noqa: E402
from pyspark.ml.evaluation import BinaryClassificationEvaluator  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold"})
RESULTS, DATA_DIR = Path("../resultados"), Path("../data")
MAXMEM = 64
NPART = os.cpu_count()
print(f"Python {platform.python_version()} · PySpark {pyspark.__version__} · {NPART} hilos lógicos")

# %% [markdown]
# ## 3c.1 Sesión (la del capítulo 3) y datos
#
# La sesión sigue teniendo `spark.default.parallelism = 400`; lo único que cambia es el número de
# particiones del **DataFrame de entrenamiento** de cada ajuste.

# %%
t0 = time.time()
spark = lcs.start_session()
train_f, test_f, prep_model, keep, to_model_input, feat, train_raw = lcs.prepare(spark, DATA_DIR)
n_train_f = train_f.count()
print(f"Sesión y datos listos en {time.time() - t0:.0f} s · train {n_train_f:,}")
evaluador = BinaryClassificationEvaluator(labelCol="default", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
test_pocas = test_f.coalesce(3)   # pocas tareas simultáneas: cada una recibe una copia del modelo

# %% [markdown]
# ## 3c.2 Experimento de escala con 12 particiones
#
# Mismo bosque fijo que en 3b.5 (50 árboles, profundidad 10) y los mismos subconjuntos (`rank` < N).

# %%
tam = [10_000, 50_000, 100_000, 250_000, 500_000, n_train_f]
esc = []
for n in tam:
    sub = train_f.filter(F.col("rank") < n).repartition(NPART).persist(StorageLevel.MEMORY_AND_DISK)
    filas = sub.count()
    t0 = time.time()
    mdl = RandomForestClassifier(labelCol="default", featuresCol="features", numTrees=50, maxDepth=10, seed=lc.SEED,
                                 maxMemoryInMB=MAXMEM).fit(sub)
    t_fit = time.time() - t0
    a = evaluador.evaluate(mdl.transform(test_pocas))
    esc.append({"filas_train": int(filas), "t_ajuste_s": t_fit, "auc_test": a})
    print(f"{filas:>10,} filas · ajuste {t_fit:7.1f} s · AUC {a:.4f}", flush=True)
    sub.unpersist()
    del mdl
    pd.DataFrame(esc).to_csv(RESULTS / "escala_spark_12part.csv", index=False)
esc = pd.DataFrame(esc)
display(esc)
spark.stop()

# %% [markdown]
# ## 3c.3 Comparación con scikit-learn y con las 400 particiones
#
# La comparación de las tres versiones (scikit-learn, Spark con 400 particiones y Spark con 12) se presenta
# en la sección 6.4, junto con el ajuste afín `t = a + b·n` de cada curva. Con el entrenamiento completo, el
# mismo bosque tarda ≈ 16 s en scikit-learn, ≈ 69 s en Spark con 12 particiones y ≈ 258 s con 400 (≈ 4× y
# ≈ 16× más). Reagrupar a 12 particiones reduce el costo fijo de Spark de ≈ 170 s a ≈ 12 s, pero su costo por
# fila sigue siendo mayor que el de scikit-learn (≈ 54 s por millón de filas frente a ≈ 15 s), así que las
# rectas no se cruzan en el equipo utilizado.
#
# La ventaja de Spark aparece cuando los datos **no caben en la memoria de una máquina**, donde scikit-learn
# requeriría estrategias fuera de memoria, o cuando hay un **clúster** con muchos nodos. Con el equipo utilizado (16 GB
# de RAM) y este volumen (1.08 millones de filas de entrenamiento, 21 variables) no se llega a ese punto.
