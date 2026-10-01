# %% [markdown]
# # 3b · Costos de PySpark: latencia de predicción y efecto de la configuración
#
# El capítulo 3 entrenó y evaluó los seis modelos. Aquí se hacen las **pruebas de costo** que sustentan
# la discusión sobre el desempeño de Spark:
#
# 1. **Latencia de predicción** desde Python y qué implica para LIME (¿por qué tardaba ≈ 8 minutos
#    una predicción en una ejecución preliminar?).
# 2. **Efecto de cada elemento de la configuración** (caché, particiones, pliegues del `CrossValidator`).
# 3. **Experimento de escala**: ¿desde qué volumen conviene Spark?
#
# ```{note}
# Este capítulo corre en una **sesión nueva** (misma configuración del capítulo 3): las pruebas se
# hacen con **bosques pequeños** entrenados aquí, no con el bosque final de 100 árboles y
# profundidad 15. En 3b.2 se describe por qué, en una **ejecución preliminar** de este trabajo (sin
# Apache Arrow), una sola predicción tardaba ≈ 8 minutos, y en 3b.3 se comprueba. La sesión definitiva del
# capítulo 3 ya usa Arrow.
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
from pyspark.ml.functions import vector_to_array  # noqa: E402
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold"})
RESULTS, DATA_DIR = Path("../resultados"), Path("../data")
MAXMEM = 64
ESTADO = {}


def guardar():
    (RESULTS / "spark_costos.json").write_text(json.dumps(ESTADO, indent=2, default=float), encoding="utf-8")


print(f"Python {platform.python_version()} · PySpark {pyspark.__version__} · {os.cpu_count()} hilos lógicos")

# %% [markdown]
# ## 3b.1 Sesión y datos
#
# Misma configuración de sesión del capítulo 3 (con **Arrow desactivado al inicio**, para medir su efecto) y los
# mismos datos, preparados y **cacheados**.

# %%
t0 = time.time()
spark = lcs.start_session()
train_f, test_f, prep_model, keep, to_model_input, feat, train_raw = lcs.prepare(spark, DATA_DIR)
n_train_f, n_test_f = train_f.count(), test_f.count()
print(f"Sesión y datos listos en {time.time() - t0:.0f} s · train {n_train_f:,} · test {n_test_f:,} · "
      f"particiones de train_f: {train_f.rdd.getNumPartitions()}")
evaluador = BinaryClassificationEvaluator(labelCol="default", rawPredictionCol="rawPrediction", metricName="areaUnderROC")


def pred_con(modelo, df, n=3):
    """Aplica un modelo con pocas tareas simultáneas (cada tarea recibe una copia del modelo)."""
    return modelo.transform(df.coalesce(n))


# %% [markdown]
# ## 3b.2 El síntoma: una predicción con el bosque final tardaba ≈ 8 minutos
#
# En una ejecución preliminar de este trabajo (sin Arrow) se intentó medir la latencia con el **bosque final** (100 árboles,
# profundidad 15). La interfaz de Spark registró el tiempo de cada llamada
# (`resultados/spark_ui_llamadas_bosque_final.json`, copiado de la interfaz web de Spark). El bosque
# tiene millones de nodos, así que la primera hipótesis fue que el costo estuviera en el modelo: que
# cada llamada tuviera que serializarlo y enviarlo a las tareas. La prueba de 3b.3 no respalda esa
# hipótesis, aunque sin repetir la medición con el bosque final (ver el límite señalado en 3b.3).
# (La correspondencia entre cada llamada y su tamaño de lote se deduce del orden en que el código las
# ejecutaba: cinco llamadas con 1 fila y luego las primeras con 100; la ejecución se interrumpió allí.) Como estos
# tiempos se transcribieron de la interfaz web de Spark, se presentan como una observación preliminar y no
# como una medición controlada.

# %%
ui = json.loads((RESULTS / "spark_ui_llamadas_bosque_final.json").read_text())
ui = pd.DataFrame(ui)
display(ui[["filas por llamada", "segundos"]].rename(columns={"segundos": "tiempo de la llamada (s)"}))
print(f"Mediana de las {len(ui)} llamadas: {ui['segundos'].median():.0f} s = {ui['segundos'].median() / 60:.1f} min "
      f"(scikit-learn: {json.loads((RESULTS / 'sk_resultados.json').read_text())['latencia_predict_s']['1'] * 1000:.1f} ms para 1 fila)")
ESTADO["latencia_bosque_final_mediana_s"] = float(ui["segundos"].median())

# %% [markdown]
# El tiempo es **igual** para 1 fila o para 100: no depende de cuántos préstamos se
# predicen, sino de un **sobrecosto fijo elevado por llamada**. Como LIME necesita al menos una llamada por
# explicación, con esa configuración explicar **un solo préstamo** costaría ≈ 8 minutos (frente a
# fracciones de segundo con scikit-learn). Hay dos causas posibles y se miden **por separado**:
# (a) el costo de **enviar el lote** a Spark y (b) el costo de **enviar el modelo** a las tareas.

# %% [markdown]
# ## 3b.3 Dos costos por separado
#
# ### (a) Enviar el lote a Spark (`createDataFrame` desde pandas)
# Con `spark.default.parallelism = 400` (fijado en la sesión), `createDataFrame` reparte incluso un lote
# diminuto en hasta 400 particiones, y cada partición arranca un proceso de Python (operación especialmente lenta en Windows).
# Se mide con y sin **Apache Arrow** (transferencia en bloques). Sin Arrow se hace **una sola
# repetición** porque cada medición tarda minutos.

# %%
lote_base = feat.filter(F.col("is_test")).select(*lc.FEATURES).limit(5000).toPandas()     # 5 000 filas del test (post-entrenamiento)


def tiempo_crear(n, arrow, reps=3):
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true" if arrow else "false")
    ts = []
    for _ in range(reps):
        t0_ = time.perf_counter()
        spark.createDataFrame(lote_base.iloc[:n]).coalesce(1).count()
        ts.append(time.perf_counter() - t0_)
    return float(np.median(ts))


crear = []
for arrow in (True, False):
    for n in (1, 100, 5000):
        crear.append({"Arrow": "sí" if arrow else "no", "filas": n, "createDataFrame + count (s)": tiempo_crear(n, arrow, reps=3 if arrow else 1)})
        print(crear[-1], flush=True)
crear = pd.DataFrame(crear)
display(crear)
ESTADO["createDataFrame_s"] = crear.to_dict(orient="records")
guardar()

# %% [markdown]
# ### (b) Enviar el modelo a las tareas
# Se entrenan dos bosques pequeños (con **44** y con **≈ 19 000 nodos**) y se mide la predicción de un
# lote (con Arrow activado, para que el costo del lote sea mínimo). Si el tiempo dependiera del
# tamaño del modelo, debería crecer con el número de nodos.

# %%
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
modelos = {}
for nombre, kw in (("10 árboles, prof. 5", dict(numTrees=10, maxDepth=5)), ("20 árboles, prof. 10", dict(numTrees=20, maxDepth=10))):
    t0 = time.time()
    m_ = RandomForestClassifier(labelCol="default", featuresCol="features", seed=lc.SEED, maxMemoryInMB=MAXMEM, **kw).fit(train_f)
    modelos[nombre] = m_
    print(f"{nombre}: {m_.totalNumNodes:,} nodos · ajuste {time.time() - t0:.0f} s", flush=True)


def predecir(modelo, lote_pdf):
    s = spark.createDataFrame(lote_pdf).coalesce(1)
    x = prep_model.transform(to_model_input(s))
    return np.array(pred_con(modelo, x, n=1).select(vector_to_array("probability")[1]).toPandas().iloc[:, 0])


lat_rows = []
for nombre, m_ in modelos.items():
    for n in (1, 100, 5000):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            predecir(m_, lote_base.iloc[:n])
            ts.append(time.perf_counter() - t0)
        lat_rows.append({"modelo": nombre, "nodos": m_.totalNumNodes, "filas": n, "latencia mediana (s)": float(np.median(ts))})
lat = pd.DataFrame(lat_rows)
display(lat)
ESTADO["latencia_modelos_pequenos"] = lat.to_dict(orient="records")
guardar()

# %%
sk = json.loads((RESULTS / "sk_resultados.json").read_text())
mediano = lat[lat["modelo"] == "20 árboles, prof. 10"].set_index("filas")["latencia mediana (s)"]
comp = pd.DataFrame({"scikit-learn (mejor modelo con probabilidades)": {int(k): v for k, v in sk["latencia_predict_s"].items()},
                     "PySpark (20 árboles, prof. 10, Arrow)": mediano.to_dict()})
comp["PySpark / scikit-learn"] = comp.iloc[:, 1] / comp.iloc[:, 0]
comp.index.name = "filas por llamada"
display(comp.style.format({"scikit-learn (mejor modelo con probabilidades)": "{:.4f} s", "PySpark (20 árboles, prof. 10, Arrow)": "{:.3f} s",
                           "PySpark / scikit-learn": "{:,.0f}×"}))
ESTADO["latencia_predict_s"] = {str(k): float(v) for k, v in mediano.items()}
guardar()

# %% [markdown]
# De estas mediciones se desprenden tres conclusiones:
#
# * **(a) Sin Arrow, mover incluso una sola fila a Spark tarda ≈ 426 s (7.1 min)**, casi lo mismo para 1,
#   100 o 5 000 filas: el costo es fijo. La explicación más probable son las 400 particiones fijadas en la
#   sesión (400 tareas de Python que arrancar), aunque esta prueba solo varió Arrow y no el número de
#   particiones. Con Arrow es de centésimas de segundo. Ese costo fijo coincide con las llamadas de
#   ≈ 465 s del bosque final (3b.2): **la causa de los ≈ 8 minutos era el envío del lote sin Arrow, no el
#   modelo.**
# * **(b) Con Arrow, el tiempo de predicción es ≈ 0.3 s y prácticamente igual para un modelo de 44 nodos
#   que para uno de ≈ 19 000**, y casi igual para 1 fila que para 5 000. Es decir, todavía queda un
#   costo fijo de ≈ 0.3 s por llamada (planificar el trabajo, lanzar tareas, traer el resultado), que
#   **no depende del modelo ni del lote**.
# * scikit-learn tiene el modelo en la memoria del mismo proceso: la misma predicción de 1 fila cuesta
#   décimas de milisegundo. La diferencia va de **poco más de un orden de magnitud (≈ 38×) con 5 000 filas
#   a cerca de tres órdenes de magnitud (≈ 1 190×) con una sola fila**, porque el costo fijo de Spark pesa
#   más cuanto más pequeño es el lote; en ningún caso llega a "minutos" si se usa Arrow.
#
# ```{admonition} Alcance de la medición
# :class: warning
# No se repitió la medición con el bosque final (100 árboles, profundidad 15, millones de nodos): la
# sesión de la ejecución preliminar se cerró. Por eso **no se descarta** que un
# modelo de ese tamaño agregue algún costo de serialización, pero **no se observó** hasta 19 000 nodos
# y la diferencia de ≈ 8 min es consistente con el costo de enviar el lote sin Arrow.
# ```

# %% [markdown]
# ## 3b.4 ¿Qué efecto tuvo cada elemento de la configuración? (micro-*benchmarks*)
#
# Sobre **un bosque pequeño fijo** se mide qué ocurre al **quitar o cambiar** cada elemento. Son pruebas de
# tiempo, no el modelo final. Cada condición se midió una sola vez y en orden fijo, así que los factores
# son orientativos: el calentamiento de la JVM puede favorecer a la segunda medición.

# %%
def tiempo_fit(df_, n_trees=10, depth=10, seed=lc.SEED):
    t0_ = time.time()
    RandomForestClassifier(labelCol="default", featuresCol="features", numTrees=n_trees, maxDepth=depth, seed=seed,
                           maxMemoryInMB=MAXMEM).fit(df_)
    return time.time() - t0_


abl = []
# (a) Con y sin caché
peq = dict(n_trees=5, depth=5)
train_f.unpersist()
t_nocache = tiempo_fit(train_f, **peq)
train_f.persist(StorageLevel.MEMORY_AND_DISK)
train_f.count()
t_cache = tiempo_fit(train_f, **peq)
abl.append({"condición": "Caché tras VectorAssembler (5 árboles, prof. 5)", "con la condición (s)": t_cache,
            "sin / alternativa (s)": t_nocache, "alternativa": "sin .persist()"})
print(abl[-1], flush=True)

# (b) Particiones: 400 (fijadas en la sesión) vs 12 (≈ hilos)
tiempos_part = {}
for n_part in (400, os.cpu_count()):
    dfp = train_f.repartition(n_part).persist(StorageLevel.MEMORY_AND_DISK)
    dfp.count()
    tiempos_part[n_part] = tiempo_fit(dfp)
    dfp.unpersist()
abl.append({"condición": "400 particiones del DataFrame de entrenamiento (10 árboles, prof. 10)",
            "con la condición (s)": tiempos_part[400], "sin / alternativa (s)": tiempos_part[os.cpu_count()],
            "alternativa": f"{os.cpu_count()} particiones"})
print(abl[-1], flush=True)

# (c) CrossValidator: numFolds 2 vs 3 y parallelism 1 vs 3 (malla mínima de 2 combinaciones)
rf = RandomForestClassifier(labelCol="default", featuresCol="features", seed=lc.SEED, maxMemoryInMB=MAXMEM)
mini = ParamGridBuilder().addGrid(rf.numTrees, [5, 10]).addGrid(rf.maxDepth, [5]).build()
tiempos_cv = {}
for nf, par_ in ((2, 1), (3, 1), (3, 3)):
    t0_ = time.time()
    CrossValidator(estimator=rf, estimatorParamMaps=mini, evaluator=evaluador, numFolds=nf, seed=lc.SEED,
                   parallelism=par_).fit(train_f)
    tiempos_cv[(nf, par_)] = time.time() - t0_
abl.append({"condición": "CrossValidator con numFolds=3 (usado en los seis modelos) vs 2, malla de 2 combinaciones",
            "con la condición (s)": tiempos_cv[(3, 1)], "sin / alternativa (s)": tiempos_cv[(2, 1)], "alternativa": "numFolds=2"})
abl.append({"condición": "CrossValidator parallelism=1 (elegido) vs 3, 3 pliegues, malla de 2 combinaciones",
            "con la condición (s)": tiempos_cv[(3, 1)], "sin / alternativa (s)": tiempos_cv[(3, 3)], "alternativa": "parallelism=3"})
abl = pd.DataFrame(abl)
abl["factor (alternativa / condición)"] = abl["sin / alternativa (s)"] / abl["con la condición (s)"]
abl.to_csv(RESULTS / "spark_ablacion.csv", index=False)
display(abl.style.format({"con la condición (s)": "{:.1f}", "sin / alternativa (s)": "{:.1f}", "factor (alternativa / condición)": "{:.2f}"}))

# %% [markdown]
# **Lectura de la tabla.** "Factor" es cuántas veces tarda más (> 1) o menos (< 1) la alternativa frente a la
# opción usada en los capítulos 3 y 3b. Sirve para cuantificar qué efecto tuvo cada elemento de la
# configuración en el tiempo de entrenamiento.

# %% [markdown]
# ## 3b.5 Experimento de escala (mismo bosque fijo que en scikit-learn)
#
# Se ajusta **un bosque fijo** (50 árboles, profundidad 10) con subconjuntos crecientes del
# entrenamiento (`rank` < N, el mismo orden aleatorio fijo que usó scikit-learn), para ver cuándo
# cambia el ganador en velocidad. Es un *benchmark*, por eso aquí sí hay submuestras.

# %%
tam = [10_000, 50_000, 100_000, 250_000, 500_000, n_train_f]
esc = []
for n in tam:
    sub = train_f.filter(F.col("rank") < n).persist(StorageLevel.MEMORY_AND_DISK)
    filas = sub.count()
    t0 = time.time()
    mdl = RandomForestClassifier(labelCol="default", featuresCol="features", numTrees=50, maxDepth=10, seed=lc.SEED,
                                 maxMemoryInMB=MAXMEM).fit(sub)
    t_fit = time.time() - t0
    t0 = time.time()
    a = evaluador.evaluate(pred_con(mdl, test_f))
    t_pred = time.time() - t0
    esc.append({"filas_train": int(filas), "t_ajuste_s": t_fit, "t_prediccion_s": t_pred, "auc_test": a})
    print(f"{filas:>10,} filas · ajuste {t_fit:7.1f} s · predicción+AUC {t_pred:5.1f} s · AUC {a:.4f}", flush=True)
    sub.unpersist()
    del mdl
    pd.DataFrame(esc).to_csv(RESULTS / "escala_spark.csv", index=False)
esc = pd.DataFrame(esc)

# %% [markdown]
# ## 3b.6 Guardar y cerrar

# %%
guardar()
print("Resultados guardados. Claves del capítulo 3b:", sorted(ESTADO))
spark.stop()
