# %% [markdown]
# # 1 · Análisis exploratorio de datos (EDA)
#
# **Objetivo del capítulo.** Entender la estructura, la calidad y las relaciones de los datos de
# *Lending Club* **antes** de preprocesar o modelar, y dejar por escrito las decisiones que de ahí
# se derivan.
#
# **Pregunta de investigación.** Dado un préstamo en el momento de otorgarlo, ¿cuál es la probabilidad
# de que termine en incumplimiento (*Charged Off*)?
#
# ## Orden del análisis: partición previa a la exploración
#
# Si se explora el conjunto completo antes de dividirlo, las decisiones que se toman al explorar (qué
# variables usar, qué umbrales, qué transformaciones) quedan influidas por datos que, en la práctica,
# todavía no se conocerían. Por eso este capítulo sigue este orden:
#
# 1. **Carga y estructura** del archivo completo (dimensiones, tipos, columnas): no usan la etiqueta.
# 2. **Se define la población y se hace la partición 80/20** (estratificada) y se guarda.
# 3. **Todo análisis que mire la variable objetivo, las distribuciones o las relaciones** (auditoría
#    de fuga, faltantes, valores atípicos, correlaciones, pruebas) usa **solo el entrenamiento**. El 20 % de
#    prueba se aparta y no se examina hasta la evaluación final.
#
# ```{note}
# Se usa el archivo **completo** (2 260 701 filas × 151 columnas), sin muestreo. Las únicas excepciones
# son unas pocas visualizaciones costosas (*pairplot*, violines, matriz de dispersión), donde se
# dibuja una muestra aleatoria **solo para el gráfico**, y dos pruebas que se calculan sobre muestras
# por costo o por validez de la prueba: Shapiro-Wilk (5 000 filas) y el V de Cramér entre pares de
# variables categóricas (300 000 filas). El resto de estadísticas y pruebas usa todas las filas de
# entrenamiento.
# ```
#
# Orden: (1.1) carga y visión general · (1.2) la variable objetivo · (1.3) partición ·
# (1.4) auditoría de fuga · (1.5) valores faltantes · (1.6) univariado · (1.7) bivariado ·
# (1.8) multicolinealidad · (1.9) visualizaciones avanzadas · (1.10) sesgo temporal ·
# (1.11) resumen ejecutivo.

# %%
import gc
import platform
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from IPython.display import display
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path.cwd().parent))
from src import lc_utils as lc  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="The figure layout has changed to tight")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold"})
RESULTS, DATA_DIR = Path("../resultados"), Path("../data")
RESULTS.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
proc = psutil.Process()
print(f"Python {platform.python_version()} · pandas {pd.__version__} · numpy {np.__version__} "
      f"· RAM total {psutil.virtual_memory().total / 1e9:.1f} GB")

# %% [markdown]
# ## 1.1 Carga inicial y visión general
#
# Se carga el CSV **completo** con `pandas` (sin `nrows`, `sample` ni filtros de filas). En el
# capítulo de PySpark se mide el mismo proceso con Spark para comparar tiempos de carga.

# %%
t0 = time.time()
df = pd.read_csv(lc.DATA_PATH, low_memory=False)
T_CARGA_PANDAS = time.time() - t0
print(f"Carga con pandas: {T_CARGA_PANDAS:.1f} s | memoria del proceso: {proc.memory_info().rss / 1e9:.2f} GB")
print(f"Dimensión del dataset: {df.shape[0]:,} filas × {df.shape[1]} columnas")
n_filas_archivo, n_cols_archivo = df.shape

# %%
cols_vista = ["id", "loan_amnt", "term", "int_rate", "grade", "emp_length", "home_ownership",
              "annual_inc", "purpose", "dti", "fico_range_high", "loan_status"]
print("Primeras filas (head):")
display(df[cols_vista].head())
print("Últimas filas (tail):")
display(df[cols_vista].tail())

# %% [markdown]
# **Observación.** Las dos últimas filas del archivo **no son préstamos**: en `id` traen el texto
# "Total amount funded in policy code 1/2: …", es decir, totales que Lending Club agregó al final.
# Por eso el archivo trae 33 filas con `loan_status` vacío (las 2 de totales y otras 31 sin
# información). Estas filas se descartan al definir la población; el problema solo se advierte al inspeccionar el final del archivo.

# %%
print("info() resumido:")
df.info(verbose=False, memory_usage="deep")
tipos = df.dtypes.value_counts().rename("columnas").to_frame()
tipos.index = tipos.index.astype(str)
display(tipos)

# %%
desc = df.describe(include="number").T
desc.to_csv(RESULTS / "describe_completo.csv")
print(f"describe() de las {len(desc)} columnas numéricas guardado en resultados/describe_completo.csv")
display(desc.loc[["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "fico_range_high",
                  "open_acc", "revol_bal", "revol_util", "total_acc"]])

# %% [markdown]
# Los rangos ya insinúan problemas de calidad que se estudiarán en el entrenamiento:
# `annual_inc` llega a más de 100 millones, `dti` a 999 y `revol_util` a cerca de 900 %.

# %% [markdown]
# ## 1.2 Variable objetivo y población de estudio
#
# La variable `loan_status` toma diez valores (tabla siguiente). La definición más directa de la variable
# objetivo sería `default = 1` si `loan_status` es `Charged Off` y `0` en cualquier otro caso.

# %%
estados = df["loan_status"].value_counts(dropna=False).rename("préstamos").to_frame()
estados["%"] = 100 * estados["préstamos"] / len(df)
estados = estados.rename(index={np.nan: "sin estado"})
display(estados)

fig, ax = plt.subplots(figsize=(9, 4))
estados["préstamos"].iloc[::-1].plot.barh(ax=ax, color="#3b6ea5")
ax.set(title="Estados de los préstamos (archivo completo)", xlabel="préstamos")
ax.xaxis.set_major_formatter(lambda v, pos: f"{v:,.0f}")
plt.tight_layout()
plt.show()

# %% [markdown]
# Esa definición marcaría como `0 = Fully Paid` a **todos** los préstamos que no son
# *Charged Off*, incluidos 878 317 préstamos *Current* (siguen pagándose; su desenlace aún no se
# conoce), además de 34 252 atrasados o en periodo de gracia y 2 822 filas en otros estados. Etiquetar
# como pagados a los vigentes sesgaría la etiqueta (mezclaría "pagó" con "todavía no se sabe") y
# reduciría la tasa de incumplimiento aparente del 20.0 % al 11.9 %.
#
# La variable objetivo se define entonces solo para préstamos con desenlace conocido: `Fully Paid` (0) y
# `Charged Off` (1). La restricción se aplica por igual a todas las filas y no es un muestreo: se lee y
# procesa el archivo completo.

# %%
literal = (df["loan_status"] == "Charged Off").astype(int)
resuelto = df["loan_status"].isin(lc.RESOLVED)
comp = pd.DataFrame({
    "filas": [len(df), int(resuelto.sum())],
    "default = 1": [int(literal.sum()), int((df.loc[resuelto, "loan_status"] == "Charged Off").sum())],
}, index=["Fórmula literal (todo el archivo)", "Solo desenlace conocido (Fully Paid / Charged Off)"])
comp["tasa de incumplimiento"] = comp["default = 1"] / comp["filas"]
display(comp.style.format({"filas": "{:,.0f}", "default = 1": "{:,.0f}", "tasa de incumplimiento": "{:.2%}"}))

# Conteos por año de otorgamiento (estructura del archivo; se usan en la sección 1.10)
anio_todos = pd.to_numeric(df["issue_d"].str[-4:], errors="coerce").dropna().astype(int).value_counts().sort_index()
anio_resueltos = pd.to_numeric(df.loc[resuelto, "issue_d"].str[-4:], errors="coerce").dropna().astype(int).value_counts().sort_index()

# %% [markdown]
# La tasa real de incumplimiento entre préstamos con desenlace conocido es ≈ 20 %; la definición
# directa la subestima (≈ 12 %) porque cuenta los préstamos aún vigentes como si fueran buenos.

# %% [markdown]
# ## 1.3 Partición entrenamiento / prueba (antes de explorar)
#
# Se construye la tabla de modelado (solo desenlace conocido) y se hace **ahora** la partición 80/20
# **estratificada** (misma proporción de incumplimientos en ambos lados) con semilla fija. La asignación
# se guarda **una sola vez** en `data/split_ids.parquet` con las columnas `id` y `split` (`train` / `test`),
# más `fold` (pliegue 0, 1 o 2 de las filas de entrenamiento, también estratificado) y `rank` (orden aleatorio
# fijo, para los experimentos de escala). Todos los capítulos (scikit-learn y PySpark) leen
# **exactamente este archivo**: así se comparan modelos sobre los mismos préstamos (requisito de la prueba de
# DeLong) y ambos entornos validan con **los mismos 3 pliegues**.
#
# **A partir de aquí, el conjunto de prueba no se toca**: se descarta de la memoria de este cuaderno.

# %%
m_all = lc.build_model_frame(df[lc.RAW_COLUMNS])
split = lc.make_split(m_all)
split.to_parquet(DATA_DIR / "split_ids.parquet", index=False)
es_test = (split["split"] == "test").to_numpy()
print(f"Población modelada: {len(m_all):,} préstamos · entrenamiento {int((~es_test).sum()):,} "
      f"({100 * (~es_test).mean():.0f} %) · prueba {int(es_test.sum()):,} ({100 * es_test.mean():.0f} %)")
print(f"Estratificación: incumplimiento en train {m_all.loc[~es_test, 'default'].mean():.4f} · "
      f"en test {m_all.loc[es_test, 'default'].mean():.4f} (deben ser casi idénticos)")
print("Pliegues de validación cruzada (solo entrenamiento):",
      split.loc[~es_test, "fold"].value_counts().sort_index().to_dict(),
      "· columnas guardadas:", ["id", "split", "fold", "rank"])

# Se conservan SOLO las filas de entrenamiento (tabla de modelado y las 151 columnas originales).
m = m_all.loc[~es_test].reset_index(drop=True)
pos_resueltos = np.flatnonzero(resuelto.to_numpy())              # posiciones en df de los préstamos con desenlace
df_tr = df.iloc[pos_resueltos[~es_test]]                          # sus filas de entrenamiento, con todas las columnas
n_train, n_test = int((~es_test).sum()), int(es_test.sum())
del df, m_all, split, literal
gc.collect()
print(f"Memoria del proceso tras separar el test: {proc.memory_info().rss / 1e9:.2f} GB · "
      f"de aquí en adelante solo existe el entrenamiento ({len(m):,} filas)")

# %% [markdown]
# ## 1.4 Auditoría de fuga de datos (*data leakage*)
#
# Lending Club publica columnas que se generan **después** de otorgar el préstamo (pagos recibidos,
# recuperaciones, último pago…). Un modelo que las use incurriría en fuga de datos: aprendería del
# desenlace, y ese modelo jamás podría usarse para decidir a quién prestar. Se mide qué tan
# informativas son (AUC de cada columna por separado contra `default`; 0.5 = nada, 1.0 = perfecta)
# para justificar que **no** se usen.
#
# ```{note}
# `int_rate` sí está disponible al originar el préstamo, así que no es fuga temporal. Pero la tasa la
# fija Lending Club con su propio modelo de riesgo, por lo que resume parcialmente esa evaluación
# interna. Los resultados que dependen de `int_rate` deben leerse como una predicción condicionada a
# la tasa ya asignada, no como una evaluación de riesgo desde cero.
# ```

# %%
y_tr = m["default"].to_numpy()
aud = {}
for c in lc.POST_ORIGINATION:
    x = lc._year_month(df_tr[c]) if c.endswith("_d") else pd.to_numeric(df_tr[c], errors="coerce")
    a = roc_auc_score(y_tr, x.fillna(-1).to_numpy())
    aud[c] = max(a, 1 - a)
DISPONIBLES = ["int_rate", "fico_range_high", "dti", "annual_inc", "loan_amnt"]      # disponibles al originar
for c in DISPONIBLES:
    a = roc_auc_score(y_tr, pd.to_numeric(df_tr[c], errors="coerce").fillna(-1).to_numpy())
    aud[c] = max(a, 1 - a)
aud = pd.Series(aud).sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(aud.index, aud.values, color=["#2e8b57" if k in DISPONIBLES else "#c0392b" for k in aud.index])
ax.axvline(0.5, color="gray", ls="--")
ax.set(xlim=(0.4, 1.0), xlabel="AUC de la columna por sí sola",
       title="AUC univariado de cada columna frente al incumplimiento")
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color="#c0392b", label="generada después del otorgamiento"),
                   Patch(color="#2e8b57", label="disponible al otorgar")], loc="lower right")
plt.tight_layout()
plt.show()
post = aud[[i for i in aud.index if i not in DISPONIBLES]]
AUD_MAX = float(post.max())
disp = aud[DISPONIBLES]
print(f"Mayor AUC individual entre columnas posteriores al préstamo: {post.idxmax()} = {AUD_MAX:.3f}")
print(f"Mayor AUC individual entre las variables disponibles al originar: {disp.idxmax()} = {disp.max():.3f}")
print(f"Columnas posteriores con AUC > 0.7: {int((post > 0.7).sum())} de {len(post)}")

# %% [markdown]
# Varias columnas generadas después del préstamo separan por sí solas a los
# incumplidores mucho mejor que cualquier variable disponible al originar (comparar las barras rojas
# con las verdes). Un modelo que las use tendría un AUC casi perfecto y un valor práctico **nulo**:
# no se pueden conocer al decidir a quién prestar. Por eso el modelo usa solo variables conocidas en
# el momento de otorgar el préstamo (lista en `src/lc_utils.py`).

# %% [markdown]
# ## 1.5 Valores faltantes
#
# ### 1.5.1 Panorama de las 151 columnas (filas de entrenamiento)

# %%
nulos = pd.DataFrame({"nulos": df_tr.isna().sum(), "%": 100 * df_tr.isna().mean()}).sort_values("%", ascending=False)
nulos.to_csv(RESULTS / "nulos_por_columna.csv")
N_MAS70, N_MAS30 = int((nulos["%"] > 70).sum()), int((nulos["%"] > 30).sum())
print(f"Columnas sin ningún nulo: {(nulos['nulos'] == 0).sum()} · con >70 % nulos: {N_MAS70} · con >30 % nulos: {N_MAS30}")
display(nulos.head(20))

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(range(len(nulos)), nulos["%"].values, color="#3b6ea5", width=1.0)
ax.axhline(70, color="#c0392b", ls="--", label="70 % (eliminar)")
ax.axhline(30, color="#e67e22", ls="--", label="30 % (revisar)")
ax.set(xlabel="columnas (ordenadas)", ylabel="% de nulos", title="Porcentaje de nulos por columna (151 columnas)")
ax.legend()
plt.tight_layout()
plt.show()

# %%
top40 = nulos.head(40).index.tolist()
muestra = df_tr[top40].sample(3000, random_state=lc.SEED)     # solo para dibujar el patrón
msno.matrix(muestra, figsize=(14, 5), sparkline=False, fontsize=8, color=(0.23, 0.43, 0.65))
plt.title("Patrón de nulos (3 000 filas al azar × las 40 columnas con más nulos; solo visualización)")
plt.show()
prefijos = pd.Series([("hardship" if "hardship" in c else "settlement" if "settlement" in c else
                       "joint / co-solicitante" if ("joint" in c or c.startswith("sec_app")) else "otras")
                      for c in nulos[nulos["%"] > 70].index]).value_counts().rename("columnas con >70 % de nulos")
display(prefijos.to_frame())

# %% [markdown]
# Los nulos **no están repartidos al azar**: se concentran en bloques de columnas que se
# vacían juntas (la tabla anterior agrupa las de más de 70 %: *hardship*, *settlement*, préstamos
# conjuntos / co-solicitantes…). Es un patrón *sistemático* (depende del tipo de préstamo o de qué
# empezó a registrar Lending Club y cuándo), no aleatorio.
#
# ### 1.5.2 Plan de tratamiento (regla por columna)
#
# | Porcentaje de nulos | Tratamiento |
# |---|---|
# | > 70 % | Eliminar la columna (casi todas son posteriores al préstamo o de subpoblaciones) |
# | 30 % – 70 % | Se excluyen del modelo, salvo justificación explícita (tabla siguiente) |
# | < 30 % (variables del modelo) | Numéricas: valor centinela −1 (los árboles lo aíslan como "sin dato"); categóricas: categoría `Desconocido` |
#
# El análisis de las variables que **sí** entran al modelo se presenta en la sección 1.5.3. La tabla
# siguiente documenta cada columna de la franja 30–70 %, agrupada por el motivo real de su
# nulidad en vez de una única regla genérica.

# %%
banda = nulos[(nulos["%"] > 30) & (nulos["%"] <= 70)].copy()
anio_col = pd.to_numeric(df_tr["issue_d"].str[-4:], errors="coerce")
early, late = anio_col <= 2012, anio_col >= 2016
banda["% nulos (préstamos ≤ 2012)"] = [100 * df_tr.loc[early, c].isna().mean() for c in banda.index]
banda["% nulos (préstamos ≥ 2016)"] = [100 * df_tr.loc[late, c].isna().mean() for c in banda.index]
banda["justificación"] = "nulidad estructural: Lending Club empezó a registrar esta variable en años recientes"
banda.loc["mths_since_last_delinq", "justificación"] = (
    "disponible al otorgar el préstamo; el nulo indica ausencia de morosidad registrada, no falta al azar")
banda.loc["mths_since_recent_revol_delinq", "justificación"] = (
    "disponible al otorgar el préstamo; el nulo indica ausencia de morosidad revolvente registrada, no falta al azar")
banda["decisión"] = "excluida (posible mejora: indicador de nulo)"
display(banda[["%", "% nulos (préstamos ≤ 2012)", "% nulos (préstamos ≥ 2016)", "decisión", "justificación"]]
        .rename(columns={"%": "% nulos (train)"}).style.format(
            {"% nulos (train)": "{:.1f}", "% nulos (préstamos ≤ 2012)": "{:.1f}", "% nulos (préstamos ≥ 2016)": "{:.1f}"}))

# %%
# Se conservan solo columnas auxiliares (alineadas con `m`) para justificar descartes por redundancia.
extra = df_tr[["fico_range_low", "funded_amnt", "grade", "sub_grade"]].reset_index(drop=True)
assert len(extra) == len(m)
del df_tr
gc.collect()
print(f"Memoria del proceso ahora: {proc.memory_info().rss / 1e9:.2f} GB · tabla de modelado (train): {m.shape[0]:,} × {m.shape[1]}")

# %% [markdown]
# ### 1.5.3 Faltantes en las variables del modelo

# %%
nm = m[lc.FEATURES].isna().sum().rename("nulos").to_frame()
nm["%"] = 100 * nm["nulos"] / len(m)
nm = nm[nm["nulos"] > 0].sort_values("%", ascending=False)
display(nm)

# ¿Se relacionan los nulos con el objetivo? (si sí, no son "completamente al azar", MCAR)
filas = []
for c in nm.index:
    falta = m[c].isna()
    n_falta = int(falta.sum())
    chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(falta, m["default"]), correction=False)
    filas.append({"variable": c, "n con nulo": n_falta,
                  "tasa default si falta": m.loc[falta, "default"].mean() if n_falta >= 10 else np.nan,
                  "tasa default si no falta": m.loc[~falta, "default"].mean(),
                  "diferencia": (m.loc[falta, "default"].mean() - m.loc[~falta, "default"].mean()) if n_falta >= 10 else np.nan,
                  "chi² p-valor": p if n_falta >= 10 else np.nan})
mcar = pd.DataFrame(filas).set_index("variable")
print("Con menos de 10 filas con nulo, la tasa de default y el χ² no son fiables y se dejan en blanco "
      f"({', '.join(mcar.index[mcar['n con nulo'] < 10])}).")
display(mcar)

nulos_anio = pd.DataFrame({c: m.groupby("issue_year")[c].apply(lambda s: 100 * s.isna().mean()) for c in nm.index})
display(nulos_anio.loc[[a for a in (2007, 2009, 2011, 2012, 2013, 2015, 2018) if a in nulos_anio.index]]
        .rename_axis("año").style.format("{:.1f} %"))
fig, ax = plt.subplots(figsize=(9, 3.6))
for c in nm.index:
    nulos_anio[c].plot(ax=ax, label=c, marker="o")
ax.set(title="% de nulos por año de otorgamiento", ylabel="% nulos", xlabel="año")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Las tablas anteriores responden a dos preguntas:
#
# * **¿Son nulos "al azar" (MCAR)?** Donde el χ² es significativo **y** la diferencia de tasas es
#   material (`emp_length`, `mort_acc`), faltar **está relacionado con el desenlace**: no son MCAR.
#   Imputar con la media borraría esa señal. Es un motivo para usar el centinela −1 en lugar de un
#   estadístico y dejar que el árbol decida si "sin dato" es informativo.
# * **¿Son estructurales?** En las variables donde el porcentaje de nulos depende fuertemente del año
#   (tabla y gráfico por año), el faltante nace de **cuándo** se registró la variable, no de un error
#   aleatorio.
#
# Para las variables con muy pocos nulos (`revol_util`, `dti`, `inq_last_6mths`) el χ² no muestra
# relación con el objetivo: el tratamiento es indiferente.

# %% [markdown]
# ## 1.6 Análisis univariado
#
# ### 1.6.1 Variables numéricas

# %%
def resumen_numerico(s: pd.Series) -> dict:
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = s.notna().sum()
    fuera = int(((s < lo) | (s > hi)).sum())
    return {"n": n, "faltantes %": 100 * s.isna().mean(), "media": s.mean(), "mediana": q2,
            "desv. est.": s.std(), "mín": s.min(), "P25": q1, "P75": q3, "máx": s.max(), "IQR": iqr,
            "límite inf.": lo, "límite sup.": hi, "outliers": fuera, "outliers %": 100 * fuera / n,
            "asimetría": s.skew(), "curtosis": s.kurt()}


tabla_num = pd.DataFrame({c: resumen_numerico(m[c]) for c in lc.NUMERIC}).T
tabla_num.to_csv(RESULTS / "eda_numericas.csv")
display(tabla_num[["n", "faltantes %", "media", "mediana", "desv. est.", "mín", "P25", "P75", "máx", "IQR"]])
display(tabla_num[["límite inf.", "límite sup.", "outliers", "outliers %", "asimetría", "curtosis"]])

# %% [markdown]
# **Valores imposibles o sospechosos** (más allá de los valores atípicos en sentido estadístico):

# %%
sospechosos = pd.Series({
    "dti = -1 (dato inválido)": int((m["dti"] < 0).sum()),
    "dti > 100 (más de 100 % de la deuda sobre ingreso)": int((m["dti"] > 100).sum()),
    "revol_util > 100 (uso de crédito > 100 %)": int((m["revol_util"] > 100).sum()),
    "annual_inc = 0": int((m["annual_inc"] == 0).sum()),
    "annual_inc > 1 000 000": int((m["annual_inc"] > 1_000_000).sum()),
}, name="filas")
sospechosos_pct = (100 * sospechosos / len(m)).rename("% de las filas")
display(pd.concat([sospechosos, sospechosos_pct], axis=1))

# %% [markdown]
# Solo `dti = -1` se trata como faltante: coincide con el centinela −1 que usa el imputador
# (sección 2.3), así que ese caso queda cubierto sin código adicional. Los demás valores de la tabla
# (`dti` o `revol_util` por encima de 100, ingresos en cero o superiores al millón, historial de
# crédito negativo) **no se recodifican**: son pocos, son valores extremos pero no imposibles para
# estas variables, y los modelos de árbol no los tratan de forma especial.

# %%
graficas = ["loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_high", "emp_length"]
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
for a, c in zip(ax.ravel(), graficas):
    s = m[c].dropna()
    tope = s.quantile(0.995)         # solo para dibujar: recorta la cola extrema del gráfico
    sns.histplot(s[s <= tope], bins=50, ax=a, color="#3b6ea5")
    a.axvline(s.mean(), color="#c0392b", ls="--", label=f"media {s.mean():,.1f}")
    a.axvline(s.median(), color="#27ae60", ls="-", label=f"mediana {s.median():,.1f}")
    a.set(title=f"{c} (hasta P99.5)", ylabel="frecuencia")
    a.legend(fontsize=8)
plt.suptitle("Histogramas de las variables numéricas principales", y=1.02, fontweight="bold")
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(2, 3, figsize=(15, 6))
for a, c in zip(ax.ravel(), graficas):
    sns.boxplot(x=m[c], ax=a, color="#8fb3d9", fliersize=0.6, flierprops={"alpha": 0.25, "markersize": 0.6})
    a.set(title=f"{c}: {tabla_num.loc[c, 'outliers %']:.1f} % fuera de [Q1−1.5·IQR, Q3+1.5·IQR]")
    if c == "annual_inc":
        a.set_xscale("log")
plt.tight_layout()
plt.show()

# %% [markdown]
# El método del IQR no distingue por sí solo un valor extremo **posible** de uno **inválido**; esa
# distinción se hace variable por variable, a partir de lo que cada una puede tomar en la realidad:
#
# * `annual_inc`, `revol_bal`, `dti` y `revol_util` tienen colas muy largas (asimetría muy alta,
#   ingresos de millones, `dti` hasta 999): el método del IQR marca varios puntos porcentuales como
#   atípicos. Salvo `dti = -1` (tabla de valores sospechosos, más arriba en esta sección, tratado
#   como faltante por coincidir con el centinela), el resto son valores extremos pero posibles
#   (ingresos altos, deudas muy comprometidas) y se conservan sin modificar.
# * En **conteos casi siempre cero** (`pub_rec`, `delinq_2yrs`) el IQR es 0 y cualquier valor
#   distinto de cero cuenta como atípico (≈ 17–19 %): el método **no aplica** a esas variables, así
#   que no se recodifica nada por este criterio.
# * `int_rate`, `fico_range_high` y `loan_amnt` son más simétricas y no presentan este problema.
# * **Normalidad.** Con más de un millón de filas cualquier prueba (Shapiro-Wilk, D'Agostino)
#   rechaza normalidad por desviaciones mínimas, así que no informa. Son más útiles la **asimetría y la
#   curtosis** (tabla) y el histograma. Más abajo se reporta Shapiro-Wilk sobre 5 000 datos (el tamaño hasta
#   el que la implementación de SciPy garantiza la precisión del valor p) solo como referencia.

# %%
norm = []
for c in lc.NUMERIC:
    es_binaria = m[c].nunique() <= 2
    s = m[c].dropna().sample(5000, random_state=lc.SEED)
    w, p = stats.shapiro(s)
    log_skew = np.log1p(m[c].dropna().clip(lower=0)).skew()
    reduce_suficiente = abs(m[c].skew()) > 1 and abs(log_skew) < 0.5 * abs(m[c].skew())
    norm.append({"variable": c, "asimetría": m[c].skew(), "asimetría tras log1p": log_skew,
                 "Shapiro W (n=5000)": w, "p-valor": p,
                 "¿transformar?": "no aplica (binaria)" if es_binaria else ("sí (log)" if reduce_suficiente else "no")})
display(pd.DataFrame(norm).set_index("variable"))

# %% [markdown]
# No se aplica ninguna transformación logarítmica a estas variables. Los modelos de árbol (árbol de
# decisión, bosque aleatorio, *gradient boosting*) son invariantes a transformaciones monótonas: dependen
# solo del *orden* de los valores, así que log o Box-Cox no cambian sus particiones. La regresión
# logística y la SVM lineal se entrenan con variables estandarizadas (`StandardScaler`), que igualan la
# escala entre variables pero no corrigen la asimetría. Naive Bayes gaussiano es invariante a cambios
# lineales de escala (estandarizar no cambia sus predicciones), pero sí supone normalidad dentro de cada
# clase, así que es el modelo más afectado por las colas largas de `annual_inc`, `revol_bal` y `dti`: una
# transformación logarítmica sí le habría servido a este modelo en particular. Es una limitación
# declarada y una causa posible de su menor AUC en los capítulos 2 y 3 (≈ 0.688).

# %% [markdown]
# ### 1.6.2 Variables categóricas

# %%
COL_RARA = "< 1 % (agrupar en 'Otros')"


def tabla_cat(s: pd.Series) -> pd.DataFrame:
    t = s.value_counts(dropna=False).rename("frecuencia").to_frame()
    t["frecuencia relativa %"] = 100 * t["frecuencia"] / len(s)
    t[COL_RARA] = t["frecuencia relativa %"] < 1
    return t


freq_cat = {c: tabla_cat(m[c]) for c in lc.CATEGORICAL + ["term_months"]}
for c, t in freq_cat.items():
    print(f"\n=== {c}: {len(t)} categorías · {int(t[COL_RARA].sum())} con menos del 1 % ===")
    display(t.head(15))

# %%
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
for a, c in zip(ax.ravel(), lc.CATEGORICAL):
    t = freq_cat[c]["frecuencia relativa %"].head(15)
    sns.barplot(x=t.values, y=t.index, ax=a, color="#3b6ea5")
    a.set(title=f"{c} (15 más frecuentes)", xlabel="% de los préstamos", ylabel="")
plt.tight_layout()
plt.show()
resumen_rara = {c: (int(t[COL_RARA].sum()), len(t)) for c, t in freq_cat.items()}
print("Categorías con < 1 % / total:", {k: f"{a}/{b}" for k, (a, b) in resumen_rara.items()})

# %% [markdown]
# De las tablas y el resumen anteriores se desprende lo siguiente:
#
# * `purpose` está dominada por una categoría (`debt_consolidation`) y varias categorías tienen
#   menos del 1 % de los préstamos.
# * `home_ownership`: hay categorías residuales (`ANY`, `NONE`, `OTHER`) que pueden unificarse.
# * `addr_state`: 51 categorías, de las cuales muchas tienen menos del 1 %.
# * **Decisión:** las categorías con < 1 % del entrenamiento se agruparán en una sola ("infrecuente")
#   en el `OneHotEncoder`. Esto reduce dimensionalidad y evita columnas casi vacías.
# * No hay categorías redundantes con distinto nombre; `Desconocido` agrupa los pocos nulos.

# %% [markdown]
# ### 1.6.3 Variable objetivo `default` y desbalance

# %%
vc = m["default"].value_counts().sort_index()
dist = pd.DataFrame({"clase": ["0 = Fully Paid", "1 = Charged Off"], "préstamos": vc.values,
                     "%": 100 * vc.values / len(m)})
display(dist.set_index("clase"))
RAZON_DESBALANCE = vc[0] / vc[1]
print(f"Razón de desbalance: {RAZON_DESBALANCE:.1f} pagados por cada incumplido")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].bar(dist["clase"], dist["préstamos"], color=["#2e8b57", "#c0392b"])
ax[0].set(title="Distribución de clases (entrenamiento)", ylabel="préstamos")
ax[1].pie(dist["préstamos"], labels=dist["clase"], autopct="%1.1f%%", colors=["#2e8b57", "#c0392b"], startangle=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Impacto del desbalance (≈ 80 / 20).**
#
# 1. **La exactitud es engañosa:** un modelo que predijera "todos pagan" tendría ≈ 80 % de exactitud y sería
#    inútil. Se reportan AUC, AUC-PR, sensibilidad (*recall*) y F1.
# 2. **Estratificar:** la partición entrenamiento/prueba y la validación cruzada mantienen la
#    proporción 80/20 en cada pliegue.
# 3. **Umbral:** con clases desbalanceadas las probabilidades de la clase minoritaria rara vez pasan
#    0.5; el umbral de decisión se elegirá con datos de entrenamiento (fuera de pliegue), no con el conjunto de prueba.
# 4. **Balanceo:** no se aplica remuestreo (SMOTE, etc.) ni pesos por clase (`class_weight`): la clase
#    minoritaria tiene más de 200 000 ejemplos en el entrenamiento, suficientes para estimarla, y el desbalance (≈ 4:1) es
#    moderado; se maneja con métricas que no dependen de la prevalencia y con el umbral. Los pesos por
#    clase o SMOTE son alternativas válidas que **no se probaron aquí**: cambiarían las probabilidades
#    del modelo (habría que recalibrarlas) y el umbral óptimo.
# 5. **Qué error importa más:** en crédito, prestar a quien no pagará (falso negativo) puede perder
#    buena parte del capital (menos las recuperaciones); rechazar a quien sí pagaría (falso positivo)
#    pierde solo los intereses. Por eso se
#    presta atención a la sensibilidad y al F1, y no a la exactitud.

# %% [markdown]
# ## 1.7 Análisis bidimensional
#
# ### 1.7.1 Variables numéricas vs `default`
#
# Con más de un millón de filas, casi toda diferencia resulta "estadísticamente significativa"
# (los valores p quedan por debajo de 0.001, muchas veces por debajo de la precisión de impresión):
# el tamaño de muestra vuelve significativas diferencias minúsculas. Por eso, además del valor p, se reporta el
# **tamaño del efecto**: *d* de Cohen (diferencia de medias en desviaciones estándar) y la
# correlación punto-biserial. Criterio de referencia (Cohen, 1988): |d| < 0.2 efecto pequeño · 0.2–0.5 moderado · > 0.5 grande.

# %%
# term_months solo toma dos valores (36 o 60 meses): se trata como categórica más abajo, no aquí, donde
# un d de Cohen o una prueba t no tendrían sentido (una variable de dos valores no es continua).
NUM_CONTINUAS = [c for c in lc.NUMERIC if c != "term_months"]
filas = []
for c in NUM_CONTINUAS:
    x1 = m.loc[m["default"] == 1, c].dropna()
    x0 = m.loc[m["default"] == 0, c].dropna()
    n1, n0 = len(x1), len(x0)
    welch = stats.ttest_ind(x1, x0, equal_var=False)
    mw = stats.mannwhitneyu(x1, x0, alternative="two-sided", method="asymptotic")
    sp = np.sqrt(((n1 - 1) * x1.var() + (n0 - 1) * x0.var()) / (n1 + n0 - 2))
    ok = m[c].notna()
    filas.append({"variable": c, "media default=1": x1.mean(), "media default=0": x0.mean(),
                  "dif. de medias": x1.mean() - x0.mean(), "d de Cohen": (x1.mean() - x0.mean()) / sp,
                  "corr. punto-biserial": stats.pointbiserialr(m.loc[ok, "default"], m.loc[ok, c]).statistic,
                  "p Welch": welch.pvalue, "p Mann-Whitney": mw.pvalue})
bi_num = pd.DataFrame(filas).set_index("variable").sort_values("corr. punto-biserial", key=abs, ascending=False)
bi_num.to_csv(RESULTS / "eda_bivariado_numericas.csv")
display(bi_num)

# %%
top6 = bi_num.index[:6].tolist()
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
for a, c in zip(ax.ravel(), top6):
    sns.boxplot(data=m, x="default", y=c, ax=a, showfliers=False, palette=["#2e8b57", "#c0392b"])
    a.set(title=f"{c} · d = {bi_num.loc[c, 'd de Cohen']:+.2f}", xlabel="default (0 = pagado, 1 = incumplido)")
plt.suptitle("Las 6 variables numéricas más asociadas a default (sin dibujar outliers)", y=1.02, fontweight="bold")
plt.tight_layout()
plt.show()

# %%
vio = m.sample(60000, random_state=lc.SEED)      # solo para dibujar
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, c in zip(ax, ["int_rate", "fico_range_high", "dti"]):
    d = vio[vio[c] <= vio[c].quantile(0.99)]
    sns.violinplot(data=d, x="default", y=c, ax=a, palette=["#2e8b57", "#c0392b"], cut=0)
    a.set(title=f"Violín · {c} (60 000 filas, P99)")
plt.tight_layout()
plt.show()

# %%
tabla_term = pd.crosstab(m["term_months"], m["default"])
tasa_term = (100 * m.groupby("term_months")["default"].mean()).rename("tasa de incumplimiento (%)")
v_term = lc.cramers_v(m["term_months"], m["default"])
display(pd.concat([tabla_term, tasa_term], axis=1))
print(f"V de Cramér (term_months vs default) = {v_term:.2f}")

# %% [markdown]
# La tabla ordena las variables **continuas** por su asociación con el incumplimiento. `int_rate` (la
# tasa que Lending Club asigna según su propio análisis de riesgo) es, con diferencia, la de mayor
# efecto (|d| ≈ 0.7, efecto grande); `fico_range_high` tiene un efecto moderado; `dti` queda justo en el
# límite (d ≈ 0.21) y el resto por debajo de 0.2 (efecto pequeño). `term_months`, al ser binaria (36 o
# 60 meses), se trata aparte: los préstamos a 60 meses incumplen el doble que los de 36 (32.5 % frente a
# 16.0 %; V de Cramér = 0.18), una asociación moderada. Aun así, "efecto pequeño" no significa "inútil":
# varias variables débiles combinadas pueden aportar. Ninguna variable sola separa bien las clases (los
# diagramas de caja y de violín se solapan mucho), por eso se espera un AUC moderado.

# %% [markdown]
# ### 1.7.2 Variables categóricas vs `default`
#
# Tabla de contingencia con la **tasa de incumplimiento** por categoría, prueba χ² de independencia
# y **V de Cramér** (tamaño del efecto: < 0.1 débil · 0.1–0.3 moderado · > 0.3 fuerte).

# %%
resumen_cat = []
for c in lc.CATEGORICAL + ["term_months"]:
    tab = pd.crosstab(m[c], m["default"])
    chi2, p, dof, _ = stats.chi2_contingency(tab, correction=False)
    resumen_cat.append({"variable": c, "categorías": tab.shape[0], "χ²": chi2, "gl": dof, "p-valor": p,
                        "V de Cramér": lc.cramers_v(m[c], m["default"])})
resumen_cat = pd.DataFrame(resumen_cat).set_index("variable").sort_values("V de Cramér", ascending=False)
resumen_cat.to_csv(RESULTS / "eda_bivariado_categoricas.csv")
display(resumen_cat)
print("Tasa de default por plazo:", {f"{int(k)} meses": f"{v:.1%}" for k, v in m.groupby("term_months")["default"].mean().items()})

# %%
fig, ax = plt.subplots(2, 3, figsize=(18, 9))
tasa_global = m["default"].mean()
for a, c in zip(ax.ravel(), lc.CATEGORICAL + ["term_months"]):
    t = m.groupby(c)["default"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    t = t[t["size"] >= 500].head(15)          # evita categorías diminutas en el gráfico
    sns.barplot(x=t["mean"] * 100, y=t.index.astype(str), ax=a, color="#c0392b")
    a.axvline(tasa_global * 100, color="black", ls="--", label=f"global {tasa_global:.1%}")
    a.set(title=f"Tasa de incumplimiento por {c}", xlabel="% de incumplimiento", ylabel="")
    a.legend(fontsize=8)
for a in ax.ravel()[len(lc.CATEGORICAL) + 1:]:
    a.remove()                                # la cuadrícula tiene un panel más que variables
plt.tight_layout()
plt.show()

# %%
cat_top = m.groupby("purpose")["default"].agg(tasa="mean", n="size").sort_values("tasa", ascending=False)
apil = pd.crosstab(m["purpose"], m["default"], normalize="index").loc[cat_top.index] * 100
apil.columns = ["pagado (0)", "incumplido (1)"]
ax = apil.plot.barh(stacked=True, figsize=(9, 5.5), color=["#2e8b57", "#c0392b"])
ax.set(title="purpose · composición pagado/incumplido (barras apiladas, 100 %)", xlabel="%")
plt.tight_layout()
plt.show()
display(pd.concat([cat_top.head(4), cat_top.tail(4)]).rename(columns={"tasa": "tasa de default", "n": "préstamos"})
        .style.format({"tasa de default": "{:.1%}", "préstamos": "{:,.0f}"}))

# %% [markdown]
# La tabla de V de Cramér ordena las categóricas por la fuerza de su asociación con el
# incumplimiento; el plazo (`term_months`) es la más asociada. Con más de un millón de filas, la
# prueba χ² da p < 0.001 en todos los casos, así que **el valor p no distingue entre ellas** y por
# eso se usa el V de Cramér como criterio. El gráfico y la
# tabla de `purpose` (4 categorías con más y 4 con menos incumplimiento) muestran que las diferencias
# entre propósitos existen pero son moderadas. `addr_state` casi no separa clases entre estados
# grandes.

# %% [markdown]
# ## 1.8 Multicolinealidad
#
# ### 1.8.1 Entre variables numéricas (Pearson)

# %%
corr = m[lc.NUMERIC].corr(method="pearson")
fig, ax = plt.subplots(figsize=(12, 9.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={"size": 7}, cbar_kws={"label": "r de Pearson"})
ax.set_title("Matriz de correlaciones (Pearson) entre variables numéricas")
plt.tight_layout()
plt.show()

iu = np.triu_indices_from(corr, k=1)
pares = pd.DataFrame({"variable A": corr.index[iu[0]], "variable B": corr.columns[iu[1]], "r": corr.values[iu]})
altos = pares[pares["r"].abs() > 0.7].sort_values("r", key=abs, ascending=False)
print(f"Pares con |r| > 0.7: {len(altos)}")
display(altos.set_index(["variable A", "variable B"]))

# %% [markdown]
# **Variables candidatas descartadas por redundancia.** Se comprueba también con columnas que
# *no* entran al modelo, para justificar por qué se dejaron fuera:

# %%
red = pd.DataFrame({
    "fico_range_high vs fico_range_low": [m["fico_range_high"].corr(extra["fico_range_low"])],
    "loan_amnt vs funded_amnt": [m["loan_amnt"].corr(extra["funded_amnt"])],
    "loan_amnt vs installment": [m["loan_amnt"].corr(m["installment"])],
    "int_rate vs grade (código ordinal)": [m["int_rate"].corr(extra["grade"].astype("category").cat.codes.astype(float))],
    "open_acc vs total_acc": [m["open_acc"].corr(m["total_acc"])],
}, index=["r de Pearson"]).T
display(red)

# %% [markdown]
# * `fico_range_low` es prácticamente idéntica a `fico_range_high` (r ≈ 1): se conserva solo
#   `fico_range_high`, porque la otra es redundante.
# * `funded_amnt` ≈ `loan_amnt` (r ≈ 1): redundante, no se usa.
# * `int_rate` se deriva de la subcategoría de riesgo (`grade`/`sub_grade`) que Lending Club asigna al
#   originar el préstamo (r = 0.95 con el código ordinal de `grade`): ambas codifican la misma evaluación
#   interna, así que se conserva solo la tasa (`int_rate`) y no `grade`/`sub_grade`.
# * `loan_amnt` e `installment` (r > 0.9) e `open_acc`/`total_acc` (r ≈ 0.7) sí se mantienen: aportan
#   información distinta (la cuota incorpora además el plazo y la tasa) y, para los modelos de árbol, la
#   colinealidad no sesga el resultado, solo reparte la importancia entre las variables (se tiene en
#   cuenta al interpretar). Para la regresión logística y la SVM lineal sí puede inflar la varianza de
#   los coeficientes; se conservan porque ambos modelos usan regularización L2, que atenúa ese efecto.
#
# ### 1.8.2 Entre variables categóricas (V de Cramér)

# %%
cv = pd.DataFrame(index=lc.CATEGORICAL, columns=lc.CATEGORICAL, dtype=float)
muestra_cv = m.sample(300000, random_state=lc.SEED)     # V de Cramér entre pares, con 300 000 filas por costo
for a in lc.CATEGORICAL:
    for b in lc.CATEGORICAL:
        cv.loc[a, b] = 1.0 if a == b else lc.cramers_v(muestra_cv[a], muestra_cv[b])
display(cv.round(3))
print("Nota: el V de Cramér entre pares de variables se calcula con 300 000 filas de entrenamiento al azar por "
      "costo computacional; el resto de análisis usa todas las filas de entrenamiento.")

# %% [markdown]
# Las variables categóricas están poco asociadas entre sí (ver la matriz): no hay una redundancia
# fuerte que justifique excluir alguna.

# %% [markdown]
# ## 1.9 Visualizaciones avanzadas

# %%
cols_pp = ["int_rate", "loan_amnt", "dti", "fico_range_high", "annual_inc"]
pp = m.sample(6000, random_state=lc.SEED)[cols_pp + ["default"]].copy()
pp["annual_inc"] = np.log10(pp["annual_inc"].clip(lower=1))
pp = pp.rename(columns={"annual_inc": "log10(annual_inc)"})
pp = pp[pp["dti"].between(0, 60)]
g = sns.pairplot(pp, hue="default", corner=True, palette=["#2e8b57", "#c0392b"],
                 plot_kws={"alpha": 0.35, "s": 9}, diag_kind="kde")
g.figure.suptitle("Pairplot (muestra de 6 000 filas de entrenamiento, solo visualización)", y=1.02, fontweight="bold")
plt.show()

# %%
kd = m.sample(150000, random_state=lc.SEED)
fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
for a, c in zip(ax, ["int_rate", "fico_range_high", "dti", "loan_amnt"]):
    d = kd[kd[c] <= kd[c].quantile(0.99)]
    sns.kdeplot(data=d, x=c, hue="default", common_norm=False, fill=True, alpha=0.35,
                palette=["#2e8b57", "#c0392b"], ax=a)
    a.set(title=f"Densidad por clase · {c}", ylabel="densidad")
plt.tight_layout()
plt.show()

# %%
sm = m.sample(20000, random_state=lc.SEED)[["int_rate", "fico_range_high", "dti", "revol_util"]]
sm = sm[(sm["dti"].between(0, 60)) & (sm["revol_util"].between(0, 120))]
pd.plotting.scatter_matrix(sm, figsize=(9, 8), alpha=0.15, diagonal="hist", s=4, color="#3b6ea5")
plt.suptitle("Matriz de dispersión (20 000 filas de entrenamiento, solo visualización)", y=1.0, fontweight="bold")
plt.show()

# %% [markdown]
# El *pairplot* y las densidades por clase confirman que las dos clases **se solapan
# mucho** en todas las variables: no hay una frontera limpia. Las relaciones entre variables no son
# perfectamente lineales (dispersión amplia en la matriz de dispersión), algo que un modelo de
# árboles puede capturar sin transformaciones.

# %% [markdown]
# ## 1.10 Sesgo temporal y censura
#
# Los préstamos se otorgaron entre 2007 y 2018. Los más recientes, sobre todo los de 60 meses,
# **aún no han terminado**: si solo se consideran los ya resueltos, los de años recientes están
# **sobrerrepresentados en desenlaces rápidos**. Es un problema clásico de **censura**. La primera
# tabla usa el archivo completo (solo cuenta filas, sin mirar la variable objetivo de la prueba); la
# tasa de incumplimiento se calcula con el entrenamiento.

# %%
resumen_anio = pd.DataFrame({"otorgados (archivo completo)": anio_todos,
                             "con desenlace conocido (archivo completo)": anio_resueltos})
resumen_anio["% resuelto"] = 100 * resumen_anio["con desenlace conocido (archivo completo)"] / resumen_anio["otorgados (archivo completo)"]
resumen_anio["tasa de default (entrenamiento)"] = m.groupby("issue_year")["default"].mean()
display(resumen_anio.style.format({"otorgados (archivo completo)": "{:,.0f}",
                                   "con desenlace conocido (archivo completo)": "{:,.0f}",
                                   "% resuelto": "{:.1f}", "tasa de default (entrenamiento)": "{:.1%}"}))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
resumen_anio["% resuelto"].plot(marker="o", ax=ax[0], color="#3b6ea5")
ax[0].set(title="% de préstamos con desenlace conocido, por año", ylabel="%", xlabel="año de otorgamiento")
(100 * resumen_anio["tasa de default (entrenamiento)"]).plot(marker="o", ax=ax[1], color="#c0392b")
ax[1].set(title="Tasa de default (entrenamiento), por año", ylabel="%", xlabel="año de otorgamiento")
plt.tight_layout()
plt.show()

# %% [markdown]
# Dos consecuencias de este patrón:
#
# * De cierto año en adelante cae el porcentaje de préstamos con desenlace conocido (muchos siguen
#   vigentes), y la tasa de incumplimiento de los resueltos deja de ser comparable entre años. El
#   modelo aprenderá una mezcla de épocas con **distinta economía y distinto grado de censura**.
# * Además de la partición aleatoria estratificada, se incluye una **validación temporal**
#   complementaria (entrenar con años anteriores y probar con uno posterior). La validación aleatoria
#   tiende a sobrestimar el desempeño cuando los datos tienen estructura temporal, porque el modelo se
#   entrena con observaciones posteriores a las que evalúa (Bergmeir y Benítez, 2012).

# %% [markdown]
# ## 1.11 Resumen ejecutivo del EDA

# %%
mas_num = bi_num.index[0], bi_num.index[1]
mas_cat = resumen_cat.index[0], resumen_cat.index[1]
resumen = pd.DataFrame([
    ["Tamaño del archivo", f"{n_filas_archivo:,} filas × {n_cols_archivo} columnas (carga pandas: {T_CARGA_PANDAS:.0f} s)"],
    ["Población y partición", f"{n_train + n_test:,} préstamos con desenlace conocido (Fully Paid / Charged Off): "
                              f"{n_train:,} de entrenamiento y {n_test:,} de prueba (apartada); se excluyen "
                              f"{n_filas_archivo - n_train - n_test:,} vigentes/atrasados/otros"],
    ["Calidad: nulos", f"{N_MAS70} columnas con >70 % de nulos (se descartan); variables del modelo con nulos: "
                       f"{', '.join(nm.index)}"],
    ["Calidad: valores extremos o inválidos", f"{int(sospechosos.iloc[0]):,} fila(s) con dti = −1 (inválido, tratado como faltante); "
                                    f"{int(sospechosos.iloc[1]):,} con dti > 100; {int(sospechosos.iloc[2]):,} con revol_util > 100 % (extremos, se conservan)"],
    ["Desbalance", f"{m['default'].mean():.1%} de incumplimiento (≈ {RAZON_DESBALANCE:.1f}:1)"],
    ["Variables numéricas más asociadas", f"{mas_num[0]} (d = {bi_num.loc[mas_num[0], 'd de Cohen']:+.2f}), "
                                          f"{mas_num[1]} (d = {bi_num.loc[mas_num[1], 'd de Cohen']:+.2f})"],
    ["Variables categóricas más asociadas", f"{mas_cat[0]} (V = {resumen_cat.loc[mas_cat[0], 'V de Cramér']:.2f}), "
                                            f"{mas_cat[1]} (V = {resumen_cat.loc[mas_cat[1], 'V de Cramér']:.2f})"],
    ["Colinealidad", (f"{len(altos)} pares numéricos con |r| > 0.7 (p. ej. {altos.iloc[0, 0]} y {altos.iloc[0, 1]}, "
                      f"r = {altos.iloc[0, 2]:.2f}); " if len(altos) else "ningún par numérico con |r| > 0.7; ")
     + "fico_range_low y funded_amnt redundantes (no usadas)"],
    ["Fuga de datos", f"columnas posteriores al préstamo con AUC individual de hasta {AUD_MAX:.2f}: excluidas del modelo"],
    ["Sesgo temporal", "los préstamos recientes están censurados; se añade validación temporal complementaria"],
], columns=["Hallazgo", "Detalle"]).set_index("Hallazgo")
pd.set_option("display.max_colwidth", 200)
display(resumen)
resumen.to_csv(RESULTS / "eda_resumen.csv")
pd.Series({"t_carga_pandas_s": T_CARGA_PANDAS, "filas": int(n_filas_archivo), "columnas": int(n_cols_archivo),
           "filas_modelo": int(n_train + n_test), "filas_train": int(n_train), "filas_test": int(n_test),
           "tasa_default_train": float(m["default"].mean())}).to_json(RESULTS / "eda_metricas.json")

# %% [markdown]
# ### Decisiones que guían el preprocesamiento
#
# 1. **Población y etiqueta:** solo `Fully Paid` / `Charged Off` (desenlace conocido).
# 2. **Partición:** 80/20 estratificada, hecha **antes** del EDA; el conjunto de prueba no se tocó.
# 3. **Variables:** las disponibles al originar el préstamo (17 numéricas + 4 categóricas). Se
#    descartan las posteriores al préstamo (fuga), `fico_range_low`, `funded_amnt`, `grade` y
#    `sub_grade` (redundantes).
# 4. **Nulos:** centinela −1 en numéricas (sin parámetros aprendidos ⇒ sin fuga); `Desconocido` en categóricas.
# 5. **Categóricas:** *One-Hot* agrupando en "infrecuente" las de < 1 % del entrenamiento.
# 6. **Valores atípicos:** se conservan (en su mayoría son valores extremos posibles y los árboles son
#    robustos); los pocos valores imposibles (`dti` < 0) quedan cubiertos por el centinela.
# 7. **Escalado:** no lo necesitan los modelos de árbol; regresión logística y SVM lineal sí se
#    escalan (`StandardScaler`, ajustado solo con el entrenamiento).
# 8. **Desbalance:** estratificar, métricas robustas (AUC, AUC-PR, sensibilidad, F1) y umbral elegido sin
#    el conjunto de prueba. Sin remuestreo.
# 9. **Validación:** partición aleatoria estratificada 80/20 más validación temporal complementaria.
