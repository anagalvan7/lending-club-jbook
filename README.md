# Lending Club: scikit-learn frente a PySpark (Jupyter Book)

Libro (Jupyter Book) con el análisis exploratorio, los seis modelos, LIME, las pruebas estadísticas y la
discusión sobre el incumplimiento de préstamos de Lending Club, comparando scikit-learn y PySpark.

**Libro publicado:** <https://anagalvan7.github.io/lending-club-jbook/>

## Contenido

```
lending-club-jbook/
├── intro.md                          portada y resumen
├── notebooks/
│   ├── 01_eda                        análisis exploratorio (solo con el entrenamiento)
│   ├── 02_sklearn_seis_modelos       los seis modelos con GridSearchCV, umbral, métricas, escala
│   ├── 03_pyspark_seis_modelos       los seis modelos con CrossValidator (misma configuración de sesión)
│   ├── 03b_pyspark_costos            latencia, efecto de cada elemento de la configuración, escala
│   ├── 03c_pyspark_escala_ajustada   variante con 12 particiones (análisis de sensibilidad)
│   ├── 04_lime                       explicación de dos préstamos, estabilidad, comparación entre motores
│   ├── 05_estadistica                DeLong pareado, McNemar, bootstrap pareado, Holm
│   └── 06_comparacion                métricas, tiempos, curvas ROC/PR, escala, configuración, LIME
├── reflexion.md                      discusión, conclusiones y referencias
├── src/
│   ├── lc_utils.py                   partición común, preprocesamiento, métricas, LIME, DeLong simple
│   ├── lc_models.py                  definición de los seis modelos y espacios de búsqueda (scikit-learn)
│   ├── lc_spark.py                   sesión y preparación de datos en PySpark
│   ├── lc_spark_models.py            entrenamiento/evaluación de los seis modelos en PySpark
│   └── lc_stats.py                   DeLong rápido y directo, Holm, McNemar, bootstrap pareado
├── resultados/                       tablas y JSON generados por los cuadernos
├── _config.yml · _toc.yml            configuración del libro
└── requirements.txt                  versiones exactas
```

Cada cuaderno existe como `.py` (formato *percent* de Jupytext, fácil de revisar en Git) y como `.ipynb`
con **las salidas reales ya guardadas**. El libro **no vuelve a ejecutar** los cuadernos al construirse
(`execute_notebooks: off`) porque el capítulo 3 (los seis modelos en PySpark) tarda **≈ 29 horas** en esta
máquina.

## Cómo reproducirlo

1. Crear el entorno (Python 3.10 y JDK 17 para PySpark):
   ```bash
   conda create -n lending-club python=3.10 openjdk=17 -y
   conda activate lending-club
   pip install -r requirements.txt
   ```
2. Descargar el conjunto de datos *Lending Club* (`accepted_2007_to_2018Q4.csv`, Kaggle) y ubicarlo en
   `data/` o indicar su ruta:
   ```bash
   set LC_DATA=C:\ruta\accepted_2007_to_2018Q4.csv      # Windows (cmd)
   ```
   El CSV **no** está en el repositorio (≈ 1.7 GB).
3. Ejecutar los cuadernos **en orden** (el 01 crea la partición y los 3 pliegues, `data/split_ids.parquet`;
   el 02 guarda las puntuaciones de scikit-learn; el 03 lee la misma partición):
   ```bash
   cd notebooks
   jupytext --to ipynb 01_eda.py
   jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
   ```
   Tiempos aproximados en un equipo de 12 hilos y 16 GB de RAM: 01 ≈ 13 min · 02 (seis modelos,
   scikit-learn) ≈ 1 h 21 min · **03 (seis modelos, PySpark) ≈ 29 horas** · 03b ≈ 1 h 15 min · 03c ≈ 30 min
   · 04 ≈ 1 min · 05 ≈ 24 min · 06 < 1 min. Cierre otras aplicaciones al correr Spark: con 16 GB de RAM la
   memoria es el límite (el bosque de 100 árboles y el *boosting* usan casi toda la RAM disponible). El
   cuaderno 03 guarda cada modelo apenas termina (`resultados/sp_modelo_<clave>.json`); si el proceso se
   interrumpe, al relanzarlo **no repite** los modelos ya guardados.

## Construir el libro localmente

```bash
jupyter-book build .
```

El resultado queda en `_build/html/index.html`.

## Decisiones importantes

- Se **parte primero** (80/20 estratificado, semilla 42, 3 pliegues compartidos por los dos motores); el
  EDA, la selección de hiperparámetros y el umbral usan solo el entrenamiento; el conjunto de prueba no interviene en ninguna decisión de
  modelado.
- **Seis modelos** (regresión logística, árbol de decisión, bosque aleatorio, *gradient boosting*, SVM lineal,
  Naive Bayes)
  con espacios de búsqueda equivalentes en scikit-learn y PySpark.
- Solo variables conocidas **al otorgar** el préstamo; población: préstamos con desenlace conocido.
- Métricas: ROC AUC, AUC-PR, recall, precisión, F1, MCC, con IC por *bootstrap*; comparación entre motores
  con **DeLong pareado** (validado contra la versión directa), **McNemar** y **bootstrap pareado**, con
  corrección de Holm y un margen de relevancia práctica (ΔAUC ≥ 0.005) fijado de antemano.
- Spark con paralelismo y memoria fijados explícitamente en la sesión, más
  `spark.sql.execution.arrow.pyspark.enabled` (ver capítulo 3 y 3b para la justificación) y las decisiones
  de memoria (`maxMemoryInMB = 64`) necesarias para que corriera en 16 GB.
- **SVM lineal de scikit-learn:** con pérdida *hinge* (la misma que minimiza el `LinearSVC` de PySpark)
  converge a una solución degenerada (AUC ≈ 0.50); se analiza en la sección 2.5 y en la discusión.
