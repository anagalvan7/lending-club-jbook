# Incumplimiento de préstamos en Lending Club: scikit-learn frente a PySpark

**Ana Galván Arias, Valeria Oñate López, María José Ruiz Arias**
Septiembre de 2026

## Resumen

Se compara el desempeño predictivo y el costo computacional de seis modelos de clasificación (regresión
logística, árbol de decisión, bosque aleatorio, *gradient boosting*, SVM lineal y Naive Bayes),
implementados de forma equivalente en scikit-learn y en PySpark, sobre 1 345 310 préstamos de Lending Club
con desenlace conocido. Los dos motores comparten partición y pliegues de validación cruzada. La
comparación usa DeLong pareado, la prueba de McNemar y un *bootstrap* pareado, con corrección de Holm y un
margen de relevancia práctica fijado de antemano (ΔAUC ≥ 0.005). Los seis modelos alcanzan un AUC entre
0.69 y 0.72 en ambos entornos (*gradient boosting*, el mejor, 0.7196 y 0.7189); solo la SVM lineal difiere de
forma relevante, porque la de scikit-learn converge a una solución degenerada.
En tiempo, scikit-learn resulta unas 17 veces más rápido que PySpark en una sola máquina de 12 hilos.
Se documentan además la escalabilidad, el efecto de cada elemento de la configuración de Spark y las
explicaciones locales con LIME de dos préstamos mal clasificados por ambos entornos. **Palabras clave:**
incumplimiento crediticio, comparación de motores de cómputo, DeLong pareado, aprendizaje distribuido,
interpretabilidad local.

## Datos y diseño

Se predice si un préstamo de Lending Club terminará en incumplimiento (`Charged Off`) o se pagará por
completo (`Fully Paid`) con información disponible en el momento de otorgarlo. El diseño busca que el
desempeño reportado estime, sin sesgo, el que se obtendría con préstamos nuevos: el conjunto de prueba se
separa antes de cualquier análisis exploratorio, los hiperparámetros y el umbral se eligen sin tocarlo, y ese
conjunto no interviene en ninguna decisión de modelado.

- **Datos:** préstamos aceptados por Lending Club entre 2007 y el cuarto trimestre de 2018, en la versión
  pública de Kaggle [*All Lending Club loan data*](https://www.kaggle.com/datasets/wordsforthewise/lending-club):
  2 260 701 registros y 151 variables.
- **Población de estudio:** la variable objetivo se define solo para préstamos con desenlace conocido,
  `Fully Paid` (0) y `Charged Off` (1): **1 345 310** préstamos. El archivo contiene además 878 317
  préstamos vigentes y 34 252 atrasados o en periodo de gracia, cuyo desenlace aún no se observa, y 2 822
  filas en otros estados; todos se excluyen. Etiquetarlos como pagados sesgaría la variable objetivo y
  reduciría la tasa de incumplimiento aparente del 20.0 % al 11.9 %. La restricción se aplica por igual a
  todas las filas y no es un muestreo: se lee y procesa el archivo completo.
- **Partición:** 80 % / 20 % estratificada, semilla 42: 1 076 248 para entrenamiento y 269 062 para
  prueba, con 3 pliegues de validación cruzada compartidos por ambos motores.
- **Variables:** 21 (17 numéricas, 4 categóricas), elegidas por estar disponibles al otorgar el préstamo,
  no tener más de 30 % de valores faltantes en el entrenamiento (o que su ausencia no dependa del año de
  registro) y no ser redundantes entre sí. Se excluyen 15 columnas generadas después del desembolso, aunque
  algunas tengan un AUC univariado de hasta 0.94: esa capacidad predictiva proviene del propio desenlace y
  no estaría disponible al decidir un préstamo nuevo.

## Resultados principales

| Modelo | AUC prueba (scikit-learn) | AUC prueba (PySpark) | ΔAUC | Tiempo entren.+CV (scikit-learn) | Tiempo entren.+CV (PySpark) |
|---|---|---|---|---|---|
| Regresión logística | 0.7095 | 0.7095 | 0.00000004 (no significativo) | 15 s | 1 h 03 min |
| Árbol de decisión | 0.7022 | 0.7004 | 0.0018 (significativo, no relevante) | 59 s | 1 h 04 min |
| Bosque aleatorio | 0.7164 | 0.7152 | 0.0012 (significativo, no relevante) | 14 min | 5 h 52 min |
| *Gradient boosting* (mejor de los seis) | 0.7196 | 0.7189 | 0.0007 (significativo, no relevante) | 54 min | 10 h 07 min |
| SVM lineal | 0.4956 (solución degenerada) | 0.5985 | −0.1029 (significativo y relevante) | 10 min | 3 h 36 min |
| Naive Bayes | 0.6884 | 0.6883 | 0.0001 (significativo, no relevante) | 5 s | 43 min |

Con los seis modelos y el preprocesamiento, el entrenamiento con validación cruzada requirió ≈ 1 h 21 min
en scikit-learn y ≈ 22 h 32 min en PySpark (≈ 17 veces más), en la misma máquina de 12 hilos.

```{admonition} La SVM lineal de scikit-learn no separa las clases en estos datos
:class: warning
Con `LinearSVC(loss="hinge")` el AUC queda en ≈ 0.50, igual que el azar. El *solver* (`liblinear`)
converge, en 93 iteraciones y sin advertencias, a coeficientes del orden de 10⁻⁸ y a un intercepto de −1,
con los tres valores de `C` de la malla. Es la solución degenerada de la SVM con pérdida *hinge* descrita
por Rifkin, Pontil y Verri (1999): cuando las clases se solapan mucho y la clase mayoritaria cuadruplica a
la minoritaria, el punto *w* = 0, *b* = −1, que deja a todos los negativos justo sobre el margen, puede ser
el óptimo del problema regularizado; `liblinear` refuerza el efecto porque también regulariza el
intercepto. Las puntuaciones resultantes son prácticamente constantes (desviación estándar de 1.8×10⁻¹³),
así que el AUC de 0.496 equivale al azar, no a un modelo peor que el azar. La SVM de PySpark, que no
regulariza el intercepto y minimiza la misma pérdida con OWL-QN (un método cuasi-Newton que puede
detenerse antes del óptimo de una función no diferenciable), sí separa algo las clases (AUC 0.60). Ambas
diferencias son candidatas a explicar esa brecha, aunque no se aislaron por separado.
```

```{admonition} La exactitud (accuracy) no es un criterio adecuado en estos datos
:class: warning
Con el umbral 0.5, que solo minimiza el error cuando las probabilidades están calibradas y los costos de
cada error son iguales (condiciones que aquí no se cumplen), los modelos con buen AUC tienen ≈ 80 % de
exactitud pero detectan muy pocos incumplimientos: un modelo que dijera "todos pagan" tendría también
≈ 80 %. Por eso se reportan ROC AUC, AUC-PR, sensibilidad (*recall*), precisión, F1 y MCC, con el umbral elegido a partir de
predicciones fuera de pliegue del entrenamiento.
```

## Estructura del documento

| Capítulo | Contenido |
|---|---|
| **1. Análisis exploratorio** | Partición primero; calidad de datos, fuga y desbalance con el entrenamiento |
| **2. Los seis modelos con scikit-learn** | `GridSearchCV` sin `Pipeline`, 3 pliegues, umbral, métricas, hallazgo de la SVM, importancia, validación temporal, escala |
| **3. Los seis modelos con PySpark** | Configuración de la sesión, `CrossValidator` con los mismos pliegues (`foldCol`), LIME con el mejor modelo de Spark |
| **3b. Costos de PySpark** | Latencia, efecto de cada elemento de la configuración (caché, particiones, pliegues) y escala, con un bosque de referencia |
| **3c. Análisis de sensibilidad** | La misma escala con el paralelismo ajustado a los hilos del equipo utilizado |
| **4. LIME** | Explicación de dos préstamos mal clasificados por ambos entornos, estabilidad, visión global, comparación entre motores |
| **5. Comparación estadística** | DeLong pareado (validado contra la versión directa), McNemar, *bootstrap* pareado, Holm, margen de relevancia práctica |
| **6. Comparación de resultados** | Métricas, tiempos, curvas ROC/PR, escala, efecto de cada elemento de la configuración, LIME |
| **Discusión y conclusiones** | Costo computacional, equivalencia predictiva, significancia frente a relevancia, interpretabilidad, alcance, conclusiones y referencias |

## Decisiones metodológicas

**Partición antes del análisis exploratorio.** Es habitual explorar el conjunto completo y dividirlo
después. Aquí se procede al revés porque las decisiones que se toman al explorar (qué variables usar, cómo
tratar los valores extremos, qué categorías agrupar) incorporan información de los datos que las inspiran;
si incluyeran el conjunto de prueba, este dejaría de estimar el desempeño con préstamos nuevos. Por eso
todo análisis que involucra la variable objetivo, las distribuciones o las relaciones entre variables usa
solo el entrenamiento.

**Exclusión de columnas posteriores al desembolso.** Variables como `total_pymnt` o `recoveries` se
generan después de que el préstamo ya se está pagando o no, y de hecho codifican el resultado que se
quiere predecir. Un modelo entrenado con ellas alcanzaría un AUC casi perfecto pero sería inútil: al
otorgar el préstamo esas columnas todavía no existen. Se auditó el poder predictivo individual de cada una
para justificar su exclusión con evidencia.

**DeLong pareado, con la covarianza entre los dos AUC.** Los dos modelos que se comparan se evalúan sobre
las mismas 269 062 observaciones de prueba, así que sus curvas ROC están positivamente correlacionadas
(entre 0.79 y 1.00 en todos los pares que no incluyen la SVM). Ignorar esa covarianza infla el error
estándar de la diferencia y resta potencia a la prueba: la vuelve conservadora, no optimista. La versión
pareada de DeLong, DeLong y Clarke-Pearson (1988) la incorpora; se implementó además en su variante rápida
(Sun y Xu, 2014, O(n log n)) porque la versión directa es cuadrática y, con este volumen de datos,
tardaría demasiado, y se validó frente a la versión directa antes de usarla. Ambos entornos comparten también los
mismos tres pliegues de validación cruzada, para que la selección de hiperparámetros se haga sobre los
mismos datos; esto no es un requisito de DeLong, que solo exige evaluar sobre las mismas observaciones de
prueba.

**Métricas independientes de un umbral fijo.** Con una proporción de 80/20 y probabilidades no
calibradas, un umbral de 0.5 no es un punto de operación razonable. El ROC AUC evalúa la capacidad del
modelo de ordenar los casos en todos los umbrales posibles; el umbral que sí se necesita para tomar una
decisión concreta se elige después, con predicciones del propio entrenamiento que el modelo nunca usó para
entrenarse, maximizando F1 como criterio neutral a falta de costos reales de cada tipo de error.

## Limitaciones

- Con 16 GB de RAM, el bosque más grande agotó la memoria con la configuración de histogramas por defecto;
  se usó `maxMemoryInMB = 64`, que no cambia el algoritmo pero sí resta velocidad frente a una máquina con
  más memoria disponible.
- Los tiempos se midieron en una sola máquina (12 hilos lógicos, 16 GB de RAM), con Spark en modo local; la
  comparación no se extiende a un clúster.
- Los hiperparámetros óptimos del bosque, el *gradient boosting* y la regresión logística quedaron en el
  extremo de su malla de búsqueda en ambos entornos: el AUC de 0.72 es una cota inferior de lo alcanzable
  con estas familias de modelos, no necesariamente su máximo.
- `int_rate`, la variable más asociada al incumplimiento, la fija Lending Club con su propio modelo de
  riesgo; los resultados que dependen de ella deben leerse como una predicción condicionada a esa tasa ya
  asignada, no como una evaluación de riesgo construida desde cero.
- No se reporta calibración de las probabilidades (curvas de confiabilidad, *Brier score*) ni variabilidad
  entre semillas de entrenamiento; ambas quedan como extensión de este trabajo.
- La SVM de PySpark (AUC 0.60) es débil, aunque no degenerada como la de scikit-learn; ningún modelo de
  este trabajo alcanza un AUC alto, lo que indica que la información disponible al otorgar el préstamo
  predice el incumplimiento solo moderadamente.
