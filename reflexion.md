# Discusión y conclusiones

Todos los tiempos son de **un solo equipo**: Intel Core i5-13420H (8 núcleos, 12 hilos), 16 GB de RAM,
Windows 11, Spark en modo local (`local[*]`). Nada de lo que sigue se midió en un clúster. Los cuadernos
del libro contienen el código y las salidas completas de cada número citado aquí.

## Costo computacional

**scikit-learn resultó más rápido que PySpark en los seis modelos, por un factor de ≈ 17:** ≈ 1 h 21 min
frente a ≈ 22 h 32 min (preprocesamiento + entrenamiento con validación cruzada + predicción +
transferencia).

| Modelo | scikit-learn | PySpark | Razón |
|---|---|---|---|
| Regresión logística | 15 s | 1 h 03 min | 252× |
| Árbol de decisión | 59 s | 1 h 04 min | 66× |
| Bosque aleatorio | 14 min | 5 h 52 min | 25× |
| *Gradient boosting* | 54 min | 10 h 07 min | 11× |
| SVM lineal | 10 min | 3 h 36 min | 21× |
| Naive Bayes | 5 s | 43 min | 495× |

Dos diferencias de procedimiento favorecen a scikit-learn en esta comparación. `GridSearchCV` reparte los
ajustes de la malla entre los 12 hilos (`n_jobs=-1`), mientras que el `CrossValidator` evaluó un modelo a
la vez (`parallelism = 1`) por restricciones de memoria, lo que le resta ≈ 11 % de velocidad (3b.4). Por
otra parte, en ambos entornos se excluye del total el tiempo de las predicciones fuera de pliegue usadas
para elegir el umbral.

Este resultado puede parecer contraintuitivo, dado que Spark se diseñó para acelerar el cómputo distribuido; las
razones que se pudieron comprobar en el equipo utilizado son:

1. **El tamaño completo del conjunto de datos reduce la brecha, pero no la cierra en una sola máquina.**
   En el experimento de escala, la razón de tiempos entre Spark y scikit-learn **mejora** al crecer los
   datos (de ≈ 625× con 10 000 filas a ≈ 16× con el entrenamiento completo, para un bosque fijo), porque el
   costo fijo de Spark pesa menos cuanto más grande es el lote. Pero incluso con **1.08 millones de filas**
   (el conjunto de entrenamiento completo) Spark sigue siendo más lento.
2. **La caché** (`persist` tras el `VectorAssembler`) sí ayuda: sin ella, un ajuste pequeño tardó ≈ 1.35×
   más. Es, de los elementos de configuración probados, el que **más** beneficia a Spark.
3. **El `CrossValidator`** con 3 pliegues (los mismos de scikit-learn) tardó ≈ 1.4× más que con 2: la
   validación cruzada es cara porque reajusta cada combinación de hiperparámetros una vez por pliegue, y
   Spark hace cada ajuste mucho más lento que scikit-learn (ver el punto 5).
4. **Las 400 particiones fijadas en la sesión** penalizan en el equipo utilizado: un ajuste pequeño con 400
   particiones tardó ≈ 9.4× más que con 12 (las que corresponden a los hilos reales del equipo). Son 400
   tareas diminutas por etapa, con más costo de planificación que de cómputo.
5. **En una sola máquina, Spark no puede repartir el trabajo entre nodos.** Usa los mismos 12 hilos que scikit-learn
   (`n_jobs=-1`), pero paga costos que scikit-learn no tiene: planificar tareas, serializar datos entre
   Python y la JVM y, en los árboles, recorrer los datos una vez por cada nivel de profundidad (así entrena
   Spark ML, pensado para datos que no caben en un nodo).
6. **Memoria.** Con 16 GB de RAM, `maxMemoryInMB = 64` (en vez del valor por defecto, 256) y
   `parallelism = 1` en el `CrossValidator` fueron necesarios para que el bosque más grande no agotara la
   memoria; ninguno de los dos cambia el algoritmo, pero **restan velocidad** frente a configuraciones con
   más memoria disponible (≈ 11 % en el caso de `parallelism = 1`; 3b.4).

```{note}
Con 12 particiones en vez de las 400 de la sesión, el mismo bosque de referencia tarda ≈ 69 s frente a
≈ 258 s: la brecha se reduce de ≈ 16× a ≈ 4× frente a scikit-learn (16 s), pero **no desaparece**.
```

## Equivalencia predictiva

**Depende del modelo.** Cinco de los seis dan prácticamente lo mismo en los dos entornos; el sexto (la SVM)
es una excepción notable:

| Modelo | AUC scikit-learn | AUC PySpark | ΔAUC | ¿Relevante? (≥ 0.005) |
|---|---|---|---|---|
| Regresión logística | 0.7095 | 0.7095 | +0.00000004 | No |
| Árbol de decisión | 0.7022 | 0.7004 | +0.0018 | No |
| Bosque aleatorio | 0.7164 | 0.7152 | +0.0012 | No |
| ***Gradient boosting*** | **0.7196** | **0.7189** | +0.0007 | No |
| **SVM lineal** | **0.4956** | **0.5985** | **−0.1029** | **Sí** |
| Naive Bayes | 0.6884 | 0.6883 | +0.0001 | No |

Para los otros cinco modelos, las diferencias son estadísticamente significativas en casi todos los
casos (269 062 observaciones de prueba permiten detectar diferencias muy pequeñas), pero **ninguna alcanza el margen de
relevancia práctica** (ΔAUC ≥ 0.005, fijado antes de ver resultados). El mejor modelo de los doce es
***gradient boosting***, en ambos entornos, con AUC ≈ 0.72.

**La SVM es la única excepción relevante:** en scikit-learn el AUC es 0.4956 (equivalente al azar) y en
PySpark 0.5985 (débil pero real). La diferencia (ΔAUC = 0.103) es la única de las **seis** comparaciones
entre entornos que supera el margen de relevancia práctica. La causa no es que "PySpark entienda mejor los
datos": es que `LinearSVC(loss="hinge")` de scikit-learn **converge a una solución degenerada** (ver la
sección de diferencias de implementación).

## Significancia frente a relevancia

**Se hicieron dos familias de comparaciones DeLong**, corregidas con Holm dentro de cada una:

- **Entre entornos (6 comparaciones, el mismo modelo en los dos motores):** 5 de 6 son estadísticamente
  significativas tras Holm (todas menos la regresión logística, p ajustado = 0.48). **Solo 1 de las 6 es
  relevante en la práctica: la SVM** (ΔAUC = −0.103). Las otras cuatro significativas (árbol, bosque aleatorio, *gradient boosting*, Naive Bayes) tienen ΔAUC entre 0.0001 y 0.0018: estadísticamente reales,
  prácticamente irrelevantes.
- **Entre modelos, dentro de cada entorno (15 pares por entorno):** aquí ocurre lo contrario. Los seis
  modelos difieren bastante entre sí en capacidad predictiva, así que **14 de los 15 pares por entorno
  superan el margen de relevancia práctica**; la única excepción es el bosque aleatorio frente al *gradient
  boosting* (el mejor par de cada entorno, ΔAUC ≈ 0.003–0.004: significativo, pero por debajo del margen de
  0.005). Es decir, distinguir *modelos* entre sí casi siempre importa en la práctica; distinguir el
  *mismo* modelo entre *motores* casi nunca importa, salvo por la SVM.

**Conclusión:** con 269 062 observaciones de prueba, la significancia estadística casi no distingue entre
"el mismo modelo, dos motores" y "modelos distintos": prácticamente todo resulta significativo. El **margen de relevancia
práctica** es lo que separa lo que importa (las diferencias entre modelos, y la SVM entre motores) del
ruido propio de un conjunto de datos grande (las diferencias mínimas del mismo modelo entre motores).

## Diferencias de implementación

1. **`LinearSVC(loss="hinge")` de scikit-learn converge a una solución degenerada (la diferencia más
   grande).** Se investigó en detalle (sección 2.5): en un ajuste de diagnóstico con 700 000 filas, el
   *solver* dual de `liblinear` converge en 93 iteraciones, sin advertencias, a coeficientes ≈ 10⁻⁸ y un
   intercepto de −1.0, y el colapso se repite con los tres valores de `C` de la malla: predice "paga" para
   casi todo, sin separar clases. Es compatible con la solución degenerada de la SVM de pérdida *hinge*
   descrita por Rifkin, Pontil y Verri (1999) para clases solapadas y desbalanceadas, reforzada porque
   `liblinear` también regulariza el intercepto. El `LinearSVC` de PySpark no regulariza el intercepto y
   usa otro optimizador (OWL-QN; Andrew y Gao, 2007), que no muestra el mismo colapso y da un AUC débil
   pero real (0.60); ninguna de las dos diferencias se aisló por separado. No se cambió `loss` porque
   `hinge` es la pérdida que minimiza el `LinearSVC` de PySpark y la que hace comparables ambos modelos.
2. **Discretización de los árboles (`maxBins`).** PySpark corta cada variable continua en como máximo 32
   intervalos (`maxBins = 32`, por defecto, reportado en los tres modelos de árbol); scikit-learn evalúa
   cortes **exactos**. Es una posible explicación de que el árbol de decisión y el bosque de scikit-learn
   tengan un AUC ligeramente mayor: pueden encontrar el punto de corte óptimo, mientras PySpark elige entre
   32 candidatos. El efecto es pequeño (ΔAUC ≤ 0.002) y no se aisló con un experimento aparte.
3. **La relación `C = 1 / (regParam × n)`** entre scikit-learn y PySpark es **aproximada**: `C` pondera la
   suma de las pérdidas y `regParam` su promedio. Esto afecta a la regresión logística y a la SVM; para la
   logística el efecto fue mínimo (AUC casi idéntico), para la SVM quedó opacado por la solución degenerada del punto 1.
4. **`NaiveBayes` gaussiano en ambos entornos** evita el problema de valores negativos que tendrían las
   variantes multinomial/Bernoulli con datos escalados; el resultado es prácticamente idéntico entre
   motores (ΔAUC = 0.0001), como se esperaría de un modelo con una fórmula cerrada y sin optimización
   iterativa de por medio.

## Concordancia entre las tres pruebas

**Limitaciones de DeLong, la prueba principal:**

1. Solo mide la incertidumbre del **muestreo del conjunto de prueba**; no incluye la del entrenamiento (semillas,
   pliegues, modelos estocásticos como el bosque o el *boosting*).
2. Supone observaciones **independientes** (préstamos de la misma época o estado comparten contexto).
3. El AUC resume **todos los umbrales**, y puede no reflejar el umbral operativo elegido.

**¿Coinciden DeLong, McNemar y el *bootstrap*?** En general sí, con **dos discrepancias** entre
DeLong y McNemar, y un patrón más disperso en el *bootstrap* de F1 y AUC-PR:

- **Regresión logística:** DeLong dice que **no** hay diferencia significativa (p ajustado = 0.48,
  ΔAUC ≈ 0), y el *bootstrap* de AUC tampoco (coincide con DeLong, como debía). Pero **McNemar sí encuentra
  una diferencia significativa** (p ≈ 3×10⁻¹¹⁹) en las decisiones concretas (umbral fijo): de 269 062
  préstamos, 318 los acierta solo scikit-learn y 1237 solo PySpark. Los dos modelos ordenan los préstamos
  casi igual, por eso el AUC no distingue, pero como cada uno usa un umbral ligeramente
  distinto (0.2042 en scikit-learn frente a 0.2061 en PySpark, elegidos por separado con F1 fuera de
  pliegue), **clasifican de forma algo distinta a los préstamos que quedan cerca del umbral**. Es
  precisamente la limitación 3 de DeLong: resume todos los umbrales y no ve esto. Como ambos umbrales provienen de
  mallas discretas, parte de esa diferencia puede deberse a la resolución de la malla y no a los modelos.
- **Árbol de decisión:** al revés. **DeLong sí** encuentra diferencia significativa (p ajustado = 0.003) y
  **McNemar no** (p = 0.058, con Holm). El AUC del árbol de scikit-learn es algo mayor de forma consistente
  en todo el rango de umbrales (por los cortes exactos, sección anterior), pero en el umbral operativo
  concreto las decisiones no difieren lo suficiente para que McNemar lo detecte.
- **El *bootstrap* de AUC coincide con DeLong en las seis comparaciones entre entornos:** ambas pruebas
  miden lo mismo (el AUC) con métodos distintos (asintótico vs. remuestreo). Que coincidan es evidencia de
  que ninguna de las dos está fallando.
- **El *bootstrap* de AUC-PR y F1 no siempre coinciden entre sí, ni con el AUC.** De los seis modelos, el
  F1 no resulta significativo en cuatro (regresión logística, árbol, *gradient boosting*, Naive Bayes) y el
  AUC-PR no resulta significativo en dos, pero **no los mismos dos**: regresión logística y bosque aleatorio.
  La SVM es el único modelo en el que las tres métricas (AUC, AUC-PR y F1) coinciden en ser
  significativas. Esto es consistente con que F1 y AUC-PR dependen del umbral y de la clase minoritaria de
  forma distinta al AUC: no hay razón para esperar que las tres métricas siempre estén de acuerdo.

**En conjunto:** las tres pruebas coinciden en lo más importante (la SVM es la única diferencia grande y
consistente en las tres métricas); donde discrepan, la razón siempre fue identificable (umbral vs. orden
completo, o sensibilidad distinta a la clase minoritaria), no un error de las pruebas.

## Escalabilidad

**En el equipo utilizado, en ninguno de los volúmenes probados**, ni con las 400 particiones de la sesión ni con
una variante de 12, PySpark llegó a superar a scikit-learn. Se ajustó un bosque fijo (50 árboles,
profundidad 10) con 10 000 hasta 1 076 248 filas:

| Versión | Costo fijo | Costo por millón de filas | Con el entrenamiento completo |
|---|---|---|---|
| scikit-learn | ≈ 0 s | ≈ 15 s | 16 s |
| PySpark, 400 particiones (capítulo 3) | ≈ 170 s | ≈ 86 s | 258 s |
| PySpark, 12 particiones (variante) | ≈ 12 s | ≈ 54 s | 69 s |

El **costo por fila** de Spark es mayor que el de scikit-learn en ambas configuraciones: las rectas de
tiempo **no se cruzan**, ni medido ni extrapolando linealmente. La pregunta solo tendría una respuesta real
en dos situaciones que este trabajo no puede probar: **(a)** datos que no caben en la RAM de una máquina
(scikit-learn requeriría estrategias fuera de memoria) o **(b)** un **clúster** con muchos nodos, donde el trabajo sí se
reparte entre máquinas. Cualquier cifra concreta de "cruce" para esos casos sería especulación.

## Interpretabilidad

**Qué aporta LIME.** Se explicaron los **mismos dos préstamos** con el mejor modelo de probabilidades de
cada entorno (*gradient boosting* en los dos):

- **Falso negativo** (préstamo 36099806, incumplió; ambos modelos daban ≈ 0.02 de probabilidad): las
  variables que más empujaban hacia "paga" fueron consistentes entre los dos entornos.
- **Falso positivo** (préstamo 55961482, se pagó; ambos modelos daban ≈ 0.83–0.86): igual, coherencia entre
  entornos en las variables principales.
- Promediando 60 explicaciones, cuatro de las cinco variables más influyentes (`int_rate`, `term_months`,
  `fico_range_high` y `dti`) coinciden con las cinco principales de la importancia por permutación del
  bosque aleatorio, calculada de forma independiente, aunque el orden difiere.

**Sus límites (medidos en este trabajo):**

1. **Fidelidad parcial:** el R² del modelo lineal local no es 1 (entre 0.50 y 0.58 en scikit-learn, entre
   0.51 y 0.55 en PySpark, según el préstamo): la explicación imita al modelo solo en parte.
2. **Estabilidad frente a la semilla:** para el falso negativo, se repitió la explicación con 5 semillas
   distintas y las 5 principales variables salieron **exactamente en el mismo orden** las cinco veces. Es
   una sola instancia y cinco repeticiones, así que no prueba que LIME sea siempre estable aquí; solo que,
   en este caso concreto, la aleatoriedad del muestreo no cambió la conclusión.
3. **Perturbaciones sin correlaciones:** LIME muestrea cada variable por separado, puede crear préstamos
   imposibles (p. ej. tasa muy baja con FICO muy bajo), sobre los que el modelo nunca fue entrenado.
4. **En PySpark, LIME opera en memoria local** y llama al modelo miles de veces. Con Apache Arrow, cada
   llamada del modelo final de Spark tardó ≈ 0.5 s con 5 000 filas: viable para explicar unas pocas
   instancias, no millones.
5. **Sin Arrow, una llamada tardaba ≈ 8 minutos** en una ejecución preliminar (400 particiones): con esa
   latencia, LIME sería inviable en la práctica. Se comprobó que la causa era el envío del lote sin Arrow,
   no el tamaño del modelo (capítulo 3b).
6. **Datos de referencia locales:** LIME necesita traer una muestra al *driver* (aquí, filas de
   entrenamiento) para conocer las distribuciones, lo que va en contra de la idea de no mover el conjunto de datos
   completo al *driver*.

## Efecto de la configuración de Spark

| Elemento de la configuración | Efecto medido | Evidencia |
|---|---|---|
| **Conjunto de datos completo, sin muestreo** | Ayuda a Spark relativamente (el costo por fila pesa más que el fijo a mayor volumen) pero no basta: Spark sigue siendo más lento con el conjunto completo | 3b.5, 6.4 |
| **Los mismos seis modelos y espacios de búsqueda en ambos entornos** | Permite la comparación pareada (DeLong); reveló la falla de la SVM de scikit-learn, invisible si solo se hubiera probado un modelo | 2, 3, 5 |
| **`ParamGridBuilder` + `CrossValidator`, sin bucles manuales** | Reproducible y correcto, pero es la etapa más costosa: sola representa la mayoría del tiempo total de Spark | 6.3 |
| **`numFolds = 3` (igual que scikit-learn)** | Cuesta ≈ 1.4× más que 2 pliegues; con `foldCol`, hace que la selección de hiperparámetros use exactamente los mismos pliegues que scikit-learn | 3b.4 |
| **`spark.default.parallelism` / `shuffle.partitions = 400`** | **Empeoró** el rendimiento en el equipo utilizado: ≈ 9.4× más lento que con 12 particiones (las de los hilos reales) | 3b.4 |
| **`spark.executor.memory = 8g`** | **Sin efecto en modo local**: no hay ejecutores separados; cuenta `spark.driver.memory` | 3.1 |
| **`spark.driver.memory`, `memory.fraction`, `storageFraction`** | Fijan el límite real de memoria (el heap verificado fue 8.6 GB); no se aislaron con experimentos separados | 3.1 |
| **Caché (`.persist()`) tras el `VectorAssembler`** | Reduce el tiempo de un ajuste ≈ 1.35×; con el `CrossValidator`, que reutiliza los datos decenas de veces, el ahorro acumulado es mayor | 3b.4 |
| **Ninguna transferencia de datos al driver antes de entrenar** | Solo se transfirieron agregados diminutos (conteos por categoría, sumas para el umbral); la única transferencia grande (id, etiqueta, puntuación del conjunto de prueba) ocurrió **después** de entrenar y se midió aparte (< 10 s en total para los seis modelos) | 3, 6.3 |

## Conclusiones

1. Con implementaciones equivalentes, la misma partición y los mismos pliegues, scikit-learn y PySpark
   producen modelos de calidad prácticamente idéntica: en cinco de los seis modelos, las diferencias de
   AUC entre motores (≤ 0.0018) son detectables con 269 062 observaciones de prueba, pero quedan muy por
   debajo del margen de relevancia práctica.
2. La única diferencia relevante entre motores es la de la SVM lineal (ΔAUC = −0.103), atribuible a la
   solución degenerada de la pérdida *hinge* en `liblinear` y no a una ventaja general de uno de los
   motores.
3. Entre modelos, en cambio, casi todas las diferencias son relevantes. El *gradient boosting* es el mejor
   en ambos entornos (AUC ≈ 0.72), seguido del bosque aleatorio, del que no se distingue en la práctica.
4. En una sola máquina de 12 hilos, scikit-learn es ≈ 17 veces más rápido. El mayor volumen reduce la
   brecha, pero el costo por fila de Spark sigue siendo mayor, así que no hay un punto de cruce en el
   equipo utilizado; la ventaja de Spark queda reservada a datos que no caben en memoria o a un clúster.
5. LIME produce explicaciones coherentes entre motores para los mismos préstamos, pero con fidelidad local
   moderada (R² entre 0.50 y 0.58) y un costo por llamada en Spark que lo limita a unas pocas instancias.
6. La información disponible al otorgar un préstamo predice el incumplimiento solo moderadamente, y una
   parte importante de ella proviene de la tasa de interés que asigna la propia plataforma.

## Alcance de las conclusiones

Los resultados de este trabajo no permiten afirmar:

- Que Spark sea "más lento que scikit-learn" en general: solo que lo fue **aquí**, en una máquina con 12
  hilos y 16 GB, con estos datos y esta configuración de sesión.
- Nada sobre el desempeño en un clúster real.
- El efecto individual de `spark.memory.fraction` y `storageFraction` (no se aislaron).
- Que el AUC ≈ 0.72 del *gradient boosting* sea el máximo alcanzable con estos datos.
- Por qué exactamente `maxBins = 32` produce el ΔAUC observado en los árboles: es la explicación más
  plausible, pero no se aisló con un experimento que solo cambiara `maxBins`.

## Referencias

- Andrew, G. y Gao, J. (2007). Scalable training of L1-regularized log-linear models. En *Proceedings of the 24th International Conference on Machine Learning (ICML)*, 33–40.
- Bergmeir, C. y Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192–213.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2.ª ed.). Lawrence Erlbaum.
- DeLong, E. R., DeLong, D. M. y Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*, 44(3), 837–845.
- Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895–1923.
- Efron, B. y Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Fan, R.-E., Chang, K.-W., Hsieh, C.-J., Wang, X.-R. y Lin, C.-J. (2008). LIBLINEAR: A library for large linear classification. *Journal of Machine Learning Research*, 9, 1871–1874.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N. y Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.
- Meng, X. *et al.* (2016). MLlib: Machine learning in Apache Spark. *Journal of Machine Learning Research*, 17(34), 1–7.
- Pedregosa, F. *et al.* (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Ribeiro, M. T., Singh, S. y Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. En *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144.
- Rifkin, R., Pontil, M. y Verri, A. (1999). A note on support vector machine degeneracy. En *Algorithmic Learning Theory (ALT 1999)*, Lecture Notes in Computer Science 1720, 252–263. Springer.
- Sun, X. y Xu, W. (2014). Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. *IEEE Signal Processing Letters*, 21(11), 1389–1393.
