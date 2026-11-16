# Laboratorio 10: Explainable AI / FraudLens

Este módulo utiliza el modelo final **A**, su escalador y su calibrador guardados.
El programa no contiene entrenamiento ni modifica estos artefactos. Guarda sus
hashes antes y después, comprueba paridad de probabilidades y conserva todas las
decisiones del test original.

## Regeneración

Desde la raíz del mismo repositorio del proyecto final, con Python 3.12:

```bash
python -m pip install -r requirements-xai.txt
python -m scripts.download_data
python -m explicabilidad.analizar
python -m explicabilidad.informe
python -m explicabilidad.verificar
python -m unittest discover -s tests -v
```

El análisis reconstruye el manifiesto de particiones si no existe, siguiendo la
deduplicación y los cortes originales y comprobando tamaños y hash. No necesita
reentrenar el modelo ni generar `test.csv`. El CSV descargado se conserva fuera
de Git conforme a la atribución/licencia de `data/README.md`. Las referencias y
tablas ampliadas están en [investigacion.md](investigacion.md).

Agregar el enlace real al final del PDF:

```bash
python -m explicabilidad.informe --repo https://github.com/USUARIO/REPOSITORIO
python -m explicabilidad.verificar --entrega
```

También puede guardarse el enlace en `repository_url` de `config.json` para
regeneraciones futuras. Estos comandos no publican el repositorio.

## Diseño de la explicación

- Entrada: V1-V28 y log(1 + Amount), con el escalador original de train. Time,
  source_row y Class no son entradas del modelo.
- Objetivo: `F(z) = a * MLP_A(z) + b`, el logit calibrado. La probabilidad mostrada
  es `sigmoid(F)`. Una barra positiva eleva F; una negativa lo reduce. **Las
  barras no son porcentajes.**
- Línea base principal: vector cero tras escalar, es decir, media de train en
  las entradas transformadas. Es reproducible y solo usa train; puede ser una
  transacción sintética. En Amount, no corresponde a monto cero, sino a
  `exp(media(log(1+Amount))) - 1`.
- Trayectoria recta, `multiply_by_inputs=True`, Gauss-Legendre, 256 puntos.
  Tolerancia `abs(delta) <= 0.02 + 0.001 * abs(F(x)-F(base))`; refinamiento hasta
  1,024 puntos si hace falta. Todos los casos principales usaron 256. El control
  de completitud solo comprueba la suma, no la precisión de cada componente.
- Semilla 23045; CPU, cuatro hilos; lotes de 32 ejemplos y batch interno de 2,048
  puntos. `config.json` contiene todos los parámetros configurables.
- Sensibilidad local: mediana por variable de los train legítimos y fila legítima
  real de train más próxima a esa mediana en distancia euclídea estandarizada
  (source_row 55846). Las coordenadas exactas están en `referencias.json`.
- No se usa conjunto de fondo SHAP. Los métodos alternativos solo se investigan;
  no se afirma haber ejecutado LIME, KernelSHAP, DiCE ni Anchors.

## Casos y agregación

Las reglas se fijan en `choose_cases` antes de observar atribuciones. Se toman
cinco casos distintos de test en orden: VP con p máxima, FP con p máxima, FN con
p mínima, caso restante más cercano al umbral y VP restante de monto > 100 con p
máxima. Los empates conservan el orden temporal/source_row del manifiesto. Las
etiquetas solo permiten seleccionar y evaluar casos para esta auditoría.

La muestra global aleatoria de 1,024 filas sin reemplazo contiene tres fraudes.
Su media de |IG| describe principalmente el tráfico legítimo. Por eso se muestra
aparte un panel de los 74 fraudes de test, sin mezclarlo ni ponderarlo como si
fuera una muestra de la población. Se explican 1,097 filas únicas al unir
muestra, fraudes y casos locales. Los cinco casos son ilustrativos, no una
estimación de frecuencia de errores.

## Archivos y trazabilidad

| Archivo/carpeta | Contenido |
| --- | --- |
| `analizar.py` | Inferencia congelada, IG, referencias, sustituciones, auditoría y latencia. |
| `figuras.py` | Gráficas locales, globales y sensibilidad, derivadas de los resultados. |
| `informe.py` | Generador del PDF de cuatro páginas y del texto editable `informe.md`. |
| `verificar.py` | Integridad de modelos, paridad, completitud, cobertura de casos y PDF. |
| `resultados/casos.csv` | IDs, reglas de selección, etiquetas, puntajes, márgenes y deltas. |
| `resultados/entradas_locales.csv` | Entradas originales de las cinco transacciones. |
| `resultados/entradas_estandarizadas.csv` | Entradas exactas al MLP. |
| `resultados/atribuciones_locales.csv` | Las 29 atribuciones firmadas de cada caso. |
| `resultados/importancia_global.csv` | Media absoluta y firmada; grupos sin mezclar. |
| `resultados/muestra_global.csv` | IDs y etiquetas de la muestra aleatoria. |
| `resultados/atribuciones.npz` | Todas las atribuciones y sus índices, pasos y deltas. |
| `resultados/sensibilidad_referencia.csv` | Top 5 y Spearman de |IG| con referencias alternativas. |
| `resultados/referencias.json` | Los tres vectores de referencia. |
| `resultados/sustituciones_diagnosticas.csv` | Reemplazo univariado por la media para los cinco casos. |
| `resultados/subgrupos.csv` | Métricas por monto, con denominadores. |
| `resultados/coincidencias_train_test.csv` | IDs de las 648 entradas de test repetidas en train. |
| `resultados/auditoria.json` | Parámetros, hashes, versiones, costo y comprobaciones. |
| `figuras/` | Cada figura en PNG y SVG. |

Las tablas numéricas del PDF provienen de `casos.csv`, `auditoria.json`,
`importancia_global.csv`, `subgrupos.csv` y `sustituciones_diagnosticas.csv`.
Las tablas de investigación y selección se regeneran en `informe.py` y tienen
fuentes en `investigacion.md`. Las métricas de la logística/A se leen o cotejan
con `results/metrics.json` del proyecto original.

El benchmark mide solo el cálculo en memoria: 30 repeticiones tras cinco
calentamientos, CPU, un ejemplo, y por separado el lote de 1,097 filas. No incluye
red, lectura de archivos, preproceso ni figuras. Los tiempos varían con equipo,
concurrencia y cachés; no se espera reproducibilidad bit a bit de latencias.
Las atribuciones y la muestra sí se regeneran con semilla fija y dependencias
fijadas; puede existir variación numérica pequeña entre plataformas.

## Interpretación y límites

V4 domina en la muestra global y V14 en fraudes. La falsa alarma C2 comparte
señales con los aciertos; cambiar V14 por su media cruza el umbral en una entrada
sintética. En C3 ninguna sustitución individual lo consigue. Esto no constituye
un contrafactual causal ni acciones que deba realizar un titular.

La auditoría encontró 648 filas de test con entradas idénticas a train sin Time
(cero fraudes entre ellas), confirmadas por igualdad exacta de columnas. La
deduplicación original incluía Time. No se cambian los datos ni se reentrena aquí;
se documenta la limitación y se propone resolverla en un futuro ciclo con
particiones por entidad. No se conocen identidades de tarjetas ni el ajuste
original de PCA: no se puede asegurar ausencia de proxies/fuga previa.

No se atribuye significado a los componentes PCA ni se presenta el prototipo como
apto para decisiones reales sobre personas/pagos. Las recomendaciones son revisión
humana, mejor información de dominio y validación independiente de las políticas.
