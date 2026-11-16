# Laboratorio 10: investigación de XAI para FraudLens

El dato es tabular: V1-V28 son componentes PCA anónimos y la entrada restante es
log(1 + Amount). El modelo final es un MLP PyTorch con calibración Platt. Se
investigan métodos que explican entradas tabulares; Grad-CAM, diseñado para mapas
de activación espaciales, no corresponde a esta arquitectura.

## Métodos considerados

Los costos son orientativos por instancia, no tiempos medidos: d = número de
variables, K = puntos de integración, B = filas de fondo, S = coaliciones o
perturbaciones. Una evaluación con gradientes cuesta más que una inferencia.

| Método / herramienta | Idea y requisitos | Costo y limitaciones relevantes |
| --- | --- | --- |
| Integrated Gradients / `captum` | Integra el gradiente entre una referencia y la entrada. Requiere un modelo diferenciable y una referencia explícita. | K evaluaciones con gradientes; depende de referencia y trayectoria. Los puntos interpolados pueden ser artificiales. [1,2] |
| LIME / `lime` | Ajusta un sustituto sencillo ponderando predicciones de un vecindario. Solo necesita consultas de caja negra, un muestreador y un kernel de distancia. | S inferencias y ajuste del sustituto. El kernel, el muestreo, la discretización y la semilla cambian la explicación. La fidelidad se refiere al vecindario elegido, no al modelo entero. [3] |
| KernelSHAP / `shap` | Estima valores de Shapley mediante regresión ponderada sobre coaliciones; necesita caja negra y un fondo para reemplazar variables ausentes. | Aproximadamente S x B inferencias; caro para muchos casos. El fondo y el tratamiento de dependencias cambian el juego explicado. Combinaciones de variables pueden ser irreales. [4,5] |
| TreeSHAP / `shap` | Aprovecha la estructura de árboles para calcular atribuciones aditivas de forma eficiente. Necesita acceso a árboles y una política de perturbación/fondo. | Costo polinómico especializado. No es compatible directamente con el MLP final; entrenar un árbol sustituto explicaría otro modelo. [5] |
| DeepSHAP / `shap` | Propaga contribuciones tipo DeepLIFT y las combina sobre un fondo para aproximar valores SHAP. Requiere una red y operaciones compatibles. | Crece con B y las pasadas de propagación. Es una aproximación dependiente del fondo y de las reglas para las operaciones de la red. [4,5] |
| GradientSHAP / `shap`, `captum` | Promedia gradientes sobre referencias y puntos de trayectorias muestreadas; algunas variantes añaden ruido. Requiere gradientes y una distribución de fondo. | Número de muestras con gradientes. Añade variabilidad de Monte Carlo y supuestos para aproximar atribuciones SHAP. No equivale a IG de una sola referencia. [2,5] |
| Contrafactuales DiCE / `dice-ml` | Busca varias entradas próximas que obtengan la clase deseada. Requiere consultas o gradientes y restricciones de mutabilidad/factibilidad. | Búsqueda iterativa; el costo depende del algoritmo y restricciones. No identifica causas. En PCA no sabemos qué acciones producirían un cambio factible de componente. [6] |
| Anchors / `alibi` | Encuentra reglas suficientes para mantener una predicción con alta precisión bajo un muestreador. Necesita caja negra y distribución de perturbación. | Búsqueda y consultas adaptativas, potencialmente costosas. Precisión y cobertura corresponden a esa distribución; una regla en PCA sigue sin significado de negocio. [7] |
| Oclusión / `captum` o PyTorch | Sustituye entradas por una referencia y observa el cambio de salida. Solo requiere caja negra y estrategia de reemplazo. | d + 1 inferencias para oclusión univariada. Las diferencias no forman una descomposición aditiva y no resuelven interacciones; pueden salir de la distribución. [2] |

## Interpretable por diseño y explicación a posteriori

Una regresión logística permite inspeccionar su función mediante coeficientes;
la explicación es parte de su estructura. IG explica después un MLP ya fijado.
FraudLens ya tiene una logística entrenada, con AP de test 0.6946 frente a 0.8222
de A, recall 0.7297 frente a 0.8378 y F1 0.6968 frente a 0.6263. Los umbrales
permanecen en los valores elegidos originalmente con validación. La interpretación
estructural de los coeficientes no devuelve el significado perdido por PCA.
La alternativa simple merece preferencia si alcanza resultados operativos
comparables y facilita la supervisión; esta corrida no demuestra equivalencia.

## Método principal y propiedades formales

Se selecciona IG por compatibilidad directa con PyTorch, descomposición firmada,
referencia verificable y costo acotado por puntos de integración. Se explica
F(z) = a * MLP(z) + b, el logit calibrado, y se muestran las probabilidades
sigmoid(F). Evitar la sigmoide en la atribución reduce el problema de saturación;
no cambia la regla de decisión. Las contribuciones se expresan en log-odds.

Para una entrada x y referencia b:

`IG_j(x,b) = (x_j - b_j) * integral_0^1 dF(b + alpha*(x-b))/dx_j d alpha`.

La **completitud** establece que la suma de las atribuciones es F(x) - F(b)
para la integral exacta. Permite comprobar contabilidad de la diferencia de
puntajes. **Invariancia de implementación** exige la misma atribución a funciones
equivalentes con la misma referencia/trayectoria. **Sensibilidad** impide atribuir
cero a la única variable que produce una diferencia de salida bajo las condiciones
del axioma. La aproximación numérica se comprueba con un residuo, que no demuestra
exactitud de cada componente ni causalidad. [1]

En SHAP, la exactitud local o eficiencia de la descomposición requiere que el valor
base y las contribuciones sumen la salida del juego definido. Depende de cómo se
modela la ausencia de variables. En LIME, minimizar el error local ponderado no
garantiza una buena explicación fuera del vecindario ni una fidelidad perfecta.

## Parámetros y decisiones de diseño

- **IG:** referencia media de train (cero tras el escalador guardado), trayectoria
  recta en las 29 entradas transformadas, salida logit calibrada, Gauss-Legendre,
  256 puntos y tolerancia documentada. Sensibilidad con mediana legítima de train
  y una fila legítima real próxima a ella. No hay fondo SHAP en el método principal.
- **SHAP:** tamaño/composición del fondo, función de enmascaramiento, tratamiento
  de dependencias, número de coaliciones y escala de la salida. Son opciones de
  la alternativa investigada, no parámetros usados en los resultados IG.
- **LIME:** ancho del kernel, distancia, tamaño del vecindario, discretización,
  complejidad del sustituto y semilla. También corresponden a una alternativa.
- **Oclusión:** valor de reemplazo y variables/grupos sustituidos; aquí se prueba
  solo reemplazo individual por la media de train para diagnóstico.

Se mantiene IG tras usarlo: el ranking principal conserva parte importante de sus
variables bajo referencias distintas, pero C2 baja a 3/5 coincidencias. Por tanto,
no se comunica una lista única de razones como si fuera una verdad del fenómeno.
KernelSHAP sería útil sin gradientes o para unificar explicaciones entre familias,
con mayor costo de consultas y una elección de fondo que también debe auditarse.

## Fuentes primarias consultadas (16 de noviembre de 2026)

1. Sundararajan, Taly y Yan (2017), *Axiomatic Attribution for Deep Networks*.
   https://proceedings.mlr.press/v70/sundararajan17a.html
2. Captum, documentación de Integrated Gradients y algoritmos de atribución.
   https://captum.ai/api/integrated_gradients.html
   https://captum.ai/api/occlusion.html
   https://captum.ai/api/gradient_shap.html
3. Ribeiro, Singh y Guestrin (2016), *Why Should I Trust You?*.
   https://arxiv.org/abs/1602.04938
4. Lundberg y Lee (2017), *A Unified Approach to Interpreting Model Predictions*.
   https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
5. SHAP, documentación de KernelExplainer, TreeExplainer, DeepExplainer y GradientExplainer.
   https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html
   https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
   https://shap.readthedocs.io/en/latest/generated/shap.DeepExplainer.html
   https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html
6. Mothilal, Sharma y Tan (2020), *Explaining Machine Learning Classifiers through Diverse Counterfactual Examples*.
   https://www.microsoft.com/en-us/research/publication/explaining-machine-learning-classifiers-through-diverse-counterfactual-examples/
7. Ribeiro, Singh y Guestrin (2018), *Anchors: High-Precision Model-Agnostic Explanations*;
   documentación del método en Alibi.
   https://cdn.aaai.org/ojs/11491/11491-13-15019-1-2-20201228.pdf
   https://docs.seldon.ai/alibi-explain/explanations/methods/anchors

Las cifras del proyecto proceden de `results/metrics.json` y las observaciones del
laboratorio de `explicabilidad/resultados/`; no proceden de los artículos.
