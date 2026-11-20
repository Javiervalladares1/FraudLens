# Verificación reproducible de la demo

Este diagnóstico permite comprobar los archivos que utiliza FraudLens antes de
presentarlo. Ejecuta la misma inferencia JavaScript de la aplicación, muestra cuatro
tipos de resultado y revisa que los conteos y valores económicos coincidan con los
resultados guardados. Utiliza el experimento original con el modelo A seleccionado.

## Ejecutarlo

Se necesita Node.js 18 o posterior. No requiere npm, PyTorch, descargar el CSV ni
reentrenar las redes. Desde la raíz del repositorio:

```bash
node scripts/check_demo.mjs
```

El comando termina con código `0` si todo coincide, y código `1` y un mensaje
`VERIFICACIÓN FALLIDA` si detecta un problema. Solo lee archivos y escribe en la
terminal. Las rutas se resuelven desde el script, por lo que no dependen del
directorio desde el que se ejecute.

## Qué se comprueba

- La versión seleccionada y los supuestos son coherentes entre `model.json` y
  `metrics.json` de `site/dist`.
- Todos los puntajes de test de A, B y LR son números finitos entre 0 y 1; sus
  etiquetas son binarias. Los conteos, precisión, recall, F1 y valor mensual se
  recalculan y comparan con las métricas guardadas.
- Los umbrales de A y B coinciden entre los pesos exportados y las métricas.
- Cuatro transacciones contienen exactamente V1 a V28 y Amount. Sus predicciones
  se comparan con referencias guardadas para ambas redes, con tolerancia de
  `0.00002` en probabilidad.
- Se rechazan siete entradas inválidas, incluidos JSON incompleto, valores no
  finitos, texto en una variable numérica y monto negativo.
- Duplicar el volumen conserva el alojamiento fijo; el umbral 1 no genera alertas
  en este test y mantiene el costo de operación. También se comprueba el umbral 0.5.

Los cuatro ejemplos están en `tests/demo_cases.json`. La etiqueta se guarda fuera
de `features`; nunca entra a `predict` ni a la red.

## Resultados esperados

| Modelo | VP | FP | FN | VN | Valor mensual estimado |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 62 | 62 | 12 | 56610 | US$12662.02 |
| B | 62 | 101 | 12 | 56571 | US$12421.48 |
| LR | 54 | 27 | 20 | 56645 | US$11193.20 |

En el escenario base, A con umbral 0.5 estima US$10727.96. Con su umbral guardado
y el doble de volumen estima US$25334.04. Sin alertas conserva un costo mensual de
US$12. Son escenarios hipotéticos, no dinero ahorrado en una institución.

| Caso | Etiqueta de referencia | Puntaje aproximado en A | Interpretación en A |
| --- | --- | ---: | --- |
| TX-227820 | Legítima | 0.0010% | Verdadero negativo |
| TX-229712 | Fraude | 97.6432% | Verdadero positivo |
| TX-252720 | Legítima | 33.5679% | Falso positivo |
| TX-274382 | Fraude | 0.0052% | Falso negativo |

Para mostrar un caso en la web, copia únicamente el objeto `features` de ese
ejemplo y pégalo en el campo de transacción JSON. No pegues el archivo completo ni
el campo `label`. La etiqueta de referencia permite explicar el resultado; el
modelo por sí solo no confirma que una transacción sea fraudulenta.

## Cómo explicar esta aportación

El entrenamiento aprende los pesos en Python. La aplicación usa esos pesos
exportados para predecir en JavaScript. Este verificador importa directamente
`predict` y `summarize` de `site/dist/inference.mjs`, de modo que comprueba el
cálculo que utiliza la web sin duplicar el motor de inferencia.

La comparación de una predicción con la etiqueta produce cuatro resultados.
Un verdadero positivo es un fraude con alerta; un verdadero negativo es un pago
legítimo sin alerta. Un falso positivo molesta a un cliente legítimo y un falso
negativo deja pasar un fraude. Mostrar los cuatro evita presentar el prototipo
como si fuera infalible. Estos ejemplos sirven para explicar, no para estimar el
rendimiento global; para eso se utilizan todas las filas del test.

El valor mensual relaciona las decisiones con sus consecuencias. Al duplicar el
volumen también crecen el beneficio y los costos por transacción; el alojamiento
mensual no se duplica. Por eso el nuevo valor es el doble del valor base más el
costo fijo original. Al dejar de generar alertas desaparece el beneficio, pero
permanece el costo de operación.

Las probabilidades de referencia ayudan a detectar si se combinaron pesos,
escalador o calibración de experimentos distintos. Si un experimento nuevo cambia
deliberadamente estos artefactos, hay que revisar y actualizar los casos con
evidencia de ese experimento. No se deben cambiar las referencias solo para
ocultar un fallo del diagnóstico.

## Alcance y ensayo manual

La comprobación usa los archivos locales. No comprueba acceso a la publicación
privada, botones, estilos, compatibilidad del navegador o entradas propuestas en
vivo. Antes de exponer, abre la aplicación, prueba los dos ejemplos principales,
compara A y B y restaura la política del experimento. Para abrir la copia local:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory site/dist
```

Después visita `http://127.0.0.1:8765`. El diagnóstico tampoco demuestra ausencia
de sesgo, calibración perfecta o independencia del test. Se conserva la limitación
ya documentada: 648 entradas legítimas de test coinciden con train al excluir Time.

## Procedencia de los ejemplos

Los ejemplos pertenecen al test original de ULB / Worldline Credit Card Fraud
Detection, transacciones europeas de septiembre de 2013. Se incluyen sus 29 entradas
anonimizadas y referencias derivadas del experimento. La base identifica términos
ODbL y sus contenidos DbCL; estos ejemplos conservan esa atribución y no se
relicencian como código.

- Fuente: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- ODbL: https://opendatacommons.org/licenses/odbl/1-0/
- DbCL: https://opendatacommons.org/licenses/dbcl/1-0/
