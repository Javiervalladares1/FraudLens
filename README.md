# FraudLens

Proyecto final de **CC3092 Deep Learning y Sistemas Inteligentes**.

**Javier Valladares · 23045**  
**Ian Cumes · 23236**

Repositorio Git: [Javiervalladares1/FraudLens](https://github.com/Javiervalladares1/FraudLens) (privado).

Clasificador de fraude con dos redes neuronales entrenadas, una referencia clásica, evaluación temporal y función explícita de valor. La aplicación ejecuta los pesos aprendidos en el navegador.

## Abrir la demo

Demo privada publicada con Sites: https://fraudlens-javier-ian.javiervalladares.chatgpt.site

Para abrirla localmente, sin instalar PyTorch:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory site/dist
```

Abrir http://127.0.0.1:8765. Se necesita HTTP, no abrir index.html con file://. El sitio descarga archivos estáticos del mismo origen y realiza las predicciones localmente. No integra pagos ni bloqueos. Los ejemplos vienen de test y sus etiquetas no entran al modelo.

## Resultados ejecutados

| Modelo | AP test | Precisión | Recall | F1 | Valor mensual estimado |
| --- | --- | --- | --- | --- | --- |
| Regresión logística | 0.6946 | 66.67% | 72.97% | 0.6968 | US$11,193.20 |
| MLP A (1,505 parámetros) | 0.8222 | 50.00% | 83.78% | 0.6263 | US$12,662.02 |
| MLP B (4,545 parámetros) | 0.7595 | 38.04% | 83.78% | 0.5232 | US$12,421.48 |

A fue seleccionada por **valor de validación**, antes de reportar el test. Matriz A: VP=62, FP=62, FN=12, VN=56,610. Los resultados corresponden a una sola semilla. La diferencia B-A tiene IC bootstrap 95% de -US$866.29 a +US$366.70 y no es concluyente.

El valor es **hipotético**, frente a aprobar todas las transacciones, para N=100,000/mes, pérdida L=US$150/fraude, efectividad q=80%, falsa alarma Cfp=US$3, revisión Cr=US$0.50/alerta, inferencia Ci=US$0.00002 y alojamiento H=US$10/mes:

```text
V = (N/n) * (VP * L * q - FP * Cfp - (VP + FP) * Cr) - N * Ci - H
```

No es ahorro observado, utilidad contable ni validación con una institución financiera. No incluye desarrollo inicial, mantenimiento o riesgo regulatorio. La tasa de fraude mensual se extrapola del test.

## Reproducir entrenamiento y evaluación

Probado con Python 3.12 en macOS arm64, CPU y 4 hilos. Para inferencia web basta un navegador moderno. requirements.txt fija versiones del experimento; en otros sistemas las ruedas disponibles pueden variar.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.download_data
python -m scripts.train
python -m scripts.export_web_data
python -m unittest discover -s tests -v
node tests/parity.mjs
```

El CSV se descarga de la copia usada por el tutorial oficial de TensorFlow. El entrenamiento verifica formato y ausencia de nulos. Se eliminan 1,081 duplicados de entradas. Luego se separa por Time en 60% train, 10% calibración, 10% validación y 20% test, sin separar el mismo segundo. El escalador aprende solo del train. Se utiliza log(1+Amount) y V1-V28. Time se usa únicamente para partir. No se sobremuestrea evaluación.

La semilla es 23045. Ambos MLP usan AdamW, lr=0.001, batch=1024, weight_decay=0.0001 y gradient clipping=5. A entrena 12 épocas, B 20. Checkpoints por AP de validación. B incorpora dropout 0.15 y peso positivo sqrt(n_neg/n_pos)=22.288. Platt se ajusta en calibración sin pesos. Umbrales exactos y selección de la red se fijan con valor en validación. El test se utiliza para informar, no para seleccionar.

El manifiesto de particiones y las predicciones completas se regeneran. Cambiar código, dependencias, hardware o semilla puede cambiar los números. Los intervalos bootstrap no incluyen reentrenamiento, costos ni dependencia temporal.

## Inferencia batch

```bash
python -m scripts.predict mis_transacciones.csv --output predicciones.csv --model A
```

Requiere V1-V28 y Amount numéricos y finitos, Amount >= 0. `Class` es opcional y se ignora. Los componentes PCA deben corresponder al mismo esquema del benchmark. No se puede aplicar directamente a números de tarjeta o montos sin las demás variables.

## Archivos

- `src/model.py`: arquitecturas y preprocesamiento.
- `src/value.py`: contabilidad y selección exacta de umbral.
- `scripts/train.py`: EDA, entrenamiento, calibración, evaluación, bootstrap y sensibilidad.
- `models/`: checkpoints, calibradores, escalador y referencia logística.
- `results/`: métricas, curvas, sensibilidad, explicabilidad y predicciones.
- `site/dist/`: interfaz, inferencia JS, pesos JSON y puntajes de test.
- `tests/`: pruebas de valor, validación de entradas y paridad Python-JS.
- `docs/informe.md`: informe editable.
- `docs/guia_presentacion.md`: guion de pitch, demo y preguntas.
- `output/`: informe PDF, diapositivas y video de demostración.

## Regenerar documentos

Para el informe: `python -m pip install reportlab pillow`, luego `python -m scripts.build_report`. El generador usa Arial de macOS; en otro sistema ajustar la ruta a una fuente instalada compatible. Las diapositivas usan `@oai/artifact-tool` del runtime de Codex y su script está en `scripts/build_slides.mjs`. Los entregables finales se incluyen para que no sea necesario regenerarlos.

## Fuente, privacidad y licencia de datos

Datos: ULB / Worldline, Credit Card Fraud Detection, transacciones europeas de septiembre de 2013. Se reconocen 284,807 filas originales y 492 fraudes. Fuente: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. Copia: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv. Tutorial: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data.

Kaggle identifica la base como Open Database License (ODbL) y sus contenidos como Database Contents License (DbCL). Consultar https://opendatacommons.org/licenses/odbl/1-0/ y https://opendatacommons.org/licenses/dbcl/1-0/. No se incluye el CSV completo en Git. Los ejemplos y derivados del benchmark mantienen esta atribución y los términos aplicables a los datos. No se relicencian como código.

El benchmark carece de identidad de tarjetas y grupos demográficos. Se desconoce el ajuste original del PCA. La evaluación no garantiza independencia entre titulares ni rendimiento en Guatemala en 2026. Las explicaciones por oclusión son locales y no causales. El prototipo no debe utilizarse para decisiones automáticas sobre personas o pagos reales.

## Laboratorio 10: Explainable AI

La carpeta `explicabilidad/` audita el **modelo A ya entrenado** mediante
Integrated Gradients (Captum). No entrena, no ajusta escaladores/calibradores ni
cambia el umbral. Incluye cinco explicaciones locales, una muestra global
aleatoria de 1,024 transacciones, un análisis separado de los 74 fraudes de test,
sensibilidad a referencias y mediciones de costo. La semilla es **23045**.

Desde la raíz del proyecto, con Python 3.12 y los checkpoints de `models/`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-xai.txt
python -m scripts.download_data
python -m explicabilidad.analizar
python -m explicabilidad.informe
python -m explicabilidad.verificar
python -m unittest discover -s tests -v
```

Si ya existe un entorno del proyecto, activarlo e instalar `requirements-xai.txt`
sin recrearlo. El análisis reconstruye el manifiesto temporal original cuando
falta; **no ejecutar `scripts.train` para este laboratorio**. El CSV original no
se incluye en Git, y se verifica su SHA-256 documentado en `data/README.md`.

El PDF queda en
`output/pdf/Laboratorio_10_XAI_FraudLens_Valladares_Cumes.pdf` (cuatro páginas).
`explicabilidad/README.md` explica archivos, parámetros e interpretación. Todas
las figuras y tablas se regeneran con los comandos anteriores. El enlace del
repositorio ya está guardado en `explicabilidad/config.json`. Para regenerar el
PDF con ese enlace y verificar la entrega:

```bash
python -m explicabilidad.informe --repo https://github.com/Javiervalladares1/FraudLens
python -m explicabilidad.verificar --entrega
```

El verificador `--entrega` rechaza un PDF que aún
declare pendiente ese enlace. El enlace no publica ni sube archivos por sí mismo.

**Hallazgo de auditoría:** 648 entradas de test coinciden exactamente con train
al excluir Time; todas son legítimas. La deduplicación original incluía Time.
Esto limita la independencia de la evaluación y requiere revisar particiones en
un futuro ciclo del modelo. El laboratorio mantiene el experimento original.
Las atribuciones explican log-odds calibrados relativos a una referencia, no
causas ni el significado desconocido de los componentes PCA.

## Pendientes de entrega

1. Revisar el acceso privado de la demo desde el equipo de presentación, o usar la copia local.
2. Personalizar la narración del video y ensayar el dominio de datos, modelo, costos y límites.
3. Presentar durante la semana del 16 de noviembre de 2026 (turno por confirmar). Entregar el escrito el miércoles 18 de noviembre de 2026.
