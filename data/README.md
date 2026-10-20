# Datos

Fuente: Machine Learning Group - ULB / Worldline, Credit Card Fraud Detection.
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Copia del tutorial oficial de TensorFlow:
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

Ejecutar `python -m scripts.download_data`.

SHA-256 esperado: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.

La fuente describe licencia ODbL para la base y DbCL para los contenidos. No se redistribuye aquí el CSV completo. Los ejemplos de prueba y puntajes derivados conservan atribución a la fuente. Los cortes, deduplicación y transformaciones de este proyecto son modificaciones documentadas, no datos originales adicionales.

`processed/split_manifest.csv` conserva la fila fuente y partición. `processed/test.csv` se genera para auditoría. No se usan en inferencia las etiquetas incluidas en ejemplos de demo.
