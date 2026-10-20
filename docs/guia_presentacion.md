# Guía de presentación y defensa

**Javier Valladares 23045 · Ian Cumes 23236**

## Pitch de 8 a 10 minutos

Reparto sugerido, que ambos deben practicar completo: Javier, diapositivas 1 a 6. Ian, diapositivas 7 a 12. Los tiempos en notas suman aproximadamente 9 minutos y medio. Son sugerencias, no evidencia de una división de trabajo realizada.

1. **FraudLens (30 s).** La pregunta es cuánto valor cambia al detectar mejor fraude cuando las falsas alarmas también cuestan. Presentar integrantes.
2. **Problema (45 s).** Aprobar todo logra 99.87% de exactitud y cero recall. El usuario es un analista de riesgo de un procesador hipotético.
3. **Datos (45 s).** ULB/Worldline, 2013, 284,807 originales. Duplicados eliminados antes de dividir. Cuatro bloques por tiempo, escalador ajustado en train.
4. **Modelos (50 s).** A es un MLP pequeño. B aumenta capacidad, pondera positivos e incorpora dropout. Referencia LR con mismo protocolo. No hay preentrenamiento ni API de LLM.
5. **Valor (55 s).** Explicar VP*L*q, FP*Cfp, revisión de toda alerta y operación. Todos los costos son supuestos, no tarifas bancarias comprobadas.
6. **Umbral (55 s).** En A, de t=0.5 a t=0.03461: recall 68.9% a 83.8%, F1 baja, valor sube US$1,934/mes. El umbral se eligió en validación.
7. **Resultados (55 s).** A y B detectan 62 de 74 fraudes. B añade 39 falsas alarmas. A se seleccionó en validación, no por ver los resultados de test.
8. **Arquitectura (45 s).** Describir JSON, validación, transformación, MLP, Platt, umbral y revisión humana. Pesos reales ejecutados localmente en navegador.
9. **Incertidumbre (50 s).** IC B-A incluye cero. No afirmar diferencia concluyente. Bootstrap no incluye deriva, costos ni reentrenamiento.
10. **Explicabilidad (45 s).** Oclusión local al promedio de train. Variables PCA no son atributos humanos. No hay auditoría demográfica posible.
11. **Sensibilidad (45 s).** Factores ±20% y ±50% de una variable a la vez. Prevención limitada al 100%. Costos por validar con piloto.
12. **Cierre (40 s).** Ajustar el umbral agrega más valor que crecer la red en esta corrida. Proponer datos recientes, varias semillas, reglas/árboles y modo sombra.

## Demo en vivo de 3 a 5 minutos

1. Abrir la demo publicada o servir `site/dist` localmente antes de comenzar. La publicación es privada: verificar acceso en el equipo de la exposición.
2. Mostrar el resumen y aclarar que el valor es hipotético.
3. Analizar `TX-227820`, ejemplo legítimo del test. Revelar etiqueta después de ver la predicción.
4. Analizar `TX-229712`, fraude de test que no pertenece al entrenamiento. A da aproximadamente 97.6% y recomienda revisión. Revelar etiqueta.
5. Cambiar a B y volver a predecir. Mostrar que el modelo y la explicación cambian.
6. Alterar una entrada JSON o usar una propuesta del catedrático con V1-V28 y Amount. La etiqueta será desconocida. No usar números de tarjeta ni afirmar que se admiten transacciones bancarias crudas.
7. Probar entrada inválida, por ejemplo `{}`. Mostrar el mensaje de validación.
8. En modelo de valor, cambiar el volumen a 200,000. El valor se aproxima a US$25,334/mes con A. Restaurar.
9. Mover el umbral hacia 1. Ningún caso genera alerta y la operación cuesta US$12/mes. Restaurar.
10. Explicar que la exploración de test en la demo no modifica los umbrales del informe ni vuelve a entrenar modelos.

## Preguntas que ambos deben poder responder

- **¿Por qué AP y no solo accuracy?** La clase positiva es muy escasa. Aprobar todo parece preciso y omite todo fraude. AP resume el ranking precisión-recall.
- **¿Por qué F1 y valor discrepan?** F1 pondera errores de una manera distinta a la función monetaria. Detectar 11 fraudes adicionales puede pagar 62 falsas alarmas bajo los costos asumidos.
- **¿Cómo evitaron fuga?** Corte temporal, deduplicación anterior, escalador de train, calibración separada y selección en validación. No se puede garantizar independencia por titular porque no existen IDs de tarjeta.
- **¿Qué significa el 97.6%?** Puntaje ajustado por Platt sobre 24 positivos de otro bloque. La calibración puede degradarse si cambia la prevalencia. No es una garantía individual ni prueba de culpabilidad.
- **¿Qué optimiza BCE ponderada?** Cambia el peso de errores positivos durante el entrenamiento. No reproduce exactamente la función de valor. El umbral termina de alinear la política con el costo.
- **¿Por qué B fue peor?** Puede ser capacidad, regularización, ponderación o variación muestral. Cambiaron varios factores y no hay ablación que permita atribuir una causa única.
- **¿La diferencia económica es concluyente?** No. IC pareado B-A de -US$866 a +US$367 incluye cero. Hace falta evaluar más periodos y semillas.
- **¿Es viable en Guatemala?** Es viable como prototipo. El dataset europeo de 2013 no demuestra rendimiento local ni disponibilidad del PCA de un banco.
- **¿Se ahorraron US$12,662?** No. Es valor neto extrapolado del test con parámetros hipotéticos. Falta medir intervención real, costos y volumen.
- **¿Se implementó monitoreo?** El informe propone el plan. La demo no incluye una plataforma de monitoreo productiva.
- **¿Por qué no usar regresión logística?** Debe considerarse. LR logra mayor F1 en el umbral elegido y menor AP y valor puntual que A. Su menor complejidad puede ser preferible si el resultado cambia en nuevos periodos.
- **¿Qué hace la explicación?** Sustituye una variable estandarizada por cero, el promedio de train, y mide cambio del puntaje. No es causal y puede producir entradas poco probables.
- **¿Costo de cómputo?** Microbenchmark de red en CPU, con calentamiento. Los costos en dólares son imputados. No incluyen toda la aplicación ni una factura real.

## Pendientes antes de entregar

- Elegir cuenta y publicar un repositorio público de GitHub. Agregar su URL al informe.
- Leer y personalizar el informe. Comprobar que ambos pueden ejecutar y explicar la solución.
- Regrabar el video si prefieren una explicación con sus voces. La versión entregada usa capturas verificadas, texto y narración sintética en español.
- Revisar acceso a la demo y conservar copia local para exposición sin Internet.
- Confirmar turno de la semana del 16 de noviembre. Entrega escrita: miércoles 18 de noviembre de 2026.
