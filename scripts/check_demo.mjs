// Diagnóstico de los artefactos de la demo, sin entrenamiento ni dependencias externas.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {predict, summarize, validateInput} from '../site/dist/inference.mjs';

const readJson = path => JSON.parse(fs.readFileSync(new URL(path, import.meta.url), 'utf8'));
const money = value => new Intl.NumberFormat('en-US', {
  style: 'currency', currency: 'USD', maximumFractionDigits: 2,
}).format(value);

function close(actual, expected, tolerance, name) {
  assert.ok(Number.isFinite(actual) && Number.isFinite(expected), `${name}: valor no finito.`);
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `${name}: se esperaba ${expected}, se obtuvo ${actual}.`);
}

function outcome(label, alert) {
  if (label === 1) return alert ? 'Verdadero positivo' : 'Falso negativo';
  return alert ? 'Falso positivo' : 'Verdadero negativo';
}

function main() {
  const artifact = readJson('../site/dist/model.json');
  const metrics = readJson('../site/dist/metrics.json');
  const rows = readJson('../site/dist/test_scores.json');
  const fixtures = readJson('../tests/demo_cases.json');
  assert.equal(fixtures.schema_version, 1, 'Formato de casos desconocido.');
  assert.equal(artifact.selected, 'A', 'Estos casos corresponden a la demo original con A.');
  assert.equal(metrics.selected, artifact.selected, 'No coincide la versión seleccionada.');
  assert.deepEqual(artifact.assumptions, metrics.assumptions, 'No coinciden los supuestos económicos.');
  assert.ok(rows.length > 0, 'Faltan los puntajes del test.');

  // Evita que JSON corrupto produzca conteos aparentemente válidos.
  for (const [index, row] of rows.entries()) {
    assert.ok(row.y === 0 || row.y === 1, `Etiqueta inválida en test, fila ${index}.`);
    for (const version of ['A', 'B', 'LR']) {
      assert.ok(Number.isFinite(row[version]) && row[version] >= 0 && row[version] <= 1,
        `Puntaje inválido de ${version} en test, fila ${index}.`);
    }
  }

  const summaries = {};
  for (const version of ['A', 'B', 'LR']) {
    const model = metrics.models[version];
    const expected = model.metrics.test;
    assert.equal(rows.length, expected.n, `Cantidad de filas de ${version}.`);
    if (version !== 'LR') {
      close(artifact.networks[version].threshold, model.threshold, 1e-12, `Umbral de ${version}`);
    }
    const actual = summarize(rows, version, model.threshold, artifact.assumptions);
    for (const field of ['tp', 'fp', 'fn', 'tn']) {
      assert.equal(actual[field], expected[field], `${version}: conteo ${field}.`);
    }
    for (const field of ['precision', 'recall', 'f1']) {
      close(actual[field], expected[field], 1e-12, `${version}: ${field}`);
    }
    close(actual.value, expected.business.net_value, 0.001, `${version}: valor mensual`);
    summaries[version] = actual;
  }

  // Class nunca se entrega a predict: la etiqueta solo permite interpretar el resultado.
  const inputNames = [...Array.from({length: 28}, (_, i) => `V${i + 1}`), 'Amount'].sort();
  const caseLines = [];
  const seenIds = new Set();
  const seenOutcomes = new Set();
  assert.equal(fixtures.cases.length, 4, 'Se requieren los cuatro casos de la demostración.');
  for (const sample of fixtures.cases) {
    assert.ok(!seenIds.has(sample.id), `Caso repetido: ${sample.id}.`);
    seenIds.add(sample.id);
    assert.ok(sample.label === 0 || sample.label === 1, `Etiqueta de ${sample.id}.`);
    assert.deepEqual(Object.keys(sample.features).sort(), inputNames, `Entradas de ${sample.id}.`);
    validateInput(sample.features);
    const results = [];
    for (const version of ['A', 'B']) {
      const probability = predict(sample.features, artifact, version);
      close(probability, sample.expected_probability[version], 2e-5, `${sample.id}, ${version}`);
      const alert = probability >= artifact.networks[version].threshold;
      const kind = outcome(sample.label, alert);
      if (version === 'A') {
        assert.equal(kind, sample.expected_outcome_A, `Tipo de resultado de ${sample.id}.`);
        seenOutcomes.add(kind);
      }
      results.push(`${version}: ${(100 * probability).toFixed(4)}% | ${kind}`);
    }
    caseLines.push(`${sample.id} | referencia ${sample.label === 1 ? 'fraude' : 'legítima'}\n  ${results.join('\n  ')}`);
  }
  assert.equal(seenOutcomes.size, 4, 'Falta mostrar algún tipo de acierto o error en A.');

  const valid = fixtures.cases[0].features;
  const invalidCases = [
    [null, /objeto JSON/],
    [[], /objeto JSON/],
    [{}, /V1/],
    [{...valid, V5: Infinity}, /V5/],
    [{...valid, V5: NaN}, /V5/],
    [{...valid, V5: '0'}, /V5/],
    [{...valid, Amount: -1}, /Amount/],
  ];
  for (const [input, error] of invalidCases) assert.throws(() => validateInput(input), error);

  // Comprueba las tres situaciones económicas que se explican en la presentación.
  const threshold = metrics.models.A.threshold;
  const assumptions = artifact.assumptions;
  const doubled = summarize(rows, 'A', threshold, {
    ...assumptions, monthly_transactions: assumptions.monthly_transactions * 2,
  });
  close(doubled.value, 2 * summaries.A.value + assumptions.monthly_hosting, 0.001,
    'Doble volumen con alojamiento fijo');
  const noAlerts = summarize(rows, 'A', 1, assumptions);
  assert.equal(noAlerts.tp + noAlerts.fp, 0, 'El umbral 1 todavía genera alertas en este test.');
  close(noAlerts.value, -summaries.A.operation, 0.001, 'Costo sin alertas');
  const conventional = summarize(rows, 'A', 0.5, assumptions);
  const expectedHalf = metrics.models.A.test_at_05;
  for (const field of ['tp', 'fp', 'fn', 'tn']) assert.equal(conventional[field], expectedHalf[field]);
  close(conventional.value, expectedHalf.business.net_value, 0.001, 'Valor al umbral 0.5');

  console.log('FRAUDLENS | VERIFICACIÓN DE LA DEMO\n');
  console.log(`OK: ${rows.length} filas de test y coherencia de A, B y LR.`);
  for (const version of ['A', 'B', 'LR']) {
    const s = summaries[version];
    console.log(`${version}: VP ${s.tp}, FP ${s.fp}, FN ${s.fn}, VN ${s.tn}; valor mensual estimado ${money(s.value)}.`);
  }
  console.log('\nCUATRO CASOS PARA EXPLICAR ACIERTOS Y ERRORES\n' + caseLines.join('\n'));
  console.log(`\nOK: ${invalidCases.length} entradas inválidas rechazadas.`);
  console.log(`Escenario A base: ${money(summaries.A.value)}; doble volumen: ${money(doubled.value)}.`);
  console.log(`A con umbral 0.5: ${money(conventional.value)}; sin alertas: ${money(noAlerts.value)}.`);
  console.log('\nRESULTADO: todas las comprobaciones pasaron.');
  console.log('Los valores son estimaciones bajo supuestos. Este diagnóstico no mide ahorro real, acceso a la publicación ni funcionamiento visual del navegador.');
}

try {
  main();
} catch (error) {
  console.error(`VERIFICACIÓN FALLIDA: ${error.message}`);
  process.exitCode = 1;
}
