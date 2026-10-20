import fs from 'node:fs';
import assert from 'node:assert/strict';
import {predict,validateInput,summarize} from '../site/dist/inference.mjs';
const artifact=JSON.parse(fs.readFileSync(new URL('../site/dist/model.json',import.meta.url)));
const cases=JSON.parse(fs.readFileSync(new URL('parity_cases.json',import.meta.url)));
let maxError=0;
for(const c of cases)for(const version of ['A','B']){
  const actual=predict(c.features,artifact,version),error=Math.abs(actual-c.expected[version]);
  maxError=Math.max(error,maxError);assert.ok(error<2e-5,`Paridad ${version}: ${error}`);
}
assert.throws(()=>validateInput({}),/V1/);
assert.throws(()=>validateInput({...cases[0].features,Amount:-1}),/Amount/);
assert.throws(()=>validateInput({...cases[0].features,V5:Infinity}),/V5/);
const metrics=JSON.parse(fs.readFileSync(new URL('../results/metrics.json',import.meta.url)));
const rows=JSON.parse(fs.readFileSync(new URL('../site/dist/test_scores.json',import.meta.url)));
for(const version of ['A','B','LR']){
  const m=metrics.models[version],actual=summarize(rows,version,m.threshold,artifact.assumptions);
  assert.equal(actual.tp,m.metrics.test.tp);assert.equal(actual.fp,m.metrics.test.fp);
  assert.ok(Math.abs(actual.value-m.metrics.test.business.net_value)<.001);
}
console.log(JSON.stringify({cases:cases.length,models:2,max_probability_error:maxError,value_parity:'passed',invalid_inputs:'passed'}));
