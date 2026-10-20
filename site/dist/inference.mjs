// Inferencia del MLP exportado de PyTorch. Sin dependencias ni llamadas a servicios.
export function validateInput(row) {
  if (!row || Array.isArray(row) || typeof row !== 'object') throw new Error('Se requiere un objeto JSON.');
  for (const name of [...Array.from({length:28},(_,i)=>`V${i+1}`),'Amount']) {
    if (typeof row[name] !== 'number' || !Number.isFinite(row[name])) throw new Error(`${name} debe ser un número finito.`);
  }
  if (row.Amount < 0) throw new Error('Amount debe ser mayor o igual que cero.');
  return row;
}
export function transform(row, artifact) {
  validateInput(row);
  const raw=[...Array.from({length:28},(_,i)=>row[`V${i+1}`]),Math.log1p(row.Amount)];
  return raw.map((v,i)=>(v-artifact.scaler.mean[i])/artifact.scaler.scale[i]);
}
export function fromStandardized(input, network) {
  let values=input;
  network.layers.forEach((layer,index)=>{
    values=layer.weight.map((weights,j)=>{
      let total=layer.bias[j];
      for(let k=0;k<weights.length;k++) total+=weights[k]*values[k];
      return index<network.layers.length-1 ? Math.max(total,0) : total;
    });
  });
  const z=values[0]*network.calibration.coef+network.calibration.intercept;
  return z>=0 ? 1/(1+Math.exp(-z)) : Math.exp(z)/(1+Math.exp(z));
}
export function predict(row,artifact,version=artifact.selected) {
  if (!artifact.networks[version]) throw new Error('Modelo desconocido.');
  return fromStandardized(transform(row,artifact),artifact.networks[version]);
}
export function explain(row,artifact,version=artifact.selected) {
  const z=transform(row,artifact), net=artifact.networks[version], base=fromStandardized(z,net);
  return z.map((_,i)=>{
    const changed=[...z];changed[i]=0;
    return {feature:artifact.feature_names[i],delta:base-fromStandardized(changed,net)};
  }).sort((a,b)=>Math.abs(b.delta)-Math.abs(a.delta));
}
export function summarize(rows,version,threshold,assumptions) {
  let tp=0,fp=0,fn=0,tn=0;
  for(const row of rows) {
    const flag=row[version]>=threshold;
    if(row.y===1){if(flag)tp++;else fn++;}else{if(flag)fp++;else tn++;}
  }
  const scale=assumptions.monthly_transactions/rows.length;
  const avoided=tp*assumptions.loss_per_fraud*assumptions.prevention_rate*scale;
  const friction=fp*assumptions.false_alarm_cost*scale;
  const reviews=(tp+fp)*assumptions.review_cost*scale;
  const operation=assumptions.monthly_transactions*assumptions.inference_cost+assumptions.monthly_hosting;
  return {tp,fp,fn,tn,recall:tp/Math.max(tp+fn,1),precision:tp/Math.max(tp+fp,1),
    f1:2*tp/Math.max(2*tp+fp+fn,1),value:avoided-friction-reviews-operation,
    avoided,friction,reviews,operation,alerts:(tp+fp)*scale};
}
