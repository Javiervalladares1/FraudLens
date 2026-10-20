import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const root=process.cwd();
const runtime='/Users/javiervalladares/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=path.join(runtime,'node/node_modules');
const skill='/Users/javiervalladares/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const m=JSON.parse(await fs.readFile('results/metrics.json','utf8'));
const presentation=Presentation.create({slideSize:{width:1280,height:720}});
presentation.theme.defaultFont='Arial';
const C={ink:'#172f3a',teal:'#318981',orange:'#ba5b35',lime:'#c8ee9d',muted:'#687c82',paper:'#f6f7f3',line:'#d7e0d5'};
const pct=x=>(100*x).toFixed(1)+'%';const usd=x=>'US$'+Math.round(x).toLocaleString('en-US');
function text(s,value,x,y,w,h,size=26,color=C.ink,bold=false){
  const sh=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  sh.text=value;sh.text.style={typeface:'Arial',fontSize:size,bold,color,verticalAlignment:'top',wrap:'word',insets:0};return sh;
}
function rect(s,x,y,w,h,fill){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});}
function slide(title,notes=''){
  const s=presentation.slides.add();s.background.fill=C.paper;
  text(s,title,64,50,1140,100,42,C.ink,true);
  const num=presentation.slides.items.length;
  rect(s,64,662,1152,1,C.line);text(s,'FraudLens / CC3092',64,677,800,20,14,C.muted);text(s,String(num).padStart(2,'0'),1164,677,52,20,14,C.muted);
  s.speakerNotes.text=notes;return s;
}
async function img(s,file,x,y,w,h){const b=await fs.readFile(file);s.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:'image/png',alt:path.basename(file),position:{left:x,top:y,width:w,height:h},fit:'contain'});}
function table(s,values,x,y,w,h,widths){
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
  t.borders.assign({style:'solid',fill:C.line,width:1});
  for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
    const cell=t.getCell(r,c);cell.fill=r===0?C.ink:r%2?'#ffffff':'#ecf1e7';
    cell.text.style={typeface:'Arial',fontSize:22,color:r===0?'#ffffff':C.ink,bold:r===0,insets:12};
  }
  return t;
}
const source='Datos ULB/Worldline: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. Copia oficial: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data. Resultados propios: results/metrics.json. Costos del escenario: src/value.py.';
{
  const s=presentation.slides.add();s.background.fill=C.ink;
  text(s,'FraudLens',72,150,1000,110,92,'#ffffff',true);
  text(s,'Detección de fraude y valor económico\nde las decisiones de un modelo',78,288,1050,112,42,'#dce8e3');
  rect(s,78,442,90,7,C.lime);
  text(s,'Javier Valladares 23045\nIan Cumes 23236',78,505,800,76,27,'#ffffff');
  text(s,'CC3092 · Deep Learning y Sistemas Inteligentes',78,644,1100,26,19,'#9cb5b7');
  s.speakerNotes.text='Apertura, 30 s. Pregunta: ¿cuánto vale detectar más fraude si molestamos a clientes legítimos? Todos los resultados técnicos se ejecutaron. Los parámetros monetarios son hipotéticos. '+source;
}
{
  const s=slide('El problema del analista de riesgo','45 s. Usuario hipotético: analista de un procesador. Referencia analítica: aprobar todo. No describir esta referencia como operación real de un banco. '+source);
  text(s,'99.87%',64,186,550,100,80,C.teal);
  text(s,'Exactitud si se aprueba todo',68,302,520,40,27,C.ink,true);
  text(s,'Ese sistema no detecta ningún fraude.\nLa exactitud oculta la clase minoritaria.',68,366,490,120,27,C.muted);
  rect(s,633,185,2,370,C.line);
  text(s,'US$19,561',700,197,490,90,58,C.ink);
  text(s,'Pérdida mensual de referencia',700,312,490,70,27,C.ink,true);
  text(s,'Escenario hipotético\n100,000 transacciones / mes\nUS$150 de pérdida por fraude',700,405,485,134,25,C.muted);
}
{
  const s=slide('Datos y separación temporal','45 s. Original: 284,807 filas, 492 fraudes. Eliminar 1,081 duplicados de entradas deja 283,726 filas y 473 fraudes. No separar segundos iguales. Escalador solo train. PCA original y titulares desconocidos limitan la independencia. '+source);
  text(s,'283,726 transacciones limpias · 473 fraudes',64,162,1150,50,32,C.teal);
  table(s,[['Bloque','Filas','Fraudes','Uso'],['Train','170,235','342','Ajustar pesos y escalador'],['Calibración','28,372','24','Ajustar Platt'],['Validación','28,373','33','Checkpoint, umbral y red'],['Test','56,746','74','Reportar resultados']],64,242,1152,295,[190,185,175,602]);
  text(s,'V1–V28 y log(1 + Amount). Time define los cortes y queda fuera del modelo.',64,568,1150,65,25,C.muted);
}
{
  const s=slide('Dos redes y una referencia clásica','50 s. MLP entrenados desde cero con PyTorch en CPU. ReLU y logit binario, AdamW 0.001, batch 1024, weight decay 0.0001, clip 5. Checkpoint por AP de validación. Platt en bloque distinto. LR con mismos datos. '+source);
  table(s,[['Decisión','MLP A','MLP B'],['Capas','29-32-16-1','29-64-32-16-1'],['Parámetros','1,505','4,545'],['Pérdida','BCE','BCE, peso positivo 22.288'],['Dropout','0','0.15'],['Checkpoint','Época 8 de 12','Época 12 de 20']],64,175,1152,345,[300,390,462]);
  text(s,'Referencia: regresión logística. Semilla 23045. La comparación cambia varios factores a la vez.',64,557,1152,70,25,C.muted);
}
{
  const s=slide('La función de valor','55 s. Explicar cada término. VP evita L por q. Las falsas alarmas generan fricción y todas las alertas generan revisión. FN y fraude no prevenido quedan en pérdida residual. Valores definidos para el escenario, sin fuentes bancarias ni ahorro observado. '+source);
  text(s,'V = (N/n) [VP · L · q − FP · Cfp',64,176,1150,70,43,C.ink,true);
  text(s,'− (VP + FP) · Cr] − N · Ci − H',64,247,1150,70,43,C.ink,true);
  table(s,[['Parámetro','Supuesto','Parámetro','Supuesto'],['N / mes','100,000','L / fraude','US$150'],['q efectiva','80%','Cfp','US$3'],['Cr / alerta','US$0.50','N·Ci + H','US$12 / mes']],64,364,1152,211,[230,320,250,352]);
  text(s,'Valor incremental frente a aprobar todo. Todos los costos son hipotéticos.',64,602,1150,38,23,C.orange);
}
{
  const s=slide('El umbral cambia el valor','55 s. La decisión principal. Al bajar el umbral, 11 VP nuevos y 62 FP en test. Se pierde F1 y se gana valor. Umbral elegido en validación, no en test. Teórico calibrado: (Cfp+Cr)/(Lq+Cfp)=0.02846. '+source);
  table(s,[['Política A','F1','Recall','VP / FP','Valor mensual'],['Umbral 0.5','0.816','68.9%','51 / 0','US$10,728'],['Umbral 0.03461','0.626','83.8%','62 / 62','US$12,662']],64,188,1152,210,[290,140,180,180,362]);
  text(s,'+US$1,934 / mes',64,450,1150,80,62,C.teal,true);
  text(s,'Cambio estimado bajo los supuestos. La mejora económica no exige un F1 mayor.',64,558,1120,70,27,C.muted);
}
{
  const s=slide('Resultados en el test temporal','55 s. A elegida por validación: 11770 vs B11450. En test A y B tienen 62 VP, 12 FN. B tiene 101 FP, A62. LR obtiene F1 mayor que A pero valor menor. AP es average precision, no área trapezoidal. '+source);
  table(s,[['Modelo','AP','Precisión','Recall','F1','Valor mensual'],['LR','0.695','66.7%','73.0%','0.697','US$11,193'],['MLP A','0.822','50.0%','83.8%','0.626','US$12,662'],['MLP B','0.759','38.0%','83.8%','0.523','US$12,421']],64,183,1152,255,[160,120,190,170,140,372]);
  text(s,'La versión B genera 39 falsas alarmas adicionales.',64,485,1130,65,35,C.orange);
  text(s,'La red más grande no mejoró el resultado puntual de esta corrida.',64,566,1130,65,27,C.muted);
}
{
  const s=slide('Demostración y arquitectura','45 s en pitch. La demo adicional toma 3 a 5 min. Entrada de test, preproceso, pesos, Platt, umbral y resultado. Mostrar etiqueta después de predecir. Las entradas no se envían a un proveedor de IA. '+source);
  await img(s,'results/figures/arquitectura.png',64,146,1152,245);
  text(s,'Inferencia local con pesos aprendidos',64,425,1140,54,35,C.teal,true);
  text(s,'Entrada JSON, versiones A/B, revisión humana y simulador de costos.\nParidad JS–PyTorch en 16 ejemplos: error máximo 2.03e-7.',64,500,1120,100,27,C.muted);
}
{
  const s=slide('Incertidumbre de la comparación','50 s. Bootstrap pareado de filas del test: 400 valor y 100 AP. Costos, política y modelo fijos. No incluye semillas, dependencia temporal o deriva. No afirmar diferencia concluyente A-B. Solo24 positivos en calibración y33validación. '+source);
  text(s,'Valor de A: US$12,662 / mes',64,181,1130,65,43,C.teal,true);
  text(s,'IC 95%: US$9,311 a US$16,019',64,269,1130,60,34,C.ink);
  rect(s,64,373,1152,2,C.line);
  text(s,'Diferencia B − A: −US$241 / mes',64,414,1130,65,39,C.ink,true);
  text(s,'IC 95%: −US$866 a +US$367',64,498,1130,60,34,C.orange);
  text(s,'El intervalo incluye cero. Hace falta evaluar más periodos y semillas.',64,583,1130,50,25,C.muted);
}
{
  const s=slide('Explicabilidad y límites de uso','45 s. Oclusión al promedio del entrenamiento. V14 no tiene interpretación comercial conocida. No causal, puede salir del manifold. Datos no permiten auditoría demográfica. Ningún bloqueo real, revisión humana. No afirmar usabilidad probada. '+source);
  await img(s,'results/figures/explicacion.png',64,166,765,355);
  text(s,'PCA anonimizado',876,184,335,60,28,C.ink,true);
  text(s,'La sensibilidad local\nno explica causas.\n\nSin identidad ni\ngrupos demográficos.',876,270,335,220,25,C.muted);
  text(s,'Datos europeos de 2013. El prototipo requiere validación reciente y local.',64,571,1150,60,27,C.orange);
}
{
  const s=slide('Sensibilidad de los supuestos','45 s. Cambios de +/-20% y +/-50%, una variable a la vez. q acotada al100%. Política fija, sin reoptimizar test. Los mayores efectos se relacionan con volumen, pérdida y prevención. Fuente: results/sensitivity.json. '+source);
  await img(s,'results/figures/sensibilidad.png',64,150,1152,380);
  text(s,'Los costos y la prevención efectiva necesitan medición en un piloto.',64,573,1152,60,28,C.muted);
}
{
  const s=slide('Conclusiones y siguiente validación','40 s. Cierre: elegir política económica, mantener red pequeña, evaluar deriva y pérdidas por monto. Repositorio público pendiente de la cuenta de los integrantes. Ensayar respuestas sobre FN/FP, selección de umbral y diferencia F1valor. '+source);
  text(s,'Ajustar el umbral aporta más valor\nque aumentar el tamaño de la red.',64,182,1140,134,45,C.ink,true);
  text(s,'MLP A: 62 de 74 fraudes detectados.\nUS$12,662 / mes bajo el escenario hipotético.',64,355,1140,110,32,C.teal);
  text(s,'Siguiente paso: datos recientes, costos medidos y piloto en modo sombra.',64,515,1120,76,28,C.muted);
  text(s,'Demo privada: fraudlens-javier-ian.javiervalladares.chatgpt.site',64,615,1150,28,19,C.muted);
}
const staging=path.join(root,'tmp/slides');await fs.mkdir(staging,{recursive:true});
for(let i=0;i<presentation.slides.items.length;i++){
  const s=presentation.slides.items[i];
  const png=await presentation.export({slide:s,format:'png',scale:1});
  await fs.writeFile(path.join(staging,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
const candidate=path.join(staging,'candidate.pptx');await(await PresentationFile.exportPptx(presentation)).save(candidate);
const final=path.join(root,'output/presentacion/FraudLens_Presentacion.pptx');
const result=await finalizePresentation({
  workspaceDir:root,candidatePath:candidate,finalPath:final,
  requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
  pythonExecutable:path.join(runtime,'python/bin/python3'),
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
  fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
  receiptPath:path.join(staging,'validation.json')
});
console.log(JSON.stringify(result));
