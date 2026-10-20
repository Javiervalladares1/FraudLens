"""Genera gráficos, informe editable en Markdown y PDF desde resultados medidos."""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from src.value import Assumptions,business,outcomes

ROOT=Path(__file__).resolve().parents[1]
INK='#172f3a';TEAL='#318981';ORANGE='#ba5b35';MUTED='#687c82';LIME='#c8ee9d'
def read(path):return json.loads((ROOT/path).read_text())
def money(x):return f'US${x:,.2f}'
def percent(x):return f'{100*x:.2f}%'

def charts():
    m=read('results/metrics.json');hist=read('results/history.json');p=pd.read_csv(ROOT/'results/test_predictions.csv');df=pd.read_csv(ROOT/'data/raw/creditcard.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':'#ccd6d0','savefig.facecolor':'white'})
    out=ROOT/'results/figures';out.mkdir(exist_ok=True)
    def end(name):plt.tight_layout();plt.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight');plt.close()
    fig,ax=plt.subplots(1,2,figsize=(9,3.1))
    for name,color in [('A',TEAL),('B',ORANGE)]:
        h=hist[name];ax[0].plot([r['epoch'] for r in h],[r['train_loss'] for r in h],label=name,color=color)
        ax[1].plot([r['epoch'] for r in h],[r['validation_ap'] for r in h],label=name,color=color)
        ax[1].scatter(m['models'][name]['best_epoch'],max(r['validation_ap'] for r in h),color=color,s=24)
    ax[0].set(xlabel='Época',ylabel='Pérdida de entrenamiento',title='BCE con pesos distintos por versión');ax[1].set(xlabel='Época',ylabel='Average precision',title='Selección de checkpoint en validación',ylim=(.45,.95))
    ax[0].legend();ax[1].legend();end('entrenamiento')
    fig,ax=plt.subplots(figsize=(8.2,3.3))
    for name,color in [('A',TEAL),('B',ORANGE),('LR','#6c7893')]:
        prec,rec,_=precision_recall_curve(p.y,p[name]);ax.plot(rec,prec,label=f'{name}: AP = {m["models"][name]["metrics"]["test"]["ap"]:.3f}',color=color,lw=2)
    ax.set(xlabel='Recall de fraude',ylabel='Precisión',xlim=(0,1.01),ylim=(0,1.03));ax.legend(loc='lower left');ax.grid(alpha=.18);end('precision_recall')
    fig,ax=plt.subplots(figsize=(8.2,3.4))
    for name,color in [('A',TEAL),('B',ORANGE)]:
        ts=np.r_[np.geomspace(1e-5,.99,180),m['models'][name]['threshold']];ts.sort()
        val=[business(outcomes(p.y,p[name],t))['net_value'] for t in ts]
        ax.plot(ts,val,label=name,color=color,lw=2)
        t=m['models'][name]['threshold'];ax.scatter(t,m['models'][name]['metrics']['test']['business']['net_value'],color=color,s=40)
    ax.axhline(0,color='#87988e',lw=.7);ax.set(xscale='log',xlabel='Umbral (escala logarítmica)',ylabel='Valor neto mensual estimado (USD)',ylim=(-25000,16000));ax.legend();ax.grid(alpha=.15);end('valor_umbral')
    fig,ax=plt.subplots(1,2,figsize=(9,3.1))
    for label,color in [(0,TEAL),(1,ORANGE)]:
        ax[0].hist(np.log1p(df.loc[df.Class==label,'Amount']),bins=45,density=True,alpha=.6,color=color,label='Legítima' if label==0 else 'Fraude')
    ax[0].set(xlabel='log(1 + Amount)',ylabel='Densidad',title='Distribución del monto por clase');ax[0].legend()
    e=read('results/eda.json');labels=list(e['split']);pre=[100*e['split'][n]['prevalence'] for n in labels]
    ax[1].bar(labels,pre,color=[INK,'#789993',TEAL,ORANGE]);ax[1].set(ylabel='Fraude (%)',title='Cambio de prevalencia entre bloques');ax[1].tick_params(axis='x',labelsize=8)
    end('eda')
    sens=read('results/sensitivity.json');fig,ax=plt.subplots(figsize=(8.2,3.2))
    labels={'loss_per_fraud':'Pérdida por fraude','false_alarm_cost':'Costo de falsa alarma','monthly_transactions':'Volumen mensual','prevention_rate':'Prevención efectiva'}
    for param,label in labels.items():
        s=[r for r in sens if r['parameter']==param];ax.plot([r['factor'] for r in s],[r['A'] for r in s],marker='o',label=label)
    ax.set(xlabel='Multiplicador del supuesto (una variable a la vez)',ylabel='Valor neto mensual de A (USD)');ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.18);end('sensibilidad')
    ex=read('results/explanation.json')['features'][:7];fig,ax=plt.subplots(figsize=(8.2,3.1))
    ax.barh([r['feature'] for r in ex][::-1],[100*r['delta'] for r in ex][::-1],color=TEAL);ax.set(xlabel='Cambio del puntaje al restituir la variable (puntos porcentuales)',title='Oclusión local al promedio del entrenamiento');end('explicacion')
    fig,ax=plt.subplots(figsize=(9,2.7));ax.axis('off')
    stages=[('Entrada','29 valores\nPCA + Amount'),('Validación','Tipos finitos\nAmount >= 0'),('Preproceso','log(1 + Amount)\nEscalador train'),('MLP','Pesos entrenados\nA o B'),('Decisión','Platt + umbral\nRevisión humana')]
    for i,(title,text) in enumerate(stages):
        x=.015+i*.2
        ax.add_patch(plt.Rectangle((x,.23),.17,.55,facecolor='#f0f4eb',edgecolor='#cbd7ce',lw=1))
        ax.text(x+.085,.64,title,ha='center',fontsize=10,fontweight='bold');ax.text(x+.085,.43,text,ha='center',va='center',fontsize=8,linespacing=1.7)
        if i<4:ax.annotate('',xy=(x+.197,.5),xytext=(x+.173,.5),arrowprops=dict(arrowstyle='->',color=TEAL))
    ax.text(.5,.05,'Python entrena y evalúa. El sitio estático ejecuta la inferencia localmente.',ha='center',fontsize=9,color=MUTED);end('arquitectura')

def build():
    charts();m=read('results/metrics.json');e=read('results/eda.json');mA=m['models']['A'];mB=m['models']['B'];mL=m['models']['LR'];A=mA['metrics']['test'];B=mB['metrics']['test'];L=mL['metrics']['test'];a=m['assumptions']
    font='/System/Library/Fonts/Supplemental/'
    for name,file in [('Arial','Arial.ttf'),('Arial-Bold','Arial Bold.ttf'),('Arial-Italic','Arial Italic.ttf')]:pdfmetrics.registerFont(TTFont(name,font+file))
    pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic')
    styles={
        'body':ParagraphStyle('Body',fontName='Arial',fontSize=10,leading=14.5,textColor=INK,spaceAfter=10),
        'small':ParagraphStyle('Small',fontName='Arial',fontSize=8.2,leading=11.5,textColor=MUTED,spaceAfter=9),
        'h1':ParagraphStyle('H1',fontName='Arial-Bold',fontSize=21,leading=25,textColor=INK,spaceAfter=18),
        'h2':ParagraphStyle('H2',fontName='Arial-Bold',fontSize=12,leading=16,textColor=TEAL,spaceBefore=8,spaceAfter=9),
        'cell':ParagraphStyle('Cell',fontName='Arial',fontSize=8.3,leading=11,textColor=INK),
        'cover':ParagraphStyle('Cover',fontName='Arial',fontSize=48,leading=52,textColor=INK,spaceAfter=20),
    }
    story=[];md=['# FraudLens\n\nCC3092 Deep Learning y Sistemas Inteligentes\n\nJavier Valladares 23045\nIan Cumes 23236\n\nFecha: 19 de octubre de 2026\n']
    def para(text,style='body'):
        story.append(Paragraph(text,styles[style]));md.append(text.replace('<b>','**').replace('</b>','**').replace('<br/>','\n')+'\n')
    def title(text):story.append(Paragraph(text,styles['h1']));md.append('\n## '+text+'\n')
    def sub(text):story.append(Paragraph(text,styles['h2']));md.append('\n### '+text+'\n')
    def page():story.append(PageBreak())
    def table(headers,rows,widths=None):
        contents=[[Paragraph(str(c),styles['cell']) for c in row] for row in [headers]+rows]
        t=Table(contents,colWidths=widths or [516/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8efdf')),('FONTNAME',(0,0),(-1,0),'Arial-Bold'),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#dce5dd'))]))
        story.extend([t,Spacer(1,13)]);md.append('| '+' | '.join(map(str,headers))+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n')
    def chart(name,height=205):
        from PIL import Image as PI
        path=ROOT/f'results/figures/{name}.png';im=PI.open(path);w,h=im.size
        scale=min(516/w,height/h);width=w*scale;actual_height=h*scale
        story.extend([Image(str(path),width=width,height=actual_height),Spacer(1,10)]);md.append(f'![{name}](../results/figures/{name}.png)\n')

    story.append(Spacer(1,70));para('CC3092 / DEEP LEARNING Y SISTEMAS INTELIGENTES','small')
    story.append(Paragraph('FraudLens',styles['cover']));para('Detección de fraude y valor económico<br/>de las decisiones de un modelo','h1')
    para('Javier Valladares <b>23045</b><br/>Ian Cumes <b>23236</b>')
    story.append(Spacer(1,28))
    table(['Experimento','Resultado medido en prueba'],[['Modelo seleccionado en validación','MLP A, 1,505 parámetros'],['Fraudes detectados','62 de 74 (83.78%)'],['Falsas alarmas','62 de 56,672 transacciones legítimas'],['Valor neto mensual estimado',money(A['business']['net_value'])+' para 100,000 transacciones']],widths=[248,268])
    para('Pregunta central: ¿cuánto valor cambia al mejorar un modelo o ajustar su umbral de decisión?')
    para('Los resultados técnicos se calcularon con un experimento ejecutado. Los costos y el volumen representan un escenario hipotético. El proyecto no reporta un piloto bancario ni ahorro observado.','small')
    para('Versión de trabajo: 19 de octubre de 2026. Entrega indicada: 18 de noviembre de 2026.','small')
    page();title('4.1 Contexto de negocio y problema')
    para('El usuario objetivo es un analista de riesgo de un procesador de pagos. El proceso propuesto recibe una transacción anonimizada, calcula un puntaje de fraude y decide si genera una alerta para revisión humana. El escenario es realista, sin una entidad financiera participante ni acceso a pagos reales.')
    sub('Línea base y alcance')
    para('La línea base económica aprueba todas las transacciones. No incurre en falsas alarmas ni revisión, pero asume la pérdida de los fraudes. Se usa como referencia analítica, no como descripción de la operación de un banco. Con la prevalencia del test, 100,000 transacciones mensuales y una pérdida supuesta de US$150 por fraude, su pérdida esperada es '+money(A['business']['baseline_loss'])+' al mes.')
    para('El sistema produce alertas. La intervención posterior se modela con una efectividad de prevención de 80%. Un verdadero positivo no equivale automáticamente a recuperar dinero. Las falsas alarmas consumen tiempo y pueden molestar a clientes legítimos.')
    table(['KPI','Referencia','Política A estimada'],[['Recall de fraude','0% al aprobar todo',percent(A['recall'])],['Fraudes sin alerta / mes','130.41',f"{A['business']['monthly_missed_frauds']:.2f}"],['Alertas por revisar / mes','0',f"{A['business']['monthly_alerts']:.2f}"],['Valor incremental / mes','US$0',money(A['business']['net_value'])]])
    sub('Justificación y alternativa simple')
    para('Un MLP puede aprender relaciones no lineales entre componentes PCA. La hipótesis se contrasta con regresión logística, entrenada sobre el mismo bloque y evaluada con la misma función de valor. En test, A mejora la AP de 0.695 a 0.822 y el valor puntual en '+money(A['business']['net_value']-L['business']['net_value'])+'. Esto apoya experimentar con una red pequeña, aunque no demuestra superioridad universal.')
    para('La exactitud de predecir siempre “legítima” alcanza 99.87% en test. Por eso la evaluación prioriza average precision (AP), recall, precisión y consecuencias económicas. El alcance excluye autorización de pagos y decisiones sobre titulares identificables.')

    page();title('4.2 Datos y división temporal')
    para('Se utiliza Credit Card Fraud Detection, atribuido a ULB y Worldline, con transacciones europeas de dos días de septiembre de 2013 [1]. Se descargó la copia CSV enlazada por el tutorial oficial de TensorFlow [2]. El archivo contiene Time, V1 a V28, Amount y Class. Las V son componentes PCA anonimizados. Class es la etiqueta y nunca se incorpora a las entradas.')
    table(['Control de calidad','Resultado'],[['Filas originales / fraudes','284,807 / 492'],['Nulos en el archivo','0'],['Duplicados de entradas eliminados','1,081 (incluyen 19 filas fraudulentas)'],['Filas limpias / fraudes','283,726 / 473'],['Desbalance limpio','0.1667% de fraude'],['Formato y licencia','CSV; base ODbL y contenidos DbCL según Kaggle [1,3]']],widths=[258,258])
    table(['Bloque','Filas','Fraudes','Prevalencia','Rango Time'],[[n,f"{s['rows']:,}",s['frauds'],percent(s['prevalence']),f"{s['time_min']:.0f} - {s['time_max']:.0f}"] for n,s in e['split'].items()],widths=[103,75,65,88,185])
    para('La división respeta el tiempo: 60% entrenamiento, 10% calibración, 10% validación y 20% test, con cortes sin separar un mismo segundo. Se elimina duplicación antes de dividir y se preserva source_row para auditar el origen. El StandardScaler se ajusta exclusivamente con entrenamiento. No se balancean ni se sobremuestrean los conjuntos de evaluación.')
    para('La anonimización impide separar por tarjeta o titular. Tampoco se conoce cómo se ajustó el PCA original. La partición temporal reduce fuga del experimento, pero no garantiza independencia entre personas ni elimina posibles limitaciones del preprocesamiento realizado por la fuente.','small')

    page();title('4.2 Análisis exploratorio y preparación')
    chart('eda',240)
    para('Figura 1. Los montos muestran asimetría y colas largas. La prevalencia cambia entre los bloques temporales: entrenamiento contiene 0.2009% de fraude y test 0.1304%. Esta diferencia justifica conservar el orden temporal y medir degradación por bloque.','small')
    table(['Monto, unidad original','Medición en datos limpios'],[['Mediana','22.00'],['Percentil 95','365.34'],['Percentil 99','1,018.97'],['Máximo','25,691.16']],widths=[258,258])
    para('El preprocesamiento aplica log(1 + Amount) y estandariza las 29 entradas con media y desviación del train. No elimina valores extremos: pueden representar fraude o transacciones legítimas importantes. Time se usa para la separación y se excluye del modelo, evitando que una tendencia de tiempo absoluta domine la inferencia.')
    para('Los costos económicos se expresan en USD como parámetros del escenario. Amount conserva la unidad original y no se utiliza como conversión monetaria ni como pérdida económica observada. La pérdida constante simplifica la comparación, pero omite que cada transacción tiene una exposición distinta.')
    para('El conjunto cubre dos días, una población europea y patrones de 2013. Su representatividad para Guatemala en 2026 es limitada. Se necesita un nuevo periodo de evaluación con operaciones locales antes de hacer afirmaciones sobre una institución concreta.')

    page();title('4.3 Función de valor y supuestos')
    para('Para un conjunto de n transacciones, un umbral t define VP, FP, FN y VN. La función estima valor incremental mensual frente a aprobar todo:')
    para('<b>V(t) = (N/n) [VP(t) · L · q - FP(t) · Cfp<br/>- (VP(t) + FP(t)) · Cr] - N · Ci - H</b>')
    para('N es volumen mensual, L pérdida por fraude, q prevención efectiva tras una alerta, Cfp costo adicional de falsa alarma, Cr revisión de cualquier alerta, Ci costo unitario de inferencia y H operación fija. La pérdida residual es (N/n)[FN + (1-q)VP]L. Esta identidad permite contrastar el ahorro con la diferencia de costos totales.')
    table(['Parámetro','Valor','Fundamento'],[['N','100,000 / mes','Volumen hipotético de un procesador pequeño'],['L','US$150','Escenario redondeado, sin estimación de una entidad'],['q','80%','Efectividad hipotética de la intervención'],['Cfp','US$3','Fricción y atención adicional supuestas'],['Cr','US$0.50 / alerta','Revisión hipotética de VP y FP'],['Ci','US$0.00002 / transacción','Costo asignado, sin factura observada'],['H','US$10 / mes','Presupuesto supuesto, sin cotización']],widths=[55,133,328])
    para('Todos los parámetros de la tabla son estimaciones de diseño. La fuente de cada uno es el escenario definido para este proyecto, no estadísticas bancarias. Para justificarlos operativamente se debe cronometrar revisión, estimar el margen perdido por fricción, medir recuperación real y cotizar alojamiento. La demo permite modificar N, L, Cfp y q.')
    para('El modelo no incluye desarrollo, soporte, regulación, capital, pérdida por abandono futuro ni colas de revisión. Por tanto, V representa contribución operativa incremental bajo supuestos y no utilidad contable ni retorno completo de inversión.','small')

    page();title('4.3 Curva de valor y punto de operación')
    chart('valor_umbral',230)
    para('Figura 2. Exploración retrospectiva del test. Los puntos marcan umbrales elegidos previamente en validación. No se usa esta curva para volver a optimizarlos. Los umbrales se buscaron entre todos los puntajes distintos de validación, incluyendo la opción de no alertar.','small')
    table(['Política de A','F1 test','Recall','VP / FP','Valor / mes'],[['Umbral 0.5','0.816','68.92%','51 / 0',money(mA['test_at_05']['business']['net_value'])],['Umbral 0.03461','0.626','83.78%','62 / 62',money(A['business']['net_value'])]],widths=[135,62,69,77,173])
    para('Al reducir el umbral, A encuentra 11 fraudes adicionales y genera 62 falsas alarmas en test. Bajo los supuestos, el incremento es '+money(A['business']['net_value']-mA['test_at_05']['business']['net_value'])+' al mes, aunque F1 cae. El resultado demuestra que maximizar F1 y maximizar valor responden a objetivos diferentes.')
    para('Con probabilidades perfectamente calibradas y costos constantes, conviene alertar cuando p &gt; (Cfp + Cr)/(Lq + Cfp), equivalente a 0.02846 en el escenario base. La selección empírica fija tA = 0.034611 y tB = 0.036738. La diferencia puede reflejar error de calibración y el bajo número de fraudes en validación.')
    para('A y B tienen el mismo recall en el punto elegido, pero B genera 39 falsas alarmas adicionales. Su diferencia puntual de valor mensual es -US$240.55. La incertidumbre de esa diferencia se presenta en la sección de resultados.')

    page();title('4.4 Metodología de desarrollo')
    table(['Decisión','MLP A','MLP B'],[['Arquitectura','29-32-16-1','29-64-32-16-1'],['Activación / salida','ReLU / logit binario','ReLU / logit binario'],['Pérdida','BCEWithLogitsLoss','BCE ponderada, peso positivo 22.288'],['Dropout / parámetros','0 / 1,505','0.15 / 4,545'],['Épocas / checkpoint elegido','12 / época 8','20 / época 12'],['Optimizador / tasa','AdamW / 0.001','AdamW / 0.001']],widths=[156,180,180])
    para('Ambas versiones usan batch 1,024, weight decay 0.0001, clip de gradiente 5 y semilla 23045. La salida inicial incorpora el prior del train. El checkpoint se escoge por AP de validación. Después, una regresión logística unidimensional ajusta el logit (Platt) usando únicamente el bloque de calibración. El umbral y la selección entre redes se fijan con valor en validación.')
    para('La búsqueda es acotada a dos configuraciones predefinidas, sin afirmar una optimización extensa. B cambia capacidad, dropout y peso de clase a la vez, por lo que no permite atribuir su resultado a un único factor. La pérdida ponderada facilita aprender la clase minoritaria, pero no incorpora exactamente los costos del negocio. La alineación económica ocurre principalmente en la selección del umbral.')
    chart('entrenamiento',185)
    para('Figura 3. Las pérdidas de A y B usan pesos distintos y no son comparables en magnitud. En B, la AP oscila y no mejora tras la época 12 aunque la pérdida baja. Se conserva el checkpoint con mejor AP. No se usan modelos preentrenados, APIs de LLM ni GPU.','small')

    page();title('4.5 Resultados de las iteraciones')
    table(['Modelo','AP test','Precisión','Recall','F1','Valor / mes'],[[n,f"{v['ap']:.3f}",percent(v['precision']),percent(v['recall']),f"{v['f1']:.3f}",money(v['business']['net_value'])] for n,v in [('LR',L),('A',A),('B',B)]],widths=[49,60,81,73,55,198])
    table(['Modelo / bloque','AP','F1','Recall','Valor / mes'],[[n+' / '+split,f"{obj['metrics'][split]['ap']:.3f}",f"{obj['metrics'][split]['f1']:.3f}",percent(obj['metrics'][split]['recall']),money(obj['metrics'][split]['business']['net_value'])] for n,obj in [('A',mA),('B',mB)] for split in ['train','validation','test']],widths=[135,65,65,85,166])
    chart('precision_recall',192)
    para('Figura 4. PR-AUC se reporta como average precision de scikit-learn, no como integración trapezoidal. El punto operativo de A tiene matriz VP=62, FP=62, FN=12, VN=56,610. La red A supera a la referencia clásica en AP y valor puntual, mientras LR obtiene mayor F1 al ser más conservadora.','small')
    para('La selección de A ocurrió con validación: US$11,770.33 frente a US$11,449.60 de B. El valor en entrenamiento no se usa como pronóstico mensual: sus prevalencias difieren entre bloques.','small')

    page();title('4.5 Incertidumbre y costo computacional')
    table(['Indicador','Estimación','Intervalo bootstrap 95%'],[['AP A','0.822','0.738 - 0.881'],['AP B','0.759','0.671 - 0.847'],['Valor mensual A',money(A['business']['net_value']),'US$9,310.51 - US$16,018.67'],['Valor mensual B',money(B['business']['net_value']),'US$9,027.10 - US$16,150.84'],['Diferencia B - A','-US$240.55','-US$866.29 a +US$366.70']],widths=[160,124,232])
    para('Los intervalos usan remuestreo pareado de filas del test: 400 réplicas para valor y 100 para AP, semilla 23045. Mantienen fijos modelo, calibración, umbral y costos. No incluyen incertidumbre por reentrenamiento, supuestos o dependencia temporal. El intervalo B-A incluye cero: esta corrida no establece una diferencia económica concluyente entre ambas redes.')
    table(['Costo medido / estimado','A','B'],[['Entrenamiento de todas las épocas',f"{mA['training_seconds']:.3f} s",f"{mB['training_seconds']:.3f} s"],['Inferencia PyTorch p50 / p95',f"{mA['latency_ms_p50']:.4f} / {mA['latency_ms_p95']:.4f} ms",f"{mB['latency_ms_p50']:.4f} / {mB['latency_ms_p95']:.4f} ms"],['Throughput batch, solo red',f"{mA['throughput_per_second']:,.0f} / s",f"{mB['throughput_per_second']:,.0f} / s"],['Costo train a US$0.10 / CPU-h',f"US${mA['training_cost_usd_estimate']:.6f}",f"US${mB['training_cost_usd_estimate']:.6f}"]],widths=[250,133,133])
    para('El benchmark usa CPU arm64, PyTorch con 4 hilos, 50 calentamientos y 500 inferencias unitarias sin gradientes. Throughput se calcula sobre un único batch de 56,746 entradas y no representa capacidad de una API ni incluye red, parseo, preproceso, descargas o concurrencia. Los tiempos cortos dependen fuertemente del equipo. El costo de entrenamiento es una imputación hipotética y excluye preparación, calibración y evaluación.')
    para('La inferencia JS se contrastó en 16 ejemplos de test para ambas redes: diferencia máxima absoluta 2.03e-7 frente a PyTorch. Cuatro pruebas verifican contabilidad, umbrales y sensibilidad. También se verifican rechazo de entradas inválidas y paridad de valor entre Python y JS.','small')

    page();title('4.6 Despliegue y experiencia de usuario')
    chart('arquitectura',166)
    para('Figura 5. Flujo de inferencia de la aplicación desplegada. La etiqueta de referencia queda fuera del cálculo de la predicción.','small')
    para('La aplicación estática contiene el escalador, los pesos y la calibración exportados a JSON. El navegador recibe 29 valores, valida su tipo, transforma Amount y ejecuta las capas del MLP. La etiqueta del ejemplo permanece fuera del cálculo de la predicción y solo se muestra a solicitud del usuario. El sitio genera una recomendación de revisión, sin ejecutar pagos.')
    para('Demo: <link href="https://fraudlens-javier-ian.javiervalladares.chatgpt.site">https://fraudlens-javier-ian.javiervalladares.chatgpt.site</link>. Publicación privada para la cuenta propietaria. Antes de presentar a otra audiencia se debe comprobar el acceso desde el equipo de exposición.')
    sub('Flujo utilizable y casos límite')
    para('La interfaz ofrece ejemplos del test, entrada JSON, versiones A y B, puntaje calibrado, decisión, oclusión local y simulador económico. Los errores de JSON, campos ausentes, valores no numéricos, infinitos y Amount negativo se rechazan. Los valores fuera de rango de train generan una advertencia. La detección de rangos no sustituye una evaluación formal de datos fuera de distribución.')
    para('La prueba manual observó una entrada legítima sin alerta y una fraudulenta con alerta (TX-229712, puntaje A 97.6%). El usuario puede editar una entrada: entonces se deshabilita revelar una etiqueta inexistente. Se comprobó el cambio de modelo y extremos del umbral. Estos ejemplos verifican funcionamiento y no constituyen una prueba de usabilidad con analistas.')
    sub('Monitoreo propuesto')
    para('En una integración futura se registrarían versiones, distribución de puntajes, porcentaje de alertas, tiempos p50/p95 y etiquetas con retraso, sin datos de tarjeta. Revisiones semanales compararían AP, recall y valor con sus intervalos y con periodos anteriores. Un aumento sostenido de falsas alarmas o cambios de distribución activaría revisión del modelo. El prototipo no implementa una plataforma de monitoreo en producción.')

    page();title('4.7 Explicabilidad, ética y riesgos')
    chart('explicacion',200)
    para('Figura 6. Explicación aplicada a TX-229712 mediante oclusión de cada variable al promedio del train. V14 tiene el mayor efecto local en A. La perturbación mide sensibilidad del puntaje, no causalidad. Al ser componentes PCA, no se puede interpretar V14 como una característica humana o comercial específica. Sustituir un componente por la media puede producir combinaciones poco probables.','small')
    table(['Subgrupo por monto','n / fraudes','Recall A','Precisión A'],[[r['group'],f"{r['n']:,} / {r['frauds']}",percent(r['A']['recall']),percent(r['A']['precision'])] for r in read('results/subgroups.json')],widths=[160,140,108,108])
    para('La auditoría por monto explora heterogeneidad, pero las muestras positivas son pequeñas. El archivo no permite evaluar equidad por género, edad, etnia o ubicación. No se asume ausencia de sesgos a partir de la anonimización. Un falso negativo puede afectar al titular y al comercio. Una falsa alarma puede retrasar un pago o imponer verificaciones adicionales.')
    para('El uso previsto requiere revisión humana de alertas. No se debe usar este prototipo para decisiones automáticas sobre cuentas, acusaciones de fraude o exclusión de clientes. Los componentes PCA públicos no permiten desplegar directamente sobre datos de un banco: se necesitaría acceso autorizado al esquema y transformación equivalente, además de una nueva evaluación.')
    para('La demo procesa las entradas localmente y no implementa persistencia de transacciones. Los datos del benchmark no contienen números de tarjeta. El repositorio conserva atribución y descarga reproducible, sin incluir el CSV original completo. Los ejemplos públicos conservan referencia a ULB/Worldline y a los términos de los datos [1,3].','small')

    page();title('4.8 Discusión y conclusiones')
    chart('sensibilidad',215)
    para('Figura 7. Sensibilidad de una variable por vez con la política A fija. Se prueban factores 0.5, 0.8, 1, 1.2 y 1.5. La prevención efectiva se limita a 100%, por lo que 1.5 veces 80% se trunca a 100%. El análisis no reoptimiza umbrales en test y no representa escenarios conjuntos.','small')
    para('<b>Hallazgo principal.</b> Ajustar el umbral por costo aumenta el valor puntual de A en US$1,934.06 al mes frente a t=0.5, pese a reducir F1. En este escenario, los fraudes adicionales detectados compensan las falsas alarmas. Cambiar a una red más grande no mejora el resultado: B detecta el mismo número de fraudes y produce más falsas alarmas.')
    para('<b>Rendimientos decrecientes.</b> A tiene menor costo computacional y mayor valor puntual que B. B necesitaría elevar su contribución en más de US$240.55 al mes, bajo los mismos supuestos, solo para igualar el resultado observado. El intervalo de la diferencia incluye cero y aconseja evaluar más periodos antes de escoger entre ellas por una diferencia pequeña.')
    para('<b>Viabilidad.</b> Una red pequeña permite una demo económica y reproducible. La principal barrera de producción es obtener datos recientes, medir costos y validar la intervención humana. Se propone recopilar otro periodo con etiquetas retrasadas, ejecutar varias semillas y un piloto en modo sombra, comparar contra reglas y árboles potenciados, y valorar pérdidas por monto con límites de exposición.')
    para('La corrida actual usa una sola semilla, 24 fraudes para calibrar, 33 para validación y 74 para prueba. Las conclusiones son del benchmark y del escenario declarado. No se reportan usuarios reales, prueba A/B, transacciones bancarias en vivo ni ahorros observados.','small')

    page();title('4.9 Repositorio y reproducibilidad')
    para('El código del proyecto se entrega en la carpeta de trabajo y en un paquete ZIP. El repositorio local incluye descarga, entrenamiento, evaluación, función de valor, pruebas, pesos y código del sitio. La publicación de un repositorio público de GitHub y su enlace definitivo quedan pendientes de la cuenta y visibilidad que elijan los integrantes.')
    sub('Secuencia de reproducción')
    para('1. Crear un entorno Python 3.12 e instalar requirements.txt.<br/>2. Ejecutar python -m scripts.download_data.<br/>3. Ejecutar python -m scripts.train.<br/>4. Ejecutar python -m scripts.export_web_data.<br/>5. Ejecutar python -m unittest discover -s tests -v y node tests/parity.mjs.<br/>6. Servir site/dist con python -m http.server 8765 --directory site/dist.')
    para('SHA-256 del CSV original: '+e['sha256'],'small')
    table(['Entrega solicitada','Estado y evidencia'],[['Implementación desplegada','Demo privada de FraudLens y código en site/dist'],['Dos versiones de deep learning','Pesos .pt y JSON; resultados A y B calculados'],['Informe escrito','Este documento y fuente editable docs/informe.md'],['Apoyo para presentación','Diapositivas y guía de defensa'],['Video 2 a 4 minutos','Demostración con capturas verificadas, en output/video'],['Repositorio público de GitHub','Pendiente de publicación y enlace definitivo'],['Presentación en vivo','Semana del 16 de noviembre de 2026, turno por confirmar'],['Entrega escrita','Miércoles 18 de noviembre de 2026']],widths=[190,326])
    sub('Referencias')
    para('[1] Machine Learning Group - ULB / Worldline. Credit Card Fraud Detection. Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud<br/>[2] TensorFlow. Classification on imbalanced data. https://www.tensorflow.org/tutorials/structured_data/imbalanced_data<br/>[3] Open Data Commons. ODbL 1.0 y DbCL 1.0. https://opendatacommons.org/licenses/','small')
    para('Los resultados y gráficos de este informe provienen de results/metrics.json, eda.json, history.json, sensitivity.json y test_predictions.csv. Consulta de fuentes y ejecución: 19 de octubre de 2026, hora de Guatemala.','small')

    (ROOT/'docs/informe.md').write_text('\n'.join(md))
    target=ROOT/'output/pdf/FraudLens_Informe_Javier_Valladares_Ian_Cumes.pdf'
    def decorate(canvas,doc):
        canvas.saveState();canvas.setFillColor(colors.HexColor(INK));canvas.rect(0,775,612,17,fill=1,stroke=0)
        canvas.setStrokeColor(colors.HexColor('#dce5dd'));canvas.line(48,43,564,43)
        canvas.setFont('Arial',8);canvas.setFillColor(colors.HexColor(MUTED));canvas.drawString(48,29,'FraudLens / CC3092 / Valladares 23045 - Cumes 23236');canvas.drawRightString(564,29,str(doc.page));canvas.restoreState()
    SimpleDocTemplate(str(target),pagesize=(612,792),leftMargin=48,rightMargin=48,topMargin=45,bottomMargin=58,title='FraudLens - Proyecto final CC3092',author='Javier Valladares 23045; Ian Cumes 23236').build(story,onFirstPage=decorate,onLaterPages=decorate)
    print(target)

if __name__=='__main__':build()
