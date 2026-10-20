"""Video de demo a partir de capturas verificadas y narración sintética local."""
from pathlib import Path
import wave,json,subprocess,textwrap
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1]
TMP=ROOT/'tmp/video';OUT=ROOT/'output/video'
TMP.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
FONT='/System/Library/Fonts/Supplemental/Arial.ttf';BOLD='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
scenes=[
('FraudLens','Deep learning y valor económico','tmp/slides/slide-01.png',
 'Presentamos FraudLens, el proyecto de detección de fraude de Javier Valladares, carné veintitrés mil cuarenta y cinco, e Ian Cumes, carné veintitrés mil doscientos treinta y seis. La pregunta es cuánto valor genera mejorar las decisiones de un modelo. Esta demostración usa capturas reales de la aplicación y narración sintética.'),
('Resultados del experimento','62 de 74 fraudes detectados','results/demo_inicio.png',
 'La aplicación muestra resultados medidos sobre un conjunto de prueba temporal. La red A detectó sesenta y dos de setenta y cuatro fraudes y generó sesenta y dos falsas alarmas. La separación por tiempo conserva transacciones futuras para prueba. Los costos y el volumen del simulador son supuestos, no ahorros observados.'),
('Una transacción legítima','Entrada de prueba, fuera del train','results/demo_legitima.png',
 'Comenzamos con una transacción legítima del conjunto de prueba. El usuario carga veintinueve variables anonimizadas. El navegador transforma el monto, estandariza las entradas y ejecuta los pesos aprendidos. El puntaje queda debajo del umbral y el sistema no genera alerta. La etiqueta se revela después de predecir.'),
('Un fraude de prueba','Puntaje A: 97.6 por ciento','results/demo_fraude.png',
 'Ahora analizamos la transacción doscientos veintinueve mil setecientos doce. No pertenece al entrenamiento. La red A calcula aproximadamente noventa y siete punto seis por ciento y recomienda revisión. La referencia confirma que era fraude. La oclusión indica qué componentes cambian más el puntaje, sin explicar causas ni características humanas.'),
('Comparación de versiones','La red más grande rindió peor','tmp/slides/slide-07.png',
 'Se comparan una regresión logística y dos redes neuronales entrenadas con PyTorch. La red B añade capacidad, ponderación y dropout. En esta corrida detecta los mismos fraudes que A, pero genera treinta y nueve falsas alarmas adicionales. La red A fue seleccionada por valor en validación, antes de informar el resultado de prueba.'),
('Modelo de valor','Supuestos editables, en dólares','results/demo_valor.png',
 'El simulador estima pérdida evitada menos falsas alarmas, revisión y operación. El escenario supone cien mil transacciones al mes, ciento cincuenta dólares por fraude y ochenta por ciento de prevención efectiva. La red A aporta un valor neto estimado de doce mil seiscientos sesenta y dos dólares al mes. Cambiar los costos o el umbral cambia esa estimación.'),
('Conclusión y límites','Más valor requiere evidencia','tmp/slides/slide-09.png',
 'Al bajar el umbral de A, el F uno disminuye, pero el valor estimado sube alrededor de mil novecientos treinta y cuatro dólares mensuales. La diferencia económica entre A y B todavía tiene incertidumbre. Los datos europeos de dos mil trece requieren validación reciente y local. El prototipo produce recomendaciones y no bloquea tarjetas ni ejecuta pagos.')
]
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();clips=[];captions=[];elapsed=0
for number,(title,takeaway,file,narration) in enumerate(scenes,1):
    audio=TMP/f'voice-{number:02d}.aiff';script=TMP/f'voice-{number:02d}.txt';script.write_text(narration)
    subprocess.run(['/usr/bin/say','-v','Paulina','-r','142','-f',str(script),'-o',str(audio)],check=True)
    wav=TMP/f'voice-{number:02d}.wav'
    subprocess.run([ffmpeg,'-y','-hide_banner','-loglevel','error','-i',str(audio),'-ar','22050','-ac','1',str(wav)],check=True)
    with wave.open(str(wav),'r') as a:duration=a.getnframes()/a.getframerate()
    duration+=1.5
    frame=Image.new('RGB',(1280,720),'#172f3a');draw=ImageDraw.Draw(frame)
    draw.text((32,25),f'{number:02d} / {title}',font=ImageFont.truetype(BOLD,30),fill='white')
    src=Image.open(ROOT/file).convert('RGB');src.thumbnail((900,510),Image.Resampling.LANCZOS)
    x=25+(900-src.width)//2;y=95+(510-src.height)//2;frame.paste(src,(x,y))
    draw.rectangle((952,102,958,153),fill='#c8ee9d')
    cursor=181
    for line in textwrap.wrap(takeaway,width=19):
        draw.text((955,cursor),line,font=ImageFont.truetype(BOLD,27),fill='#c8ee9d');cursor+=36
    note='Capturas de la demo y resultados medidos. Costos del negocio hipotéticos.'
    cursor+=25
    for line in textwrap.wrap(note,width=26):
        draw.text((955,cursor),line,font=ImageFont.truetype(FONT,18),fill='#b9cece');cursor+=25
    draw.line((32,640,1248,640),fill='#47616a',width=1)
    draw.text((32,665),'Javier Valladares 23045  /  Ian Cumes 23236',font=ImageFont.truetype(FONT,17),fill='#b9cece')
    draw.text((875,665),'CC3092 · Narración sintética',font=ImageFont.truetype(FONT,17),fill='#b9cece')
    png=TMP/f'scene-{number:02d}.png';frame.save(png);clip=TMP/f'clip-{number:02d}.mp4'
    subprocess.run([ffmpeg,'-y','-hide_banner','-loglevel','error','-loop','1','-framerate','24','-i',str(png),'-i',str(wav),'-t',str(duration),'-vf','format=yuv420p','-c:v','libx264','-preset','veryfast','-tune','stillimage','-crf','22','-c:a','aac','-b:a','128k','-af','apad','-movflags','+faststart',str(clip)],check=True)
    clips.append(clip);captions.append(dict(chapter=number,title=title,start=elapsed,duration=duration,narration=narration));elapsed+=duration
    print(number,title,round(duration,2),flush=True)
concat=TMP/'concat.txt';concat.write_text('\n'.join(f"file '{p.name}'" for p in clips))
target=OUT/'FraudLens_Demostracion.mp4'
subprocess.run([ffmpeg,'-y','-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(target)],check=True)
(OUT/'capitulos.json').write_text(json.dumps(captions,ensure_ascii=False,indent=2))
assert 120<=elapsed<=240,f'Duración fuera de 2 a 4 minutos: {elapsed}'
print(json.dumps(dict(file=str(target),duration_seconds=elapsed,narration='synthetic_Paulina',visuals='verified_app_screenshots_and_results')),flush=True)
