"""Construye el informe de cuatro páginas a partir de resultados calculados.

python -m explicabilidad.informe [--repo https://github.com/usuario/repositorio]
No vuelve a entrenar ni calcular atribuciones. El Markdown conserva el texto editable.
"""
from __future__ import annotations
import argparse
import json
from html import escape
from pathlib import Path
import re

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WIDTH = 528
INK = "#153247"
TEAL = "#007d83"
MUTED = "#526573"
DEST = ROOT / "output/pdf/Laboratorio_10_XAI_FraudLens_Valladares_Cumes.pdf"


def font_setup():
    # Fuentes locales de uso común; fallback incluido en ReportLab para Linux.
    directory = Path("/System/Library/Fonts/Supplemental")
    if (directory / "Arial.ttf").exists():
        for name, filename in [("LabArial", "Arial.ttf"), ("LabArial-Bold", "Arial Bold.ttf")]:
            pdfmetrics.registerFont(TTFont(name, str(directory / filename)))
        pdfmetrics.registerFontFamily("LabArial", normal="LabArial", bold="LabArial-Bold")
        return "LabArial", "LabArial-Bold"
    return "Helvetica", "Helvetica-Bold"


def build(repository_url=None):
    audit = json.loads((HERE / "resultados/auditoria.json").read_text())
    cfg = json.loads((HERE / "config.json").read_text())
    url = repository_url or cfg.get("repository_url")
    if url and not re.match(r"^https://[^\s<>\"]+$", url):
        raise ValueError("El enlace del repositorio debe ser una URL HTTPS válida")
    metrics = json.loads((ROOT / "results/metrics.json").read_text())
    mlp_metrics = metrics["models"][audit["model"]]["metrics"]["test"]
    lr_metrics = metrics["models"]["LR"]["metrics"]["test"]
    local = pd.read_csv(HERE / "resultados/atribuciones_locales.csv").set_index("case")
    subs = pd.read_csv(HERE / "resultados/sustituciones_diagnosticas.csv")
    global_data = pd.read_csv(HERE / "resultados/importancia_global.csv").set_index("feature")
    cases = {row["case"]: row for row in audit["cases"]}
    normal_font, bold_font = font_setup()
    styles = {
        "body": ParagraphStyle("Body", fontName=normal_font, fontSize=8.8, leading=11.7, textColor=INK, spaceAfter=6),
        "small": ParagraphStyle("Small", fontName=normal_font, fontSize=7.6, leading=10.1, textColor=MUTED, spaceAfter=5),
        "title": ParagraphStyle("Title", fontName=bold_font, fontSize=21, leading=24, textColor=INK, spaceAfter=9),
        "h2": ParagraphStyle("Sub", fontName=bold_font, fontSize=10.6, leading=13, textColor=TEAL, spaceBefore=5, spaceAfter=6),
        "cell": ParagraphStyle("Cell", fontName=normal_font, fontSize=7.4, leading=9.5, textColor=INK),
        "head": ParagraphStyle("Head", fontName=bold_font, fontSize=7.5, leading=9.5, textColor=INK),
        "refs": ParagraphStyle("Refs", fontName=normal_font, fontSize=6.9, leading=9.2, textColor=MUTED, spaceAfter=3),
    }
    story, markdown = [], []

    def p(text, style="body"):
        story.append(Paragraph(text, styles[style]))
        markdown.append(re.sub(r"<[^>]+>", "", text).replace("&gt;", ">").replace("&lt;", "<") + "\n")

    def heading(text):
        p(text, "h2")

    def title(text):
        p(text, "title")

    def table(headers, rows, widths):
        elements = [[Paragraph(escape(str(c)), styles["head" if i == 0 else "cell"])
                     for c in row] for i, row in enumerate([headers] + rows)]
        item = Table(elements, colWidths=widths, hAlign="LEFT", repeatRows=1)
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6f0f1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, -1), .35, colors.HexColor("#d5e1e6")),
        ]))
        story.extend([item, Spacer(1, 6)])
        markdown.append("| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
                        + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows) + "\n")

    def figure(name, max_height):
        path = HERE / f"figuras/{name}.png"
        w, h = PILImage.open(path).size
        scale = min(WIDTH / w, max_height / h)
        item = Image(str(path), width=w * scale, height=h * scale)
        item.hAlign = "CENTER"
        story.extend([item, Spacer(1, 5)])
        markdown.append(f"![{name}](figuras/{name}.png)\n")

    def page():
        story.append(PageBreak())
        markdown.append("\n---\n")

    # PÁGINA 1: investigación y fundamento formal.
    p("CC3092 · DEEP LEARNING Y SISTEMAS INTELIGENTES · LABORATORIO 10", "small")
    title("Explicar FraudLens")
    p("Javier Valladares · 23045 &nbsp;&nbsp; Ian Cumes · 23236 &nbsp;&nbsp; | &nbsp;&nbsp; 16 de noviembre de 2026", "small")
    p("Auditamos el <b>MLP A final, ya entrenado</b> (29-32-16-1, ReLU, 1,505 parámetros), elegido originalmente con validación. Sus entradas son V1-V28 (PCA anónimo) y log(1 + Amount), escaladas con train. Se conservan pesos, calibración Platt, particiones y umbral.")
    heading("1. Investigación de métodos para datos tabulares")
    table(["Método / herramienta", "Idea y requisitos", "Costo orientativo / limitación"], [
        ["IG / captum [1,2]", "Integra gradientes entre referencia e instancia; necesita acceso a gradientes y línea base.", "K evaluaciones con gradientes. Depende de la referencia y de una trayectoria que puede ser artificial."],
        ["LIME / lime [3]", "Ajusta un sustituto local con predicciones de caja negra y un kernel de vecindad.", "S inferencias + ajuste. Vecindario, kernel, discretización y semilla alteran la fidelidad local."],
        ["KernelSHAP / shap [4,5]", "Atribuye valores Shapley por coaliciones; requiere consultas al modelo y conjunto de fondo.", "Aprox. S x B inferencias. Alto costo; depende del fondo y del tratamiento de dependencias."],
        ["TreeSHAP / shap [5]", "Usa la estructura interna de árboles y una política para variables ausentes.", "Algoritmo especializado eficiente. No aplica directamente a nuestro MLP."],
        ["DeepSHAP y GradientSHAP / shap, captum [2,5]", "Reglas tipo DeepLIFT o gradientes esperados sobre un fondo; requieren red/gradientes compatibles.", "Propagaciones según B o muestras. Aproximan SHAP; dependen de operaciones, fondo y supuestos."],
        ["DiCE / dice-ml [6]", "Busca entradas cercanas de otra clase; necesita caja negra o gradientes y restricciones de factibilidad.", "Búsqueda iterativa. Sin semántica de PCA no podemos proponer acciones factibles sobre sus componentes."],
        ["Anchors / alibi [7]", "Busca reglas suficientes de alta precisión; requiere predicciones y un muestreador.", "Consultas adaptativas. Precisión y cobertura valen bajo la distribución de perturbación elegida."],
        ["Oclusión / PyTorch [2]", "Reemplaza una variable por una referencia y mide el cambio de salida; acepta caja negra.", "d + 1 inferencias. No descompone aditivamente interacciones; puede producir entradas irreales."],
    ], [92, 221, 215])
    p("d = variables; K = puntos de integración; S = coaliciones/perturbaciones; B = tamaño del fondo. Los costos de alternativas son teóricos, no benchmarks ejecutados.", "small")
    heading("Propiedades y referencia interpretable")
    p("IG satisface <b>completitud</b>: la integral exacta cumple suma(IG) = F(x) - F(b). Permite contabilizar la diferencia de puntajes. También satisface sensibilidad e invariancia de implementación para funciones equivalentes con igual referencia/trayectoria. El residuo numérico comprueba la suma; no garantiza exactitud individual ni causalidad. [1]")
    p(f"La regresión logística del proyecto es interpretable por estructura; IG es una explicación a posteriori de A. En test: AP logística/A = <b>{lr_metrics['ap']:.4f}/{mlp_metrics['ap']:.4f}</b>, recall = {100*lr_metrics['recall']:.2f}%/{100*mlp_metrics['recall']:.2f}% y F1 = {lr_metrics['f1']:.4f}/{mlp_metrics['f1']:.4f}, con sus umbrales originales. A mejora detección y AP; la logística obtiene mejor F1. Sus coeficientes tampoco recuperan el significado de los componentes PCA.")

    # PÁGINA 2: selección, implementación y explicación global.
    page()
    title("2. Elección e implementación")
    table(["Criterio", "IG: método principal", "KernelSHAP: alternativa descartada"], [
        ["Compatibilidad", "Acceso directo a gradientes del MLP y calibración en PyTorch.", "Compatible como caja negra; no aprovecha los gradientes disponibles."],
        ["Explicación", "Contribuciones firmadas relativas a una referencia; suma verificable en la escala elegida.", "Atribuciones Shapley relativas al fondo/juego; exactitud local depende de su estimación."],
        ["Costo y volumen", "256 puntos; se mide el costo unitario y en lote. Uso propuesto: alertas y auditoría.", "S x B consultas para cada caso; mayor volumen de evaluaciones. No se ejecutó benchmark."],
        ["Comunicación", "Barras que elevan o reducen el puntaje para analistas; mostrar p, umbral y referencia.", "Barras aditivas familiares, pero requieren explicar también el fondo y la ausencia de variables."],
        ["Limitaciones", "Referencia/recorrido artificiales; las variables PCA carecen de semántica operativa.", "Combinaciones de variables irreales y sensibilidad al fondo; no elimina el problema de PCA."],
    ], [96, 216, 216])
    p("Se mantiene IG tras usarlo: permite comprobar la contabilidad del puntaje y su costo es viable para auditoría. LIME también se descartó por añadir un sustituto cuya fidelidad depende del vecindario. Ninguno de estos métodos convierte una asociación en causa.")
    heading("Diseño reproducible")
    p(f"<b>Salida:</b> F(z) = a·MLP(z) + b, log-odds calibrados; p = sigmoid(F). Se incluye Platt y se evita la saturación de la sigmoide en la atribución. <b>Referencia:</b> z = 0, media de train en las entradas transformadas; F(b) = {audit['baseline_score']:.4f}, p(b) = {100*audit['baseline_probability']:.4f}%. Es una referencia sintética, no ausencia de fraude.")
    p(f"<b>Captum {audit['environment']['captum']}:</b> IG con multiplicador (x-b), trayectoria recta y Gauss-Legendre; {cfg['n_steps']} puntos por instancia, semilla {cfg['seed']}, CPU, {cfg['torch_threads']} hilos, lotes de {cfg['example_batch_size']} ejemplos y batch interno {cfg['internal_batch_size']}. Refinamiento hasta {cfg['max_steps']} si |delta| &gt; {cfg['absolute_tolerance']} + {cfg['relative_tolerance']}·|F(x)-F(b)|; todos convergieron con 256. Delta máximo = {audit['delta_abs_max']:.5f}, p95 = {audit['delta_abs_p95']:.5f} log-odds. Se verificaron hashes de todos los modelos sin cambios.")
    p(f"<b>Muestra global:</b> {audit['global_n']:,} filas de test sin reemplazo, elegidas al azar antes de ver atribuciones ({audit['global_frauds']} fraudes). Panel adicional: los {audit['all_frauds_explained']} fraudes de test. No se mezclan sus agregados. IG no usa fondo SHAP; se prueba sensibilidad local con dos referencias alternativas de train.")
    figure("global", 199)
    p(f"Figura 1. Media de |IG|. V4 domina en la muestra aleatoria ({global_data.loc['V4','mean_abs_ig_random']:.3f}); V14 en los fraudes ({global_data.loc['V14','mean_abs_ig_all_frauds']:.3f}). Las escalas de ambos paneles son distintas. Son influencias sobre el modelo, no significados de negocio ni efectos causales.", "small")

    # PÁGINA 3: cinco explicaciones locales y lectura de los resultados.
    page()
    title("3. Cinco explicaciones locales")
    p(f"Test original: {audit['test_n']:,} transacciones y {audit['test_frauds']} fraudes. Se alerta si p ≥ {100*audit['threshold']:.4f}% (F ≥ {audit['threshold_log_odds']:.4f}). Se seleccionan casos por reglas deterministas, usando etiquetas solo para auditoría.")
    figure("locales", 377)
    p("Figura 2. Rojo: contribución que eleva F; azul: la reduce. Se muestran las cuatro mayores |IG| y la suma firmada de las otras 25. Base común F(b) = -7.9853. VP/VN son aciertos; FP/FN son errores. Cada p es la salida real del modelo, no la suma redondeada de las barras.", "small")
    table(["Caso / selección", "Monto", "F(x)", "F(x) - umbral", "Delta IG"], [
        ["C1: VP con mayor p", f"{cases['C1']['amount']:.2f}", f"{cases['C1']['score']:.4f}", f"{cases['C1']['margin_log_odds']:+.4f}", f"{cases['C1']['delta']:+.5f}"],
        ["C2: FP con mayor p", f"{cases['C2']['amount']:.2f}", f"{cases['C2']['score']:.4f}", f"{cases['C2']['margin_log_odds']:+.4f}", f"{cases['C2']['delta']:+.5f}"],
        ["C3: FN con menor p", f"{cases['C3']['amount']:.2f}", f"{cases['C3']['score']:.4f}", f"{cases['C3']['margin_log_odds']:+.4f}", f"{cases['C3']['delta']:+.5f}"],
        ["C4: restante más próximo al umbral", f"{cases['C4']['amount']:.2f}", f"{cases['C4']['score']:.4f}", f"{cases['C4']['margin_log_odds']:+.4f}", f"{cases['C4']['delta']:+.5f}"],
        ["C5: VP restante de monto > 100, mayor p", f"{cases['C5']['amount']:.2f}", f"{cases['C5']['score']:.4f}", f"{cases['C5']['margin_log_odds']:+.4f}", f"{cases['C5']['delta']:+.5f}"],
    ], [230, 54, 64, 102, 78])
    p(f"<b>Lectura:</b> en C1 y C5 domina V14 (+{local.loc['C1','V14']:.2f} y +{local.loc['C5','V14']:.2f}). C4 combina V4/V14 positivos con V16 negativo: queda apenas {abs(cases['C4']['margin_log_odds']):.5f} log-odds bajo el umbral. C3 contiene una señal V4 positiva fuerte, superada por señales negativas. Esto explica por qué una señal aislada no basta para clasificar. El monto de C5 define un subgrupo operativo, no demográfico.")

    # PÁGINA 4: interpretación, discusión, referencias y enlace de entrega.
    page()
    title("4. Interpretación y discusión")
    corrected = float(subs[(subs.case == "C2") & (subs.feature == "V14")].probability_replaced.iloc[0])
    c3max = float(subs[subs.case == "C3"].probability_replaced.max())
    heading("Errores, señales y límites de la evaluación")
    p(f"<b>C2:</b> V14 y V4 suman +{local.loc['C2','V14']+local.loc['C2','V4']:.2f} log-odds y dominan la falsa alarma. Sustituir solo V14 por su media de train reduce p de {100*cases['C2']['probability']:.2f}% a {100*corrected:.3f}% y corrige la etiqueta. <b>C3:</b> V5/V10 aportan {local.loc['C3','V5']:.2f}/{local.loc['C3','V10']:.2f}; ninguna sustitución individual corrige el FN (p máxima {100*c3max:.4f}%). Son diagnósticos de entradas sintéticas, no acciones causales factibles ni recomendaciones para modificar transacciones.")
    p(f"No entran Class, Time ni source_row al modelo; el escalador coincide con train y las decisiones coinciden con las predicciones originales. La auditoría halló <b>{audit['train_test_identical_input_rows']} filas de test con entradas idénticas a train</b> al quitar Time ({audit['train_test_identical_input_frauds']} fraudes). No demuestra uso de la etiqueta, pero limita la independencia de la evaluación: la deduplicación original incluía Time. Falta identidad de tarjetas y se desconoce cómo se ajustó el PCA; no se puede descartar fuga previa ni variables proxy. No se asigna semántica inventada a V14/V4.")
    heading("Estabilidad y decisiones que cambia la explicación")
    figure("sensibilidad", 85)
    p("Figura 3. Al usar una fila legítima real de train (source_row 55846), C2 conserva solo 3/5 variables del top 5; con mediana legítima, 4/5. Se mantiene IG, mostrando la referencia y la sensibilidad. La figura mide coincidencia de variables, no de sus magnitudes ni signos.", "small")
    p("Proponemos revisión humana para alertas y una banda de incertidumbre cerca del umbral, definida y evaluada en validación. Para el siguiente ciclo: auditar duplicados sin Time y particiones por tarjeta, obtener variables con significado y etiquetas verificadas, y vigilar errores por monto. En monto &gt; 100 el recall es 11/15; en monto ≤ 20, 41/48: son grupos pequeños. No eliminamos V14 por causar una falsa alarma ni ajustamos el umbral con estos casos de test.")
    p("El analista vería p, umbral, referencia, contribuciones firmadas y la advertencia de PCA; la etiqueta solo aparece en auditoría. La explicación muestra patrones compartidos entre aciertos/errores y oposición entre señales que AP/F1 no muestran. Confirma heterogeneidad del puntaje, sin validar que las señales tengan sentido semántico en fraude.")
    heading("Costo y uso responsable")
    lat = audit["latency"]
    p(f"En CPU, 30 repeticiones tras 5 calentamientos: inferencia p50 {lat['inference']['p50_ms']:.3f} ms; IG p50/p95 {lat['ig']['p50_ms']:.2f}/{lat['ig']['p95_ms']:.2f} ms; oclusión p50 {lat['occlusion']['p50_ms']:.3f} ms. El lote de {audit['unique_explained']:,} casos únicos tardó {audit['ig_batch_seconds']:.2f} s, solo IG. Se excluyen disco, preproceso y red. Es viable para auditoría o alertas asíncronas; falta medir carga real antes de prometer latencia en producción.")
    p("Persisten riesgos de falsas alarmas, fraudes omitidos, proxies desconocidos y cambio de población (benchmark europeo de 2013). No usar para bloquear pagos reales o sancionar personas sin validación y supervisión. Si una alternativa interpretable logra desempeño operativo comparable, la preferiríamos para facilitar control; estos resultados no prueban equivalencia con la logística.")
    p('<b>Referencias primarias.</b> [1] <link href="https://proceedings.mlr.press/v70/sundararajan17a.html">Sundararajan et al., ICML 2017 (IG)</link>. [2] <link href="https://captum.ai/api/integrated_gradients.html">Captum, documentación de atribución</link>. [3] <link href="https://arxiv.org/abs/1602.04938">Ribeiro et al., 2016 (LIME)</link>. [4] <link href="https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html">Lundberg y Lee, NeurIPS 2017 (SHAP)</link>. [5] <link href="https://shap.readthedocs.io/en/latest/api.html">SHAP, API de explainers</link>. [6] <link href="https://www.microsoft.com/en-us/research/publication/explaining-machine-learning-classifiers-through-diverse-counterfactual-examples/">Mothilal et al., 2020 (DiCE)</link>. [7] <link href="https://docs.seldon.ai/alibi-explain/explanations/methods/anchors">Ribeiro et al., 2018 / Alibi (Anchors)</link>. Consulta: 16/11/2026. Fuentes y notas ampliadas en explicabilidad/investigacion.md.', 'refs')
    if url:
        p(f'<b>Repositorio Git del proyecto:</b> <link href="{escape(url, quote=True)}">{escape(url)}</link>', "small")
    else:
        p("<b>Repositorio Git:</b> pendiente del enlace del equipo. Se incluye la copia local reproducible del proyecto; agregar el enlace HTTPS antes de entregar.", "small")

    def decorations(canvas, document):
        canvas.setStrokeColor(colors.HexColor("#cbdce3"))
        canvas.line(42, 753, 570, 753)
        canvas.setFont(normal_font, 7)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(42, 764, "FRAUDLENS / EXPLAINABLE AI")
        canvas.drawRightString(570, 764, "MODELO FINAL CONGELADO")
        canvas.drawString(42, 26, "CC3092 · Laboratorio 10 · Valladares / Cumes")
        canvas.drawRightString(570, 26, str(document.page))

    DEST.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(DEST), pagesize=letter, rightMargin=42, leftMargin=42,
        topMargin=49, bottomMargin=40, title="Laboratorio 10 - Explainable AI - FraudLens",
        author="Javier Valladares e Ian Cumes")
    document.build(story, onFirstPage=decorations, onLaterPages=decorations)
    (HERE / "informe.md").write_text("# Laboratorio 10: Explainable AI / FraudLens\n\n" + "\n".join(markdown))
    from pypdf import PdfReader
    count = len(PdfReader(DEST).pages)
    if count != 4:
        raise RuntimeError(f"El informe debe tener 4 páginas, tiene {count}; ajustar diseño antes de entregar")
    print(json.dumps({"pdf": str(DEST), "pages": count, "repository_url": url}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="URL HTTPS del mismo repositorio del proyecto final")
    args = parser.parse_args()
    build(args.repo)
