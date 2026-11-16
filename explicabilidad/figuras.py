"""Figuras publicables del laboratorio, siempre derivadas de resultados guardados."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "resultados"
FIG = HERE / "figuras"
INK = "#153247"
TEAL = "#007d83"
RED = "#bc4b36"
BLUE = "#347ba0"
MUTED = "#566875"


def finish(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=220, facecolor="white")
    fig.savefig(FIG / f"{name}.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def make_figures():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": "#a5b5be",
        "xtick.color": MUTED, "ytick.color": INK, "svg.hashsalt": "fraudlens-lab10"})
    a = json.loads((OUT / "auditoria.json").read_text())
    data = pd.read_csv(OUT / "importancia_global.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.95), layout="constrained")
    for ax, column, title, color in zip(axes,
            ["mean_abs_ig_random", "mean_abs_ig_all_frauds"],
            [f"Muestra aleatoria: n={a['global_n']:,}", f"Todos los fraudes: n={a['all_frauds_explained']}"],
            [TEAL, RED]):
        top = data.nlargest(8, column).iloc[::-1]
        ax.barh(top.feature, top[column], color=color, height=.65)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("Media de |IG| (log-odds)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    finish(fig, "global")

    cases = a["cases"]
    attrs = pd.read_csv(OUT / "atribuciones_locales.csv").set_index("case")
    titles = ["Acierto de alta confianza", "Falsa alarma", "Fraude omitido", "Cercano al umbral", "Subgrupo: monto > 100"]
    fig, axes = plt.subplots(5, 1, figsize=(7.25, 5.9))
    fig.subplots_adjust(left=.125, right=.985, top=.965, bottom=.08, hspace=1.13)
    rows = []
    for case in cases:
        row = attrs.loc[case["case"]]
        top = np.argsort(-abs(row.to_numpy()))[:4]
        values = list(row.iloc[top]) + [float(row.drop(row.index[top]).sum())]
        names = list(row.index[top]) + ["Otras 25"]
        rows.append((values, names))
    bound = max(abs(np.array([row[0] for row in rows])).ravel()) * 1.14
    for ax, case, title, (values, names) in zip(axes, cases, titles, rows):
        y = np.arange(5)
        ax.barh(y, values, color=[RED if v >= 0 else BLUE for v in values], height=.7)
        ax.set_yticks(y, names, fontsize=7.4)
        ax.invert_yaxis()
        ax.axvline(0, color=INK, lw=.6)
        ax.set_xlim(-bound, bound)
        ax.tick_params(axis="x", labelsize=7, pad=1, length=2)
        ax.grid(axis="x", alpha=.12)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=4)
        p = 100 * case["probability"]
        ax.set_title(f"{case['case']}  {title}  |  TX-{case['source_row']:06d}  |  {case['outcome']}  |  p={p:.4f}%",
                     loc="left", fontsize=8.3, fontweight="bold", pad=5)
        for yi, value in zip(y, values):
            ax.text(value + np.sign(value) * bound * .018, yi, f"{value:+.2f}",
                    va="center", ha="left" if value >= 0 else "right", fontsize=7)
    axes[-1].set_xlabel("Atribución IG: disminuye el puntaje  /  aumenta el puntaje (log-odds)", fontsize=8)
    finish(fig, "locales")

    stability = pd.read_csv(OUT / "sensibilidad_referencia.csv")
    pivot = stability.pivot(index="case", columns="baseline", values="top5_overlap")
    columns = ["mediana_legitima_train", "fila_legitima_train"]
    fig, ax = plt.subplots(figsize=(7.25, 1.95), layout="constrained")
    vals = pivot[columns].to_numpy().T
    im = ax.imshow(vals, vmin=0, vmax=5, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(5), pivot.index, fontsize=9)
    ax.set_yticks([0, 1], ["Mediana legítima", "Fila legítima real"], fontsize=8)
    for i in range(2):
        for j in range(5):
            ax.text(j, i, f"{int(vals[i, j])}/5", ha="center", va="center",
                    color="white" if vals[i, j] >= 4 else INK, fontsize=10)
    ax.set_title("Coincidencia del top 5 frente a la media de train", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=.025, ticks=[0, 5])
    finish(fig, "sensibilidad")


if __name__ == "__main__":
    make_figures()
