"""Verificación independiente de los entregables calculados; no entrena.

--entrega también exige haber reemplazado el aviso de enlace Git pendiente.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pypdf import PdfReader
from scipy.special import expit

from explicabilidad.analizar import HERE, ROOT, load_data, load_frozen, sha256
from explicabilidad.informe import DEST


def main(ready=False):
    audit = json.loads((HERE / "resultados/auditoria.json").read_text())
    for path, digest in audit["model_sha256"].items():
        assert sha256(ROOT / path) == digest, f"Artefacto cambió: {path}"
    assert sha256(ROOT / "results/metrics.json") == audit["original_metrics_sha256"]
    assert sha256(ROOT / "data/raw/creditcard.csv") == audit["data_sha256"]
    metrics, scaler, score = load_frozen()
    _, _, _, test, _, x = load_data(scaler)
    stored = np.load(HERE / "resultados/atribuciones.npz")
    ix = stored["test_indices"]
    assert len(np.unique(ix)) == audit["unique_explained"]
    assert len(stored["local_indices"]) == len(np.unique(stored["local_indices"])) == 5
    assert len(stored["global_indices"]) == audit["global_n"]
    assert len(stored["fraud_indices"]) == audit["test_frauds"]
    assert set(test.iloc[stored["fraud_indices"]].Class) == {1}
    with torch.no_grad():
        f = score(torch.from_numpy(x[ix])).numpy()
        base = float(score(torch.zeros(1, 29)).item())
        allp = expit(score(torch.from_numpy(x)).numpy().astype(np.float64))
    difference = f - base
    residual = stored["attributions"].sum(axis=1) - difference
    np.testing.assert_allclose(residual, stored["deltas"], atol=1e-5)
    cfg = audit["config"]
    assert (abs(residual) <= cfg["absolute_tolerance"] + cfg["relative_tolerance"] * abs(difference)).all()
    saved = pd.read_csv(ROOT / "results/test_predictions.csv")
    assert np.array_equal(saved.source_row, test.source_row)
    np.testing.assert_allclose(allp, saved[metrics["selected"]], atol=1e-6, rtol=1e-5)
    assert np.array_equal(allp >= audit["threshold"], saved[metrics["selected"]].to_numpy() >= audit["threshold"])
    outcomes = {case["outcome"] for case in audit["cases"]}
    assert {"VP", "FP", "FN", "VN"}.issubset(outcomes)
    for name in ["global", "locales", "sensibilidad"]:
        with Image.open(HERE / f"figuras/{name}.png") as im:
            im.verify()
        assert (HERE / f"figuras/{name}.svg").stat().st_size > 1000
    pdf = PdfReader(DEST)
    assert len(pdf.pages) == 4
    text = "\n".join(page.extract_text() for page in pdf.pages)
    assert "Cinco explicaciones locales" in text and "648 filas" in text
    pending = "pendiente del enlace" in text
    if ready:
        assert not pending, "Falta el enlace Git: regenerar el PDF con --repo URL_REAL antes de entregar"
    print(json.dumps(dict(pdf_pages=4, frozen_models_verified=True, original_predictions_verified=True,
        completeness_verified=True, explained_instances=len(ix), local_cases=5,
        git_link_pending=pending, ready_for_submission=not pending), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrega", action="store_true")
    main(parser.parse_args().entrega)
