"""Regenera explicaciones y auditorías SIN entrenar ni modificar el modelo.

Ejecutar desde la raíz del proyecto: python -m explicabilidad.analizar
Las etiquetas de test sirven para seleccionar y analizar casos, nunca como entrada.
La atribución explica log-odds calibrados; NO son puntos porcentuales ni causalidad.
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
import time
from pathlib import Path

import captum
import joblib
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from scipy.special import expit, logit
from scipy.stats import spearmanr

from src.model import FEATURES, FraudMLP, feature_array

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "resultados"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


class CalibratedScore(torch.nn.Module):
    """F(z) = a * MLP(z) + b; sigmoid(F) es la probabilidad de fraude.

    Mantener la calibración evita explicar un puntaje distinto al de la demo.
    La sigmoide se omite SOLO en la escala de atribución, para evitar saturación.
    El umbral de probabilidad se transforma mediante logit para compararlo con F.
    """

    def __init__(self, network: FraudMLP, calibrator):
        super().__init__()
        self.network = network
        self.a = float(calibrator.coef_[0, 0])
        self.b = float(calibrator.intercept_[0])

    def forward(self, x):
        return self.a * self.network(x) + self.b


def load_frozen():
    metrics = json.loads((ROOT / "results/metrics.json").read_text())
    selected = metrics["selected"]
    checkpoint = torch.load(ROOT / f"models/model_{selected}.pt", map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    net = FraudMLP(cfg["hidden"], cfg["dropout"])
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    scaler = joblib.load(ROOT / "models/scaler.joblib")
    calibrator = joblib.load(ROOT / f"models/calibration_{selected}.joblib")
    return metrics, scaler, CalibratedScore(net, calibrator).eval()


def load_data(scaler):
    """Reconstruye las particiones originales por ID, sin volver a dividir datos."""
    raw = pd.read_csv(ROOT / "data/raw/creditcard.csv")
    eda = json.loads((ROOT / "results/eda.json").read_text())
    assert sha256(ROOT / "data/raw/creditcard.csv") == eda["sha256"], "CSV distinto al experimento original"
    manifest_path = ROOT / "data/processed/split_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        # Git excluye los CSV completos. Reconstruir exactamente el manifiesto
        # del experimento original NO implica ajustar escaladores ni redes.
        indexed = raw.assign(source_row=np.arange(len(raw)))
        columns = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
        clean = indexed.drop_duplicates(columns).sort_values(["Time", "source_row"], kind="stable")
        bounds = [float(clean.Time.quantile(q)) for q in [.6, .7, .8]]
        groups = np.searchsorted(bounds, clean.Time.to_numpy(), side="right")
        names = np.array(["train", "calibration", "validation", "test"])
        manifest = pd.DataFrame({"source_row": clean.source_row.to_numpy(), "split": names[groups]})
        for name in names:
            assert int((manifest.split == name).sum()) == eda["split"][name]["rows"]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
    assert manifest.source_row.is_unique
    train_ids = manifest.loc[manifest.split == "train", "source_row"].to_numpy()
    test_ids = manifest.loc[manifest.split == "test", "source_row"].to_numpy()
    train = raw.iloc[train_ids].copy()
    test = raw.iloc[test_ids].copy()
    train["source_row"] = train_ids
    test["source_row"] = test_ids
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    assert set(train_ids).isdisjoint(test_ids)
    assert train.Time.max() < test.Time.min()
    # Comprobar el escalador guardado; no se ajusta uno nuevo.
    train_features = feature_array(train)
    np.testing.assert_allclose(train_features.mean(axis=0), scaler.mean_, rtol=1e-8, atol=1e-8)
    xt = scaler.transform(train_features).astype(np.float32)
    xv = scaler.transform(feature_array(test)).astype(np.float32)
    assert np.isfinite(xt).all() and np.isfinite(xv).all()
    assert len(FEATURES) == 29 and not {"Class", "source_row", "Time"}.intersection(FEATURES)
    return raw, manifest, train, test, xt, xv


def choose_cases(test, probability, threshold):
    """Reglas deterministas, declaradas antes de observar las atribuciones.

    Cinco casos distintos: VP más confiado, FP más confiado, FN más confiado
    (mínima p), instancia más próxima al umbral y fraude VP de monto > 100.
    Empates: orden temporal del manifiesto (source_row dentro de cada segundo).
    """
    y = test.Class.to_numpy()
    selected = []

    def pick(label, mask, objective, description):
        candidates = np.flatnonzero(mask)
        candidates = candidates[~np.isin(candidates, [r["test_index"] for r in selected])]
        if not len(candidates):
            raise ValueError(f"No hay un caso distinto para: {label}")
        index = int(candidates[np.argmin(objective[candidates])])
        selected.append(dict(case=label, test_index=index, selection_rule=description))

    pick("C1", (y == 1) & (probability >= threshold), -probability, "VP con mayor probabilidad de fraude")
    pick("C2", (y == 0) & (probability >= threshold), -probability, "FP con mayor probabilidad de fraude")
    pick("C3", (y == 1) & (probability < threshold), probability, "FN con menor probabilidad de fraude")
    pick("C4", np.ones(len(test), dtype=bool), abs(probability - threshold), "Caso restante más cercano al umbral")
    pick("C5", (test.Amount.to_numpy() > 100) & (y == 1) & (probability >= threshold),
         -probability, "VP restante de monto > 100 con mayor probabilidad")
    return selected


def attribute(score, x, baseline, cfg):
    """IG por Gauss-Legendre y refinamiento solo si falla la completitud.

    La tolerancia es abs(delta) <= 0.02 + 0.001 * abs(F(x)-F(base)).
    Este chequeo controla la suma; no demuestra exactitud de cada componente.
    Se guardan los pasos finales de cada instancia. Falla si agota max_steps.
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    baseline = torch.as_tensor(baseline, dtype=torch.float32).reshape(1, -1)
    ig = IntegratedGradients(score, multiply_by_inputs=True)
    attrs = np.empty(tuple(x.shape), dtype=np.float32)
    deltas = np.empty(len(x), dtype=np.float32)
    steps = np.full(len(x), cfg["n_steps"], dtype=int)
    with torch.no_grad():
        change = (score(x) - score(baseline)).numpy()
    limits = cfg["absolute_tolerance"] + cfg["relative_tolerance"] * abs(change)
    pending = np.arange(len(x))
    n_steps = cfg["n_steps"]
    while len(pending):
        for indices in np.array_split(pending, max(1, int(np.ceil(len(pending) / cfg["example_batch_size"])))):
            a, d = ig.attribute(
                x[indices].clone().requires_grad_(), baselines=baseline,
                n_steps=n_steps, method="gausslegendre",
                internal_batch_size=cfg["internal_batch_size"], return_convergence_delta=True,
            )
            attrs[indices] = a.detach().numpy()
            deltas[indices] = d.detach().numpy()
            steps[indices] = n_steps
        pending = np.flatnonzero(abs(deltas) > limits)
        if not len(pending):
            break
        if n_steps >= cfg["max_steps"]:
            raise RuntimeError(f"IG no converge en {len(pending)} casos; revisar delta/pasos antes de entregar")
        n_steps = min(2 * n_steps, cfg["max_steps"])
    return attrs, deltas, steps


def benchmark(score, x, cfg):
    """Tiempos CPU en memoria: excluyen disco, preproceso, red y renderizado."""
    ig = IntegratedGradients(score)
    one = torch.from_numpy(x[:1])
    baseline = torch.zeros_like(one)
    for _ in range(5):
        with torch.no_grad():
            score(one)
        ig.attribute(one.requires_grad_(), baselines=baseline, n_steps=cfg["n_steps"],
                     internal_batch_size=cfg["internal_batch_size"])
    timings = {"inference": [], "ig": [], "occlusion": []}
    perturbed = one.detach().repeat(29, 1)
    perturbed[torch.arange(29), torch.arange(29)] = 0
    for _ in range(cfg["latency_repetitions"]):
        begin = time.perf_counter()
        with torch.no_grad():
            score(one)
        timings["inference"].append(1000 * (time.perf_counter() - begin))
        begin = time.perf_counter()
        ig.attribute(one, baselines=baseline, n_steps=cfg["n_steps"],
                     internal_batch_size=cfg["internal_batch_size"])
        timings["ig"].append(1000 * (time.perf_counter() - begin))
        begin = time.perf_counter()
        with torch.no_grad():
            score(one)
            score(perturbed)
        timings["occlusion"].append(1000 * (time.perf_counter() - begin))
    return {key: dict(p50_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                      repetitions=len(values)) for key, values in timings.items()}


def main():
    start_total = time.perf_counter()
    cfg = json.loads((HERE / "config.json").read_text())
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["torch_threads"])
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    tracked = sorted((ROOT / "models").glob("*"))
    before = {str(p.relative_to(ROOT)): sha256(p) for p in tracked if p.is_file()}
    metrics, scaler, score = load_frozen()
    raw, manifest, train, test, xt, xv = load_data(scaler)
    selected = metrics["selected"]
    threshold = metrics["models"][selected]["threshold"]
    with torch.no_grad():
        scores = score(torch.from_numpy(xv)).numpy()
        probabilities = expit(scores.astype(np.float64))
    original = pd.read_csv(ROOT / "results/test_predictions.csv")
    assert np.array_equal(original.source_row, test.source_row)
    parity_error = float(np.max(abs(original[selected].to_numpy() - probabilities)))
    assert parity_error < 1e-6
    # Decisiones exactas: no se cambia el umbral ni se selecciona una nueva política.
    assert np.array_equal(original[selected].to_numpy() >= threshold, probabilities >= threshold)
    cases = choose_cases(test, probabilities, threshold)
    local_ix = np.array([r["test_index"] for r in cases])
    rng = np.random.default_rng(cfg["seed"])
    global_ix = np.sort(rng.choice(len(test), size=cfg["global_sample_size"], replace=False))
    fraud_ix = np.flatnonzero(test.Class.to_numpy() == 1)
    union_ix = np.unique(np.concatenate([global_ix, fraud_ix, local_ix]))
    baseline = np.zeros(29, dtype=np.float32)
    print(f"Modelo {selected} congelado. IG de {len(union_ix)} instancias únicas...", flush=True)
    begin = time.perf_counter()
    attrs, deltas, steps = attribute(score, xv[union_ix], baseline, cfg)
    ig_batch_seconds = time.perf_counter() - begin
    positions = {int(index): pos for pos, index in enumerate(union_ix)}
    local_pos = [positions[int(i)] for i in local_ix]
    global_pos = [positions[int(i)] for i in global_ix]
    fraud_pos = [positions[int(i)] for i in fraud_ix]
    with torch.no_grad():
        base_score = float(score(torch.from_numpy(baseline[None])).item())
    for row, pos in zip(cases, local_pos):
        index = row["test_index"]
        datum = test.iloc[index]
        y = int(datum.Class)
        pred = int(probabilities[index] >= threshold)
        kind = {(1, 1): "VP", (0, 1): "FP", (1, 0): "FN", (0, 0): "VN"}[(y, pred)]
        row.update(source_row=int(datum.source_row), label=y, prediction=pred, outcome=kind,
                   amount=float(datum.Amount), probability=float(probabilities[index]),
                   score=float(scores[index]), margin_log_odds=float(scores[index] - logit(threshold)),
                   delta=float(deltas[pos]), n_steps=int(steps[pos]),
                   top_positive=[FEATURES[i] for i in np.argsort(-attrs[pos])[:3]],
                   top_negative=[FEATURES[i] for i in np.argsort(attrs[pos])[:3]])
    pd.DataFrame(cases).to_csv(OUT / "casos.csv", index=False)
    pd.DataFrame(attrs[local_pos], index=[r["case"] for r in cases], columns=FEATURES).rename_axis("case").to_csv(OUT / "atribuciones_locales.csv")
    local_values = test.iloc[local_ix][["source_row", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]]
    local_values.to_csv(OUT / "entradas_locales.csv", index=False)
    pd.DataFrame(xv[local_ix], columns=FEATURES).assign(case=[r["case"] for r in cases]).to_csv(OUT / "entradas_estandarizadas.csv", index=False)
    pd.DataFrame({"source_row": test.iloc[global_ix].source_row.to_numpy(),
                  "label": test.iloc[global_ix].Class.to_numpy()}).to_csv(OUT / "muestra_global.csv", index=False)
    global_table = pd.DataFrame({"feature": FEATURES,
        "mean_abs_ig_random": np.abs(attrs[global_pos]).mean(axis=0),
        "mean_abs_ig_all_frauds": np.abs(attrs[fraud_pos]).mean(axis=0),
        "mean_signed_ig_random": attrs[global_pos].mean(axis=0)})
    global_table.sort_values("mean_abs_ig_random", ascending=False).to_csv(OUT / "importancia_global.csv", index=False)

    # Sensibilidad a dos referencias de train: mediana legítima y una fila real
    # legítima próxima a ella. No se buscan referencias que den una explicación deseada.
    legitimate = train.Class.to_numpy() == 0
    median = np.median(xt[legitimate], axis=0)
    legitimate_ix = np.flatnonzero(legitimate)
    nearest = int(legitimate_ix[np.argmin(np.sum((xt[legitimate] - median) ** 2, axis=1))])
    references = {"media_train": baseline, "mediana_legitima_train": median,
                  "fila_legitima_train": xt[nearest]}
    stability = []
    sensitivity_attributions = {}
    original_top = [set(np.argsort(-abs(a))[:5]) for a in attrs[local_pos]]
    for name, reference in references.items():
        aa, dd, ss = (attrs[local_pos], deltas[local_pos], steps[local_pos]) if name == "media_train" else attribute(score, xv[local_ix], reference, cfg)
        sensitivity_attributions[name] = aa
        with torch.no_grad():
            ref_score = float(score(torch.as_tensor(reference).reshape(1, -1)).item())
        for k, row in enumerate(cases):
            top = set(np.argsort(-abs(aa[k]))[:5])
            stability.append(dict(case=row["case"], baseline=name, baseline_probability=float(expit(ref_score)),
                                  top5_overlap=len(top & original_top[k]),
                                  spearman_abs=float(spearmanr(abs(attrs[local_pos[k]]), abs(aa[k])).statistic),
                                  delta=float(dd[k]), n_steps=int(ss[k]),
                                  top_feature=FEATURES[int(np.argmax(abs(aa[k])))],
                                  baseline_source_row=int(train.iloc[nearest].source_row) if name == "fila_legitima_train" else None))
    pd.DataFrame(stability).to_csv(OUT / "sensibilidad_referencia.csv", index=False)
    save_json(OUT / "referencias.json", {name: ref.tolist() for name, ref in references.items()})

    # Sustituciones univariadas: diagnóstico del mecanismo del error, no recourse
    # causal ni cambios factibles para titulares de una transacción histórica.
    perturbations = []
    for case, pos in zip(cases, local_pos):
        index = case["test_index"]
        replaced = np.repeat(xv[index:index + 1], 29, axis=0)
        replaced[np.arange(29), np.arange(29)] = baseline
        with torch.no_grad():
            pp = expit(score(torch.from_numpy(replaced)).numpy().astype(np.float64))
        for j, name in enumerate(FEATURES):
            perturbations.append(dict(case=case["case"], feature=name,
                probability_original=case["probability"], probability_replaced=float(pp[j]),
                prediction_replaced=int(pp[j] >= threshold),
                corrects_observed_label=bool(int(pp[j] >= threshold) == case["label"]),
                ig=float(attrs[pos, j])))
    pd.DataFrame(perturbations).to_csv(OUT / "sustituciones_diagnosticas.csv", index=False)

    # Comprobar fuga accesible, duplicados, grupos operativos y valores extremos.
    audit_columns = [*[f"V{i}" for i in range(1, 29)], "Amount"]
    training_hash = pd.util.hash_pandas_object(train[audit_columns], index=False)
    testing_hash = pd.util.hash_pandas_object(test[audit_columns], index=False)
    duplicate_inputs = int(testing_hash.isin(set(training_hash)).sum())
    # Confirmar igualdad exacta de columnas después del filtro por hash.
    overlap_mask = testing_hash.isin(set(training_hash)).to_numpy()
    exact_overlap = test.loc[overlap_mask, ["source_row", "Class", *audit_columns]].merge(
        train[audit_columns].drop_duplicates(), on=audit_columns, how="inner", validate="many_to_one")
    assert len(exact_overlap) == duplicate_inputs
    duplicate_frauds = int(exact_overlap.Class.sum())
    exact_overlap[["source_row", "Class"]].to_csv(OUT / "coincidencias_train_test.csv", index=False)
    subgroups = []
    for name, mask in [("Amount <= 20", test.Amount.to_numpy() <= 20),
                       ("20 < Amount <= 100", (test.Amount.to_numpy() > 20) & (test.Amount.to_numpy() <= 100)),
                       ("Amount > 100", test.Amount.to_numpy() > 100)]:
        yy = test.Class.to_numpy()[mask]
        pred = probabilities[mask] >= threshold
        tp, fp = int(np.sum(pred & (yy == 1))), int(np.sum(pred & (yy == 0)))
        fn = int(np.sum(~pred & (yy == 1)))
        subgroups.append(dict(group=name, n=int(mask.sum()), frauds=int(yy.sum()), tp=tp, fp=fp, fn=fn,
                              recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1)))
    pd.DataFrame(subgroups).to_csv(OUT / "subgrupos.csv", index=False)
    latency = benchmark(score, xv[local_ix], cfg)
    # Integridad al terminar: todos los modelos y objetos de preproceso iguales.
    after = {str(p.relative_to(ROOT)): sha256(p) for p in tracked if p.is_file()}
    assert before == after
    np.savez_compressed(OUT / "atribuciones.npz", test_indices=union_ix, attributions=attrs,
                        deltas=deltas, n_steps=steps, global_indices=global_ix, local_indices=local_ix,
                        fraud_indices=fraud_ix, **{f"baseline_{key}": value for key, value in sensitivity_attributions.items()})
    audit = dict(model=selected, seed=cfg["seed"], threshold=threshold,
        threshold_log_odds=float(logit(threshold)), calibration=dict(a=score.a, b=score.b),
        baseline_score=base_score, baseline_probability=float(expit(base_score)),
        config=cfg, cases=cases, test_n=len(test), test_frauds=int(test.Class.sum()),
        train_n=len(train), global_n=len(global_ix), global_frauds=int(test.iloc[global_ix].Class.sum()),
        unique_explained=len(union_ix), all_frauds_explained=len(fraud_ix),
        delta_abs_max=float(np.max(abs(deltas))), delta_abs_p50=float(np.median(abs(deltas))),
        delta_abs_p95=float(np.percentile(abs(deltas), 95)),
        steps_histogram={str(n): int(np.sum(steps == n)) for n in np.unique(steps)},
        ig_batch_seconds=ig_batch_seconds, latency=latency,
        parity_probability_error_max=parity_error, models_unchanged=True,
        model_sha256=before, data_sha256=sha256(ROOT / "data/raw/creditcard.csv"),
        manifest_sha256=sha256(ROOT / "data/processed/split_manifest.csv"),
        original_metrics_sha256=sha256(ROOT / "results/metrics.json"),
        temporal_separation=True, train_test_identical_input_rows=duplicate_inputs,
        train_test_identical_input_frauds=duplicate_frauds,
        maximum_standardized_input_by_case={r["case"]: float(abs(xv[r["test_index"]]).max()) for r in cases},
        environment=dict(python=platform.python_version(), torch=torch.__version__, captum=captum.__version__,
                         numpy=np.__version__, platform=platform.platform(), device="CPU", threads=cfg["torch_threads"]),
        total_seconds=time.perf_counter() - start_total)
    save_json(OUT / "auditoria.json", audit)
    from explicabilidad.figuras import make_figures
    make_figures()
    print(json.dumps({"output": str(OUT), "unique_explained": len(union_ix),
                      "ig_batch_seconds": ig_batch_seconds, "max_abs_delta": audit["delta_abs_max"],
                      "latency": latency, "models_unchanged": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
