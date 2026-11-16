"""Contratos numéricos de la explicación del modelo final, sin reentrenar."""
import unittest
import numpy as np
import torch
from scipy.special import expit
from explicabilidad.analizar import attribute, load_frozen


class ExplainabilityTests(unittest.TestCase):
    def test_ig_linear_has_exact_signed_contributions(self):
        # Ejemplo analítico independiente de los resultados del laboratorio.
        weights = torch.tensor([2.0, -3.0, .5])
        f = lambda x: (x * weights).sum(dim=1) + 4
        x = np.array([[3, 2, -1], [-1, 1, 4]], dtype=np.float32)
        b = np.array([1, 0, 2], dtype=np.float32)
        cfg = dict(n_steps=32, max_steps=32, absolute_tolerance=1e-5,
                   relative_tolerance=0., example_batch_size=4, internal_batch_size=128)
        a, d, _ = attribute(f, x, b, cfg)
        np.testing.assert_allclose(a, (x - b) * weights.numpy(), atol=2e-6)
        np.testing.assert_allclose(d, 0, atol=2e-6)

    def test_calibrated_score_matches_saved_calibrator(self):
        import joblib
        from explicabilidad.analizar import ROOT
        metrics, _, score = load_frozen()
        cal = joblib.load(ROOT / f"models/calibration_{metrics['selected']}.joblib")
        x = torch.from_numpy(np.random.default_rng(23045).normal(size=(12, 29)).astype(np.float32))
        with torch.no_grad():
            p = expit(score(x).numpy().astype(np.float64))
            expected = cal.predict_proba(score.network(x).numpy().reshape(-1, 1))[:, 1]
        np.testing.assert_allclose(p, expected, atol=1e-7, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
