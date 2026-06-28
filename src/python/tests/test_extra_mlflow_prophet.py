"""End-to-end test for the MLflow → FNNX converter on a Prophet model.

Exercises the datetime-column code path: ``ds`` is an MLflow ``datetime``
ColSpec, which falls outside the FNNX ``Array[...]`` token table and therefore
forces ``input_mode="json"`` (single permissive ``data`` input). The original
MLflow signature stays preserved in ``meta.json``, and a round-trip forecast
through ``fnnx.Runtime`` reproduces the model's own ``predict`` output for the
same future dataframe.
"""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
import unittest
import warnings

import pytest


pytest.importorskip("mlflow")
pytest.importorskip("prophet")
pytest.importorskip("pandas")


def _train_tiny_prophet():
    """Return ``(model, future_df)`` — deterministic linear trend, no seasonality."""
    import numpy as np
    import pandas as pd
    from prophet import Prophet  # type: ignore[import-not-found]

    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    history = pd.DataFrame({"ds": dates, "y": np.linspace(1.0, 5.0, 20)})

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        uncertainty_samples=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(history)

    future = pd.DataFrame(
        {"ds": pd.date_range("2024-01-21", periods=5, freq="D")}
    )
    return model, future


def _save_prophet_model(tmp: str):
    """Save the trained Prophet model with an inferred signature."""
    import mlflow  # type: ignore[import-not-found]
    from mlflow.models.signature import infer_signature  # type: ignore[import-not-found]

    model, future = _train_tiny_prophet()
    forecast = model.predict(future)
    sig = infer_signature(future, forecast)

    model_dir = os.path.join(tmp, "prophet_model")
    mlflow.prophet.save_model(model, model_dir, signature=sig)  # type: ignore[attr-defined]
    return model_dir, model, future


class TestProphetDatetimeJsonMode(unittest.TestCase):
    """Datetime column → json input_mode; forecast round-trips losslessly."""

    def test_round_trip_and_input_mode(self):
        import numpy as np

        from fnnx.extras.mlflow import package_mlflow_model
        from fnnx.runtime import Runtime

        with tempfile.TemporaryDirectory() as tmp:
            model_dir, model, future = _save_prophet_model(tmp)
            out = os.path.join(tmp, "prophet.fnnx")
            package_mlflow_model(model_dir, out)

            with tarfile.open(out, "r") as tar:
                variant_config = json.loads(
                    tar.extractfile("variant_config.json").read().decode()  # type: ignore[union-attr]
                )
                manifest = json.loads(
                    tar.extractfile("manifest.json").read().decode()  # type: ignore[union-attr]
                )
                meta = json.loads(
                    tar.extractfile("meta.json").read().decode()  # type: ignore[union-attr]
                )

            cfg = variant_config["extra_values"]["fnnx_mlflow"]
            self.assertEqual(
                cfg["input_mode"],
                "json",
                f"expected json mode for datetime column, got {cfg['input_mode']!r}",
            )

            self.assertEqual(len(manifest["inputs"]), 1)
            self.assertEqual(manifest["inputs"][0]["name"], "data")
            self.assertEqual(manifest["inputs"][0]["content_type"], "JSON")
            self.assertEqual(
                manifest["inputs"][0]["dtype"], "ext::mlflow::input"
            )

            # Original MLflow signature (datetime column) preserved verbatim.
            sig_inputs = meta[0]["payload"]["signature"]["inputs"]
            self.assertEqual(len(sig_inputs), 1)
            self.assertEqual(sig_inputs[0]["type"], "datetime")
            self.assertEqual(sig_inputs[0]["name"], "ds")

            # Meta records the prophet loader for provenance.
            self.assertEqual(
                meta[0]["payload"]["loader_module"], "mlflow.prophet"
            )

            rt = Runtime(out)
            # json mode: ``inputs["data"]`` is forwarded straight to predict().
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = rt.compute({"data": future}, {})
                expected = model.predict(future)

        self.assertIn("predictions", result)

        # Predictions normalized to list-of-records via _to_jsonable(DataFrame).
        actual_yhat = np.asarray([rec["yhat"] for rec in result["predictions"]])
        expected_yhat = expected["yhat"].to_numpy()
        self.assertTrue(
            np.allclose(actual_yhat, expected_yhat, atol=1e-8),
            f"yhat diverged: expected={expected_yhat!r}, actual={actual_yhat!r}",
        )


if __name__ == "__main__":
    unittest.main()
