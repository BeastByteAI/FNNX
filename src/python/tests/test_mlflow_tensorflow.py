"""End-to-end test for the MLflow → FNNX converter on a Keras functional model
with two named inputs.

Exercises the named-tensor branch of ``_map_input_schema``: when the signature
carries multiple TensorSpecs with explicit names, ``input_mode="tensor"`` is
selected with ``tensor_names=["a", "b"]``, the manifest has one ``NDJSON``
input per named tensor, and the wrapper assembles a dict-of-arrays payload for
``predict``.
"""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
import unittest

import pytest


pytest.importorskip("mlflow")
pytest.importorskip("tensorflow")


N_FEATURES_A = 3
N_FEATURES_B = 2


def _build_tiny_keras_model():
    """Build a deterministic two-input Keras functional model and a sample batch."""
    import numpy as np
    import tensorflow as tf  # type: ignore[import-not-found]

    tf.random.set_seed(0)

    input_a = tf.keras.Input(shape=(N_FEATURES_A,), name="a", dtype=tf.float32)
    input_b = tf.keras.Input(shape=(N_FEATURES_B,), name="b", dtype=tf.float32)
    concat = tf.keras.layers.Concatenate()([input_a, input_b])
    output = tf.keras.layers.Dense(2, activation="linear", name="out")(concat)
    model = tf.keras.Model(inputs={"a": input_a, "b": input_b}, outputs=output)

    rng = np.random.default_rng(0)
    sample_a = rng.random((4, N_FEATURES_A)).astype(np.float32)
    sample_b = rng.random((4, N_FEATURES_B)).astype(np.float32)
    return model, sample_a, sample_b


def _save_keras_model(tmp: str):
    """Save the Keras model with a two-named TensorSpec signature."""
    import mlflow  # type: ignore[import-not-found]
    import numpy as np
    from mlflow.models import ModelSignature  # type: ignore[import-not-found]
    from mlflow.types.schema import Schema, TensorSpec  # type: ignore[import-not-found]

    model, sample_a, sample_b = _build_tiny_keras_model()
    signature = ModelSignature(
        inputs=Schema(
            [
                TensorSpec(np.dtype("float32"), [-1, N_FEATURES_A], name="a"),
                TensorSpec(np.dtype("float32"), [-1, N_FEATURES_B], name="b"),
            ]
        ),
        outputs=Schema([TensorSpec(np.dtype("float32"), [-1, 2])]),
    )

    model_dir = os.path.join(tmp, "keras_model")
    mlflow.tensorflow.save_model(model, model_dir, signature=signature)  # type: ignore[attr-defined]
    return model_dir, model, sample_a, sample_b


class TestKerasNamedTensorRoundTrip(unittest.TestCase):
    """Named multi-input Keras model round-trips through FNNX losslessly."""

    def test_round_trip_and_manifest(self):
        import numpy as np

        from fnnx.extras.mlflow import package_mlflow_model
        from fnnx.runtime import Runtime

        with tempfile.TemporaryDirectory() as tmp:
            model_dir, model, sample_a, sample_b = _save_keras_model(tmp)
            out = os.path.join(tmp, "keras.fnnx")
            package_mlflow_model(model_dir, out)

            with tarfile.open(out, "r") as tar:
                variant_config = json.loads(
                    tar.extractfile("variant_config.json").read().decode()  # type: ignore[union-attr]
                )
                manifest = json.loads(
                    tar.extractfile("manifest.json").read().decode()  # type: ignore[union-attr]
                )

            cfg = variant_config["extra_values"]["fnnx_mlflow"]
            self.assertEqual(
                cfg["input_mode"],
                "tensor",
                f"expected tensor mode, got {cfg['input_mode']!r}",
            )
            self.assertEqual(cfg["tensor_names"], ["a", "b"])

            # Manifest: one NDJSON input per named tensor.
            self.assertEqual(len(manifest["inputs"]), 2)
            specs_by_name = {s["name"]: s for s in manifest["inputs"]}
            self.assertEqual(set(specs_by_name), {"a", "b"})

            spec_a = specs_by_name["a"]
            self.assertEqual(spec_a["content_type"], "NDJSON")
            self.assertEqual(spec_a["dtype"], "Array[float32]")
            self.assertEqual(spec_a["shape"], ["batch", N_FEATURES_A])

            spec_b = specs_by_name["b"]
            self.assertEqual(spec_b["content_type"], "NDJSON")
            self.assertEqual(spec_b["dtype"], "Array[float32]")
            self.assertEqual(spec_b["shape"], ["batch", N_FEATURES_B])

            rt = Runtime(out)
            result = rt.compute({"a": sample_a, "b": sample_b}, {})

        self.assertIn("predictions", result)
        expected = model.predict({"a": sample_a, "b": sample_b}, verbose=0)  # type: ignore[arg-type]
        actual = np.asarray(result["predictions"], dtype=np.float32)
        self.assertTrue(
            np.allclose(actual, expected, atol=1e-5),
            f"predictions diverged: expected={expected!r}, actual={actual!r}",
        )


if __name__ == "__main__":
    unittest.main()
