"""End-to-end tests for the MLflow → FNNX converter on a PyTorch ``nn.Module``.

Covers the two self-containment paths for pickle-based pytorch models whose
class lives outside ``__main__``:

* Saved with ``code_paths=[<file>]`` → the class source is embedded under
  ``mlflow_model/code/`` and the wrapper can re-import it at load time. The
  round-trip predictions match the source model and no code-availability
  warning is emitted.
* Saved without ``code_paths`` → the build still succeeds, no ``code/``
  directory is embedded, and a self-containment warning is emitted naming
  the by-reference pickle flavor.
"""

from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

import pytest


pytest.importorskip("mlflow")
pytest.importorskip("torch")


# Make the fixture importable as ``torch_net`` (mlflow.pytorch.save_model
# pickles the model by reference, so the class's module must be importable at
# save time; ``code_paths`` then re-exposes it at load time).
_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "_mlflow_fixtures")
if _FIXTURES_DIR not in sys.path:
    sys.path.insert(0, _FIXTURES_DIR)


def _build_tiny_net():
    """Build a fixed-weight TinyNet plus a deterministic input batch."""
    import numpy as np
    import torch  # type: ignore[import-not-found]

    from torch_net import N_FEATURES, TinyNet  # type: ignore[import-not-found]

    torch.manual_seed(0)
    model = TinyNet()
    model.eval()

    rng = np.random.default_rng(0)
    sample = rng.random((5, N_FEATURES), dtype=np.float64).astype(np.float32)
    return model, sample, N_FEATURES


def _native_forward(model, sample):
    import torch  # type: ignore[import-not-found]

    with torch.no_grad():
        return model(torch.from_numpy(sample)).numpy()


def _save_pytorch_model(tmp: str, *, with_code_paths: bool):
    """Save a tiny pytorch model with a TensorSpec signature."""
    import mlflow  # type: ignore[import-not-found]
    import numpy as np
    from mlflow.models import ModelSignature  # type: ignore[import-not-found]
    from mlflow.types.schema import Schema, TensorSpec  # type: ignore[import-not-found]

    model, sample, n_features = _build_tiny_net()
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float32"), [-1, n_features])]),
        outputs=Schema([TensorSpec(np.dtype("float32"), [-1, 2])]),
    )

    model_dir = os.path.join(tmp, "torch_model")
    # mlflow.pytorch always serializes via torch.save (pickle, by-reference),
    # which is exactly what exercises the by-reference self-containment
    # heuristic. (There is no `serialization_format` arg on this flavor — it
    # belongs to mlflow.sklearn — and passing it raises in torch.save.)
    kwargs: dict = {"signature": signature}
    if with_code_paths:
        kwargs["code_paths"] = [os.path.join(_FIXTURES_DIR, "torch_net.py")]
    mlflow.pytorch.save_model(model, model_dir, **kwargs)  # type: ignore[attr-defined]
    return model_dir, model, sample


class TestPyTorchWithCodePaths(unittest.TestCase):
    """code_paths embeds the class source → round-trip works, no warning."""

    def test_round_trip_and_no_warning(self):
        import numpy as np

        from fnnx.extras.mlflow import package_mlflow_model
        from fnnx.runtime import Runtime

        with tempfile.TemporaryDirectory() as tmp:
            model_dir, model, sample = _save_pytorch_model(tmp, with_code_paths=True)
            out = os.path.join(tmp, "torch.fnnx")

            with mock.patch("fnnx.extras.mlflow.console.warn") as warn_mock:
                package_mlflow_model(model_dir, out)

            # ``code/`` directory was embedded verbatim under mlflow_model.
            with tarfile.open(out, "r") as tar:
                names = set(tar.getnames())
            self.assertTrue(
                any(
                    n.startswith("variant_artifacts/extra_files/mlflow_model/code/")
                    and n.endswith("torch_net.py")
                    for n in names
                ),
                f"expected mlflow_model/code/torch_net.py in package, got {sorted(names)!r}",
            )

            # No "by reference" / Hub-reference self-containment warning fires.
            warned = [c.args[0] for c in warn_mock.call_args_list]
            for msg in warned:
                self.assertNotIn("serializes Python objects by reference", msg)
                self.assertNotIn("referenced from the Hugging Face Hub", msg)

            rt = Runtime(out)
            result = rt.compute({"input": sample}, {})

        self.assertIn("predictions", result)
        expected = _native_forward(model, sample)
        actual = np.asarray(result["predictions"], dtype=np.float32)
        self.assertTrue(
            np.allclose(actual, expected, atol=1e-5),
            f"predictions diverged: expected={expected!r}, actual={actual!r}",
        )


class TestPyTorchWithoutCodePaths(unittest.TestCase):
    """Without code_paths, the build succeeds but a warning is emitted."""

    def test_self_containment_warning(self):
        from fnnx.extras.mlflow import package_mlflow_model

        with tempfile.TemporaryDirectory() as tmp:
            model_dir, _, _ = _save_pytorch_model(tmp, with_code_paths=False)
            out = os.path.join(tmp, "torch_no_code.fnnx")

            with mock.patch("fnnx.extras.mlflow.console.warn") as warn_mock:
                package_mlflow_model(model_dir, out)

            # No ``code/`` subdir was embedded.
            with tarfile.open(out, "r") as tar:
                names = set(tar.getnames())
            self.assertFalse(
                any(
                    n.startswith("variant_artifacts/extra_files/mlflow_model/code/")
                    for n in names
                ),
                f"unexpected code/ subdir in package: {sorted(names)!r}",
            )

            warned = [c.args[0] for c in warn_mock.call_args_list]
            self.assertTrue(
                any("serializes Python objects by reference" in msg for msg in warned),
                f"expected by-reference pickle warning, got {warned!r}",
            )


if __name__ == "__main__":
    unittest.main()
