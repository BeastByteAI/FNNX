"""Models-from-code script: deterministic offline LangChain runnable.

Used by the langchain e2e tests. The runnable is a pure ``RunnableLambda``
(no LLM, no network). ``mlflow.models.set_model`` registers it so the
MLflow loader can rebuild it from this file alone — which is what makes
the converted FNNX package self-contained without ``code_paths``.
"""

from __future__ import annotations

import mlflow  # type: ignore[import-not-found]
from langchain_core.runnables import RunnableLambda  # type: ignore[import-not-found]


def _shout(record):
    """Uppercase ``record['payload']['text']`` and append ``!``.

    MLflow's pyfunc adapter validates the input against the saved object
    signature (a top-level ``payload`` Object containing ``text``) and hands
    the runnable a single record dict with that shape.
    """
    if isinstance(record, dict):
        payload = record.get("payload", record)
        if isinstance(payload, dict):
            text = payload.get("text", "")
        else:
            text = str(payload)
    else:
        text = str(record)
    return {"out": str(text).upper() + "!"}


chain = RunnableLambda(_shout)

mlflow.models.set_model(chain)
