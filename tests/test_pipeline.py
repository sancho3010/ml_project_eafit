"""Test end-to-end del pipeline: fit, predict, save y reload."""

from __future__ import annotations

import joblib
from pathlib import Path
import numpy as np

from src.modeling.constants import CLIP_EDAD, MAPEO_CUARTOS, MAPEO_PERSONAS
from src.modeling.pipeline import PipelineML
from src.modeling.preprocessor import build_preprocessor


def build(seed: int = 42) -> PipelineML:
    return PipelineML(
        preprocessor=build_preprocessor(),
        mapeo_personas=MAPEO_PERSONAS,
        mapeo_cuartos=MAPEO_CUARTOS,
        clip_edad=CLIP_EDAD,
        n_trials_tree=2,
        cv_folds=2,
        parallel_trials=1,
        cores_per_model=1,
        random_seed=seed,
    )


def test_pipeline_builds_and_predicts(saber11_sample, saber11_target):
    """Smoke test: el pipeline ajusta y predice sin explotar."""
    dev = build()
    pipe = dev.build_pipeline(
        model_selected="lightGBM",
        params={"n_estimators": 20, "max_depth": 3, "num_leaves": 7},
        model_type="tree",
    )
    pipe.fit(saber11_sample, saber11_target)
    preds = pipe.predict(saber11_sample)

    assert preds.shape == saber11_target.shape
    assert np.isfinite(preds).all()


def test_pipeline_save_load_roundtrip(
    saber11_sample,
    saber11_target,
    tmp_path: Path
):
    """Serializar y recargar produce las mismas predicciones."""
    dev = build()
    pipe = dev.build_pipeline(
        model_selected="lightGBM",
        params={"n_estimators": 20, "max_depth": 3, "num_leaves": 7},
        model_type="tree",
    )
    pipe.fit(saber11_sample, saber11_target)

    out = tmp_path / "pipe.joblib"
    joblib.dump(pipe, out)
    loaded = joblib.load(out)

    a = pipe.predict(saber11_sample)
    b = loaded.predict(saber11_sample)
    np.testing.assert_allclose(a, b, rtol=1e-6)
