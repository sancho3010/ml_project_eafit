"""
Pipeline de ML para la predicción de puntaje Saber 11.

``PipelineML`` orquesta:

1. Tuning con Optuna (TPE + median pruner) sobre LightGBM, XGBoost y
   CatBoost. Cada trial reentrena el pipeline completo por fold, así que
   imputers, encoders y agregados se ajustan solo en train.
2. Reentrenamiento del champion con todo el train.
3. Evaluación única en el test holdout.
4. Interpretabilidad con SHAP sobre train.
5. Persistencia con joblib.

El pipeline interno es siempre:
``deterministic → group_aggs → preprocessor → model``.
"""

from __future__ import annotations

import os
import json
from typing import Any

from catboost import CatBoostRegressor
import joblib
from lightgbm import LGBMRegressor
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from optuna import Trial, create_study
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import xgboost as xgb

from src.modeling.transformers import (
    DeterministicFeatureProcessor,
    GroupAggregatesTransformer,
)



class PipelineML:
    """Orquestador de tuning, evaluación e interpretabilidad."""

    SUPPORTED_MODELS = ("lightGBM", "xgboost", "catboost")

    def __init__(
        self,
        preprocessor,
        mapeo_personas: dict,
        mapeo_cuartos: dict,
        clip_edad: tuple[int, int],
        n_trials_tree: int = 30,
        cv_folds: int = 4,
        parallel_trials: int | None = None,
        cores_per_model: int | None = None,
        use_gpu: bool = False,
        random_seed: int = 42,
    ) -> None:
        # Auto-detección: parallel_trials × cores_per_model ≈ cores físicos.
        n_cpus = os.cpu_count() or 4
        if parallel_trials is None:
            # GPU: 1 trial (1 sola GPU). CPU: 2 trials por defecto.
            parallel_trials = 1 if use_gpu else min(2, n_cpus // 2)

        if cores_per_model is None:
            cores_per_model = max(1, n_cpus // parallel_trials)

        self.preprocessor = preprocessor
        self.mapeo_personas = mapeo_personas
        self.mapeo_cuartos = mapeo_cuartos
        self.clip_edad = clip_edad

        self.random_seed = random_seed
        self.n_trials_tree = n_trials_tree
        self.cv_folds = cv_folds
        self.parallel_trials = parallel_trials
        self.cores_per_model = cores_per_model
        self.use_gpu = use_gpu

        self.studies: dict[str, Any] = {}

        device_label = "GPU" if use_gpu else "CPU"
        logger.info(
            f"Maquina detectada: {n_cpus} cores. Modo: {device_label}. "
            f"parallel_trials={parallel_trials}, "
            f"cores_per_model={cores_per_model}, "
            f"concurrencia total={parallel_trials * cores_per_model}"
        )

    # ------------------------------------------------------------------
    # Definición del espacio de búsqueda
    # ------------------------------------------------------------------

    def build_model_grid(self, trial: Trial, model_selected: str) -> dict:
        if model_selected == "lightGBM":
            return {
                "model_type": "tree",
                "params_grid": {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                    "max_depth": trial.suggest_int("max_depth", 4, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                    "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                },
            }

        if model_selected == "xgboost":
            return {
                "model_type": "tree",
                "params_grid": {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                    "max_depth": trial.suggest_int("max_depth", 4, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
                    "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
                },
            }

        if model_selected == "catboost":
            return {
                "model_type": "tree",
                "params_grid": {
                    "iterations": trial.suggest_int("iterations", 200, 1000),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                    "random_strength": trial.suggest_float(
                        "random_strength", 1e-3, 10.0, log=True
                    ),
                },
            }

        return {}

    # ------------------------------------------------------------------
    # Construcción del pipeline
    # ------------------------------------------------------------------

    def build_pipeline(
        self,
        model_selected: str,
        params: dict,
        model_type: str = "tree",
    ) -> Pipeline:
        """
        Arma el Pipeline completo para un modelo dado.

        Estructura: ``deterministic → group_aggs → preprocessor → model``.
        """
        deterministic = DeterministicFeatureProcessor(
            self.mapeo_personas, self.mapeo_cuartos, self.clip_edad
        )
        group_aggs = GroupAggregatesTransformer()

        if model_selected == "lightGBM":
            extra = {"device": "cuda"} if self.use_gpu else {}
            model = LGBMRegressor(
                **params,
                random_state=self.random_seed,
                verbose=-1,
                n_jobs=self.cores_per_model,
                **extra,
            )

        elif model_selected == "xgboost":
            extra = (
                {"device": "cuda", "max_bin": 256}
                if self.use_gpu
                else {"nthread": self.cores_per_model}
            )
            model = XGBRegressor(
                **params,
                random_state=self.random_seed,
                verbosity=0,
                tree_method="hist",
                **extra,
            )

        elif model_selected == "catboost":
            extra = (
                {
                    "task_type": "GPU",
                    "devices": "0",
                    "gpu_ram_part": 0.85,
                    "bootstrap_type": "Bayesian",
                }
                if self.use_gpu
                else {"thread_count": self.cores_per_model}
            )
            model = CatBoostRegressor(
                **params,
                random_state=self.random_seed,
                verbose=0,
                **extra,
            )

        else:
            raise ValueError(
                f"Modelo '{model_selected}' no soportado. "
                f"Disponibles: {self.SUPPORTED_MODELS}."
            )

        # Clonamos el preprocessor para que cada pipeline tenga su propia instancia. Sin esto, el
        # ColumnTransformer compartido guardaría el estado del último fit y rompería predicciones posteriores.
        return Pipeline(
            [
                ("deterministic", deterministic),
                ("group_aggs", group_aggs),
                ("preprocessor", clone(self.preprocessor)),
                ("model", model),
            ]
        )

    # ------------------------------------------------------------------
    # Tuning con Optuna
    # ------------------------------------------------------------------

    def objective(self, trial: Trial, model_selected: str, random_seed: int):
        config = self.build_model_grid(trial, model_selected)
        params = config.get("params_grid", {})
        model_type = config.get("model_type")

        if not params:
            raise ValueError(f"Modelo {model_selected} sin parámetros.")
        if not model_type:
            raise ValueError(f"Modelo {model_selected} sin model_type.")

        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=random_seed)

        # GPU: corremos folds en serie (1 sola GPU compartida) y reportamos
        # el RMSE acumulado a Optuna para activar el median pruner.
        if self.use_gpu:
            fold_scores: list[float] = []
            for step, (tr_idx, va_idx) in enumerate(cv.split(self.X_train)):
                X_tr = self.slice(self.X_train, tr_idx)
                X_va = self.slice(self.X_train, va_idx)
                y_tr = self.y_train[tr_idx]
                y_va = self.y_train[va_idx]

                pipe = self.build_pipeline(model_selected, params, model_type)
                pipe.fit(X_tr, y_tr)

                rmse = root_mean_squared_error(y_va, pipe.predict(X_va))
                fold_scores.append(rmse)

                trial.report(float(np.mean(fold_scores)), step=step)
                if trial.should_prune():
                    raise TrialPruned()

            return float(np.mean(fold_scores))

        # CPU: trials paralelos vía Optuna + folds en serie. Evita sobre-suscripción.
        full_pipeline = self.build_pipeline(model_selected, params, model_type)
        scores = cross_val_score(
            full_pipeline,
            self.X_train,
            self.y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )

        return -scores.mean()

    @staticmethod
    def slice(X, idx):
        return X.iloc[idx] if hasattr(X, "iloc") else X[idx]

    def tune(self, X_train, y_train, models: list[str]) -> dict[str, Any]:
        self.X_train = X_train
        self.y_train = y_train

        for model_name in models:
            logger.info(f"Ejecutando modelo {model_name}...")
            study = create_study(
                direction="minimize",
                study_name=f"Experimento usando: {model_name}",
                sampler=TPESampler(
                    multivariate=True,
                    group=True,
                    n_startup_trials=10,
                    seed=self.random_seed,
                ),
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=0),
            )

            study.optimize(
                lambda trial, m=model_name, r=self.random_seed: self.objective(
                    trial, model_selected=m, random_seed=r
                ),
                n_trials=self.n_trials_tree,
                n_jobs=self.parallel_trials,
                show_progress_bar=True,
            )

            self.studies[model_name] = study
            logger.info(f"{model_name} | Mejor RMSE CV: {study.best_value:.4f}")
            logger.info(f"{model_name} | Mejores params: {study.best_params}")

        return self.studies

    # ------------------------------------------------------------------
    # Reentrenamiento del champion y evaluación
    # ------------------------------------------------------------------

    def fit_best(self, model_name: str, X_train, y_train) -> Pipeline:
        if model_name not in self.studies:
            raise ValueError(f"Modelo '{model_name}' no tiene estudio. " "Corre tune() primero.")

        best_params = dict(self.studies[model_name].best_params)
        logger.info(f"Reentrenando {model_name} con todo el train " f"({len(X_train):,} filas)...")

        self.best_pipeline = self.build_pipeline(
            model_selected=model_name, params=best_params, model_type="tree"
        )
        self.best_pipeline.fit(X_train, y_train)
        self.best_model_name = model_name

        logger.info(f"{model_name} entrenado y disponible en self.best_pipeline")
        return self.best_pipeline

    def evaluate(self, X_test, y_test) -> dict:
        """Evalúa el champion sobre el test holdout. Llamar una sola vez."""
        if not hasattr(self, "best_pipeline"):
            raise ValueError("No hay champion. Corre fit_best() primero.")

        predictions = self.best_pipeline.predict(X_test)
        report = {
            "model": self.best_model_name,
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
            "n_test": int(len(y_test)),
        }

        logger.info(f"Test holdout: {report}")
        return report

    def comparison_table(self) -> pd.DataFrame:
        if not self.studies:
            raise ValueError("No hay estudios. Corre tune() primero.")

        rows = [
            {
                "model": name,
                "best_cv_rmse": study.best_value,
                "n_trials": len(study.trials),
                "best_params": study.best_params,
            }
            for name, study in self.studies.items()
        ]

        return pd.DataFrame(rows).sort_values("best_cv_rmse").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Interpretabilidad con SHAP
    # ------------------------------------------------------------------

    def explain(self, X_train, sample_size: int = 50_000) -> pd.DataFrame:
        """SHAP sobre una muestra de train (nunca sobre test)."""
        logger.info('Using last version for execution...')

        if not hasattr(self, "best_pipeline"):
            raise ValueError("No hay champion. Corre fit_best() primero.")

        rng = np.random.RandomState(self.random_seed)
        n = len(X_train)
        idx = rng.choice(n, min(sample_size, n), replace=False)
        X_sample = self.slice(X_train, idx)

        det = self.best_pipeline.named_steps["deterministic"]
        gaggs = self.best_pipeline.named_steps["group_aggs"]
        preproc = self.best_pipeline.named_steps["preprocessor"]
        model = self.best_pipeline.named_steps["model"]

        X_det = det.transform(X_sample)
        X_grp = gaggs.transform(X_det)
        X_transformed = preproc.transform(X_grp)
        feature_names = preproc.get_feature_names_out()

        # Forzar todos los cores en XGBoost para acelerar SHAP.
        if isinstance(model, XGBRegressor):
            booster = model.get_booster()
            booster.set_param({"nthread": -1})
            model = booster

        logger.info(f"Calculando SHAP values sobre {len(X_sample):,} muestras...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

        self._shap_values = shap_values
        self._shap_X = X_transformed
        self._shap_feature_names = list(feature_names)

        return (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )


    def plot_shap_summary(self, max_display: int = 20, save_path: str | None = None) -> None:
        if not hasattr(self, "_shap_values"):
            raise ValueError("Corre explain() primero.")

        shap.summary_plot(
            self._shap_values,
            self._shap_X,
            feature_names=self._shap_feature_names,
            max_display=max_display,
            show=False,
        )

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=120, bbox_inches="tight")

        plt.show()

    def plot_shap_importance(self, max_display: int = 20, save_path: str | None = None) -> None:
        if not hasattr(self, "_shap_values"):
            raise ValueError("Corre explain() primero.")

        shap.summary_plot(
            self._shap_values,
            self._shap_X,
            feature_names=self._shap_feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=120, bbox_inches="tight")

        plt.show()

    def plot_shap_dependence(self, feature: str, save_path: str | None = None) -> None:
        if not hasattr(self, "_shap_values"):
            raise ValueError("Corre explain() primero.")

        if feature not in self._shap_feature_names:
            raise ValueError(f"Feature '{feature}' no está en el pipeline transformado.")

        idx = self._shap_feature_names.index(feature)
        shap.dependence_plot(
            idx,
            self._shap_values,
            self._shap_X,
            feature_names=self._shap_feature_names,
            show=False,
        )

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=120, bbox_inches="tight")

        plt.show()

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        if not hasattr(self, "best_pipeline"):
            raise ValueError("No hay champion. Corre fit_best() primero.")

        joblib.dump(self.best_pipeline, path)
        logger.info(f"Pipeline guardado en {path}")

    @staticmethod
    def load(path: str) -> Pipeline:
        pipeline = joblib.load(path)
        logger.info(f"Pipeline cargado desde {path}")
        return pipeline
