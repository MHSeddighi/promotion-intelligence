"""Reusable loader and predictor for the two-stage baseline hurdle model."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import ARTIFACT_DIR

SAMPLE_INPUT = Path(__file__).resolve().parent / "sample_model_input.csv"


class BaselinePredictor:
    """Load the trained LightGBM models once and provide validated inference."""

    def __init__(self, artifact_dir: Path = ARTIFACT_DIR) -> None:
        self.artifact_dir = Path(artifact_dir).resolve()
        self.feature_spec = self._load_json(self.artifact_dir / "feature_list.json")
        self.features = self.feature_spec.get("features")
        if not isinstance(self.features, list) or not self.features:
            raise ValueError("feature_list.json has no valid non-empty 'features' list.")
        if not all(isinstance(feature, str) for feature in self.features):
            raise ValueError("Every model feature name must be a string.")
        if len(self.features) != len(set(self.features)):
            raise ValueError("feature_list.json contains duplicate feature names.")

        self.categorical_features = [
            feature for feature in self.features if feature.endswith("_enc")
        ]
        if self.feature_spec.get("cat_cols_encoded") != self.categorical_features:
            raise ValueError(
                "cat_cols_encoded does not match the *_enc features in training order."
            )

        self.stage1 = self._load_model("model_stage1.pkl")
        self.stage2 = self._load_model("model_stage2.pkl")
        self.model_alias = self._load_model("model.pkl")
        self._validate_models()
        self.category_levels = self._extract_category_levels(self.stage1)

        stage2_levels = self._extract_category_levels(self.stage2)
        if stage2_levels != self.category_levels:
            raise ValueError("Stage-1 and stage-2 categorical metadata do not match.")

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Required model artifact was not found: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Could not read {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in {path}.")
        return value

    def _load_model(self, filename: str) -> Any:
        path = self.artifact_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required model artifact was not found: {path}")
        try:
            return joblib.load(path)
        except Exception as exc:
            raise RuntimeError(f"Could not load model {path}: {exc}") from exc

    @staticmethod
    def _feature_names(model: Any, label: str) -> list[str]:
        try:
            return list(model.booster_.feature_name())
        except Exception as exc:
            raise TypeError(f"{label} is not a fitted LightGBM model: {exc}") from exc

    def _validate_models(self) -> None:
        for label, model in (
            ("model_stage1.pkl", self.stage1),
            ("model_stage2.pkl", self.stage2),
            ("model.pkl", self.model_alias),
        ):
            model_features = self._feature_names(model, label)
            if model_features != self.features:
                missing = [item for item in self.features if item not in model_features]
                unexpected = [item for item in model_features if item not in self.features]
                raise ValueError(
                    f"Feature contract mismatch for {label}. "
                    f"Missing={missing}; unexpected={unexpected}; "
                    "the feature order must also match feature_list.json exactly."
                )
            if getattr(model, "n_features_in_", None) != len(self.features):
                raise ValueError(
                    f"{label} expects {getattr(model, 'n_features_in_', None)} features; "
                    f"feature_list.json contains {len(self.features)}."
                )

        if self.stage2.booster_.model_to_string() != self.model_alias.booster_.model_to_string():
            raise ValueError("model.pkl is not an exact alias of model_stage2.pkl.")

    def _extract_category_levels(self, model: Any) -> dict[str, list[Any]]:
        model_levels = model.booster_.pandas_categorical
        if model_levels is None or len(model_levels) != len(self.categorical_features):
            raise ValueError("Saved LightGBM categorical metadata is incomplete.")
        return {
            feature: list(levels)
            for feature, levels in zip(self.categorical_features, model_levels)
        }

    def prepare(self, input_data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(input_data, pd.DataFrame):
            raise TypeError("input_data must be a pandas DataFrame.")
        if input_data.empty:
            raise ValueError("input_data must contain at least one row.")
        if input_data.columns.has_duplicates:
            duplicates = input_data.columns[input_data.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate input columns are not allowed: {duplicates}")

        missing = [feature for feature in self.features if feature not in input_data]
        if missing:
            raise ValueError(f"Missing required model feature(s): {missing}")
        extra = [column for column in input_data if column not in self.features]
        if extra:
            warnings.warn(
                f"Ignoring extra columns; they are not sent to the model: {extra}",
                UserWarning,
                stacklevel=2,
            )

        frame = input_data.loc[:, self.features].copy()
        for column in self.features:
            original = frame[column]
            numeric = pd.to_numeric(original, errors="coerce")
            invalid = original.notna() & numeric.isna()
            if invalid.any():
                examples = original.loc[invalid].astype(str).head(5).tolist()
                raise TypeError(
                    f"Feature {column!r} must be numeric; invalid examples={examples}."
                )
            frame[column] = numeric

        for column in self.categorical_features:
            values = frame[column].dropna().to_numpy(dtype="float64")
            if np.isinf(values).any():
                raise ValueError(f"Categorical feature {column!r} contains infinity.")
            if not np.equal(values, np.floor(values)).all():
                raise ValueError(f"{column!r} must contain integer encoder codes.")
            levels = self.category_levels[column]
            unknown = pd.Index(pd.unique(frame[column].dropna())).difference(
                pd.Index(levels)
            )
            if len(unknown):
                raise ValueError(
                    f"{column!r} contains values not seen during training: "
                    f"{unknown.tolist()[:10]}"
                )
            frame[column] = pd.Categorical(frame[column], categories=levels)

        numeric_features = [
            feature
            for feature in self.features
            if feature not in self.categorical_features
        ]
        infinite = [
            feature
            for feature in numeric_features
            if np.isinf(frame[feature].to_numpy(dtype="float64")).any()
        ]
        if infinite:
            raise ValueError(f"Infinite values found in feature(s): {infinite}")

        nan_counts = frame.isna().sum()
        nan_counts = nan_counts[nan_counts > 0]
        if not nan_counts.empty:
            warnings.warn(
                f"NaN values detected: {nan_counts.to_dict()}. "
                "LightGBM will use its training-time missing-value handling.",
                UserWarning,
                stacklevel=2,
            )
        return frame

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        features = self.prepare(input_data)
        probability = np.asarray(
            self.stage1.predict_proba(features)[:, 1], dtype="float64"
        )
        positive_quantity = np.asarray(
            self.stage2.predict(features), dtype="float64"
        )
        return pd.DataFrame(
            {
                "probability_positive": probability,
                "predicted_positive_quantity": positive_quantity,
                "baseline_prediction": probability * positive_quantity,
            },
            index=input_data.index,
        )

    def load_sample(self, path: Path | None = None) -> pd.DataFrame:
        sample_path = path or SAMPLE_INPUT
        if not sample_path.is_file():
            raise FileNotFoundError(f"Sample model input was not found: {sample_path}")
        return pd.read_csv(sample_path)

    def health(self) -> dict[str, Any]:
        return {
            "stage1_model": type(self.stage1).__name__,
            "stage2_model": type(self.stage2).__name__,
            "feature_count": len(self.features),
            "categorical_features": self.categorical_features,
            "artifact_dir": str(self.artifact_dir),
        }
