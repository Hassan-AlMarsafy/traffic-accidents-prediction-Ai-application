import os
import typing as t

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


APP = FastAPI(title="Accident Risk RF API", version="1.0.0")


class PredictRequest(BaseModel):
	features: dict = Field(..., description="Feature name to value mapping for one sample")


class PredictBatchRequest(BaseModel):
	rows: list[dict] = Field(..., description="List of feature dicts")


_MODEL = None
_MODEL_FEATURES: list[str] | None = None


def _resolve_model_path() -> str | None:
	default_candidates = [
		"traffic-accidents-prediction-Ai-application/Main/saved_models/rf_model_z_20250922_193516.pkl",
		"traffic-accidents-prediction-Ai-application/Main/saved_models/rf_model_20250922_193516.pkl",
	]
	env_path = os.getenv("RF_MODEL_PATH")
	if env_path and os.path.exists(env_path):
		return env_path
	for p in default_candidates:
		full = os.path.join(os.getcwd(), p)
		if os.path.exists(full):
			return full
	return None


def _load_model_lazy():
	global _MODEL, _MODEL_FEATURES
	if _MODEL is not None:
		return _MODEL
	model_path = _resolve_model_path()
	if model_path is None:
		raise FileNotFoundError("RF model file not found. Set RF_MODEL_PATH or place it in saved_models.")
	# Use joblib for large pickles
	_MODEL = joblib.load(model_path)
	_MODEL_FEATURES = None
	if hasattr(_MODEL, "feature_names_in_"):
		_MODEL_FEATURES = [str(x) for x in _MODEL.feature_names_in_]
	return _MODEL


def _df_from_features_dicts(rows: list[dict]) -> pd.DataFrame:
	df = pd.DataFrame(rows)
	# Ensure numeric where possible; keep non-numeric as-is (model may handle encoding internally if persisted)
	for col in df.columns:
		try:
			df[col] = pd.to_numeric(df[col])
		except Exception:
			pass
	return df


@APP.get("/health")
def health() -> dict:
	path = _resolve_model_path()
	return {"status": "ok", "model_path": path, "loaded": _MODEL is not None}


@APP.get("/model-info")
def model_info() -> dict:
	try:
		model = _load_model_lazy()
	except FileNotFoundError as e:
		raise HTTPException(status_code=500, detail=str(e))
	info: dict[str, t.Any] = {"type": type(model).__name__}
	if hasattr(model, "n_estimators"):
		info["n_estimators"] = getattr(model, "n_estimators")
	if hasattr(model, "classes_"):
		info["classes"] = [str(c) for c in getattr(model, "classes_")]
	if hasattr(model, "feature_names_in_"):
		info["feature_names_in"] = [str(x) for x in getattr(model, "feature_names_in_")]
	return info


@APP.post("/predict")
def predict_one(req: PredictRequest) -> dict:
	try:
		model = _load_model_lazy()
	except FileNotFoundError as e:
		raise HTTPException(status_code=500, detail=str(e))
	df = _df_from_features_dicts([req.features])
	# Align columns if model has feature_names_in_
	if _MODEL_FEATURES is not None:
		missing = [c for c in _MODEL_FEATURES if c not in df.columns]
		for m in missing:
			df[m] = np.nan
		df = df[_MODEL_FEATURES]
	try:
		pred = model.predict(df)[0]
		res: dict[str, t.Any] = {"prediction": str(pred)}
		if hasattr(model, "predict_proba"):
			proba = model.predict_proba(df)[0]
			res["probabilities"] = [float(x) for x in proba]
		return res
	except Exception as e:
		raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


@APP.post("/predict-batch")
def predict_batch(req: PredictBatchRequest) -> dict:
	try:
		model = _load_model_lazy()
	except FileNotFoundError as e:
		raise HTTPException(status_code=500, detail=str(e))
	df = _df_from_features_dicts(req.rows)
	if _MODEL_FEATURES is not None:
		missing = [c for c in _MODEL_FEATURES if c not in df.columns]
		for m in missing:
			df[m] = np.nan
		df = df[_MODEL_FEATURES]
	try:
		preds = model.predict(df)
		resp: dict[str, t.Any] = {"predictions": [str(p) for p in preds]}
		if hasattr(model, "predict_proba"):
			probas = model.predict_proba(df)
			resp["probabilities"] = [[float(x) for x in row] for row in probas]
		return resp
	except Exception as e:
		raise HTTPException(status_code=400, detail=f"Batch prediction failed: {e}")


# To run locally: uvicorn backend.main:APP --host 0.0.0.0 --port 8000
