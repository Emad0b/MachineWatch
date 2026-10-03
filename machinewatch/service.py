"""Prediction service and local dashboard; trusted local artifacts only."""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import numpy as np
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from threadpoolctl import threadpool_limits
from .data import FEATURES, WINDOW, digest, features, validate_history

ROOT = Path(__file__).resolve().parents[1]


class Reading(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    cycle: int = Field(ge=1, strict=True)
    temperature: float
    vibration: float
    pressure: float


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    machine_id: str = Field(min_length=1, max_length=100)
    history: list[Reading] = Field(min_length=WINDOW, max_length=1000)


def load_artifact(directory):
    directory = Path(directory)
    info = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if info.get("schema_version") != 1 or info.get("features") != FEATURES or info.get("window") != WINDOW:
        raise ValueError("Incompatible model feature contract")
    if info.get("sklearn") != sklearn.__version__:
        raise ValueError("Use the artifact's recorded scikit-learn version")
    if digest(directory / "model.joblib") != info["model_sha256"]:
        raise ValueError("Artifact checksum mismatch")
    # joblib uses pickle. Checksums detect damage, not malicious artifacts.
    return joblib.load(directory / "model.joblib"), info


def create_app(artifact_dir=None):
    directory = Path(artifact_dir or os.environ.get("MACHINEWATCH_ARTIFACT", ROOT / "artifacts" / "demo-v1"))

    @asynccontextmanager
    async def lifespan(app):
        # Missing artifacts allow the dashboard to explain setup; corrupt ones fail startup.
        app.state.models, app.state.info = (None, None)
        if directory.exists():
            app.state.models, app.state.info = load_artifact(directory)
        with threadpool_limits(limits=2):
            yield

    app = FastAPI(title="MachineWatch", version="1.0.0", lifespan=lifespan)

    @app.get("/")
    def dashboard():
        return FileResponse(ROOT / "web" / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": app.state.models is not None}

    @app.get("/ready")
    def ready():
        if app.state.models is None:
            raise HTTPException(503, "Train a model first: python -m machinewatch.train")
        return {"status": "ready", "metadata": app.state.info}

    @app.get("/demo")
    def demo():
        ready()
        path = directory / "demo.json"
        if not path.exists():
            raise HTTPException(404, "This release has no bundled demo")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/evaluation")
    def evaluation():
        ready()
        return json.loads((directory / "metrics.json").read_text(encoding="utf-8"))

    def predict(request):
        rows = [r.model_dump() for r in request.history]
        validate_history(rows)
        x = features(rows).reshape(1, -1)
        started = time.perf_counter()
        models, info = app.state.models, app.state.info
        score = float(models["classifier"].predict_proba(x)[0, 1])
        rul = float(max(0, models["regressor"].predict(x)[0]))
        if not np.isfinite([score, rul]).all():
            raise RuntimeError("Non-finite model output")
        return {"machine_id": request.machine_id, "cycle": rows[-1]["cycle"],
                "risk_score": score, "estimated_rul_cycles": rul,
                "alert": score >= info["threshold"], "threshold": info["threshold"],
                "horizon_cycles": info["horizon_cycles"], "model_version": info["model_version"],
                "data_provenance": info["data_provenance"],
                "latency_ms": (time.perf_counter() - started) * 1000,
                "interpretation": "Model score for failure within the horizon; not a calibrated probability or a maintenance instruction."}

    @app.post("/predict")
    async def prediction(request: PredictionRequest):
        ready()
        try:
            return await run_in_threadpool(predict, request)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    def trajectory(request):
        rows = [r.model_dump() for r in request.history]
        validate_history(rows)
        x = np.array([features(rows[i - WINDOW:i]) for i in range(WINDOW, len(rows) + 1)])
        models, info = app.state.models, app.state.info
        scores = models["classifier"].predict_proba(x)[:, 1]
        rul = np.maximum(models["regressor"].predict(x), 0)
        return {"machine_id": request.machine_id, "model_version": info["model_version"],
                "data_provenance": info["data_provenance"], "horizon_cycles": info["horizon_cycles"], "threshold": info["threshold"],
                "points": [{"cycle": row["cycle"], "risk_score": float(score), "estimated_rul_cycles": float(remaining)}
                           for row, score, remaining in zip(rows[WINDOW - 1:], scores, rul)]}

    @app.post("/trajectory")
    async def history_prediction(request: PredictionRequest):
        ready()
        try:
            return await run_in_threadpool(trajectory, request)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()
