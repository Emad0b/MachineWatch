import copy
import numpy as np
import pytest
from fastapi.testclient import TestClient
from machinewatch.data import FEATURES, features, make_examples, simulate, split_machines
from machinewatch.train import train
from machinewatch.service import create_app, load_artifact


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    path = tmp_path_factory.mktemp("training") / "test-v1"
    train(path, machines=12)
    return path


def test_machine_holdouts_and_causal_features():
    rows = simulate(10)
    x, y, rul, ids, cycles, histories = make_examples(rows)
    masks, groups = split_machines(ids)
    assert not set(groups["train"]) & set(groups["test"])
    assert not set(groups["train"]) & set(groups["validation"])
    assert not set(groups["test"]) & set(groups["validation"])
    assert sum(mask.sum() for mask in masks.values()) == len(x)
    history = next(iter(histories.values()))
    before = features(history[:10])
    changed = copy.deepcopy(history)
    changed[-1]["temperature"] = -10000
    assert np.array_equal(before, features(changed[:10]))
    assert len(before) == len(FEATURES)
    assert not np.any(y[rul > 20])


def test_invalid_history_rejected():
    history = simulate(10)[:10]
    history[1]["cycle"] = history[0]["cycle"]
    with pytest.raises(ValueError, match="increasing"):
        features(history)


def test_missing_model_returns_503(tmp_path):
    with TestClient(create_app(tmp_path / "absent")) as client:
        assert client.get("/health").json()["model_loaded"] is False
        assert client.get("/ready").status_code == 503
        assert client.get("/").status_code == 200


def test_trained_prediction_and_input_validation(trained):
    with TestClient(create_app(trained)) as client:
        assert client.get("/ready").status_code == 200
        demo = client.get("/demo").json()
        payload = {"machine_id": demo["machine_id"], "history": demo["history"][:10]}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        result = response.json()
        assert 0 <= result["risk_score"] <= 1
        assert result["estimated_rul_cycles"] >= 0
        assert result["model_version"] == "test-v1"
        assert "synthetic" in result["data_provenance"]
        trajectory = client.post("/trajectory", json={"machine_id": demo["machine_id"], "history": demo["history"][:12]}).json()
        assert len(trajectory["points"]) == 3
        assert trajectory["points"][0]["risk_score"] == pytest.approx(result["risk_score"])
        changed = copy.deepcopy(demo["history"][:12])
        changed[-1]["temperature"] += 100
        second = client.post("/trajectory", json={"machine_id": demo["machine_id"], "history": changed}).json()
        assert second["points"][0] == trajectory["points"][0]
        payload["history"] = payload["history"][:9]
        assert client.post("/predict", json=payload).status_code == 422
        payload["history"] = demo["history"][:10]
        payload["history"][1]["cycle"] = payload["history"][0]["cycle"]
        assert client.post("/predict", json=payload).status_code == 422


def test_corrupt_artifact_rejected(trained, tmp_path):
    import shutil
    copy_dir = tmp_path / "copy"
    shutil.copytree(trained, copy_dir)
    (copy_dir / "model.joblib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_artifact(copy_dir)
