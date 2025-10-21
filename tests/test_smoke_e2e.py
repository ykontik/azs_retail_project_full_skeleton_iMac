from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from service.app import MODELS_DIR, app


def _ensure_dummy_model(store: int = 2, family: str = "BEVERAGES") -> Path:
    X = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=float)
    y = np.array([0.0, 2.0], dtype=float)
    mdl = LinearRegression().fit(X, y)
    mdl.feature_names_in_ = np.array(["a", "b"], dtype=object)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    p = MODELS_DIR / f"{store}__{family}.joblib"
    joblib.dump(mdl, p)
    return p


@pytest.mark.integration
def test_smoke_ready_and_models(monkeypatch):
    # Отключаем авторизацию
    monkeypatch.setenv("DISABLE_AUTH", "true")
    _ensure_dummy_model()

    client = TestClient(app)

    # readiness должен показать наличие моделей или метрик
    r = client.get("/ready")
    assert r.status_code == 200
    js = r.json()
    assert js.get("has_models") is True

    # /models должен вернуть хотя бы одну модель
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert data.get("count", 0) >= 1
