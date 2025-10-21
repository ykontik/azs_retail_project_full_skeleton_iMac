from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from service.app import MODELS_DIR, app


def _write_dummy_model(store: int = 1, family: str = "AUTOMOTIVE") -> Path:
    """Создаёт простую sklearn‑модель с feature_names_in_ и сохраняет её в models/.

    Модель линейной регрессии обучается на фиктивных данных с двумя признаками: a, b.
    """
    X = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [3.0, 3.0]], dtype=float)
    y = np.array([0.0, 2.0, 2.0, 4.5], dtype=float)
    mdl = LinearRegression().fit(X, y)
    # Добавим имена признаков, чтобы API могло их извлечь
    mdl.feature_names_in_ = np.array(["a", "b"], dtype=object)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    safe_family = str(family).replace(" ", "_")
    path = MODELS_DIR / f"{int(store)}__{safe_family}.joblib"
    joblib.dump(mdl, path)
    return path


@pytest.mark.unit
def test_health_endpoints():
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"

    r = client.get("/live")
    assert r.status_code == 200

    r = client.get("/version")
    assert r.status_code == 200
    assert "version" in r.json()


@pytest.mark.unit
def test_predict_with_dummy_model(tmp_path, monkeypatch):
    # Отключаем авторизацию для тестов
    monkeypatch.setenv("DISABLE_AUTH", "true")

    # Готовим фиктивную модель
    _ = _write_dummy_model(store=1, family="AUTOMOTIVE")

    client = TestClient(app)
    body = {
        "store_nbr": 1,
        "family": "AUTOMOTIVE",
        "features": {"a": 1.0, "b": 2.0},
    }
    r = client.post("/predict_demand", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["store_nbr"] == 1
    assert data["family"] == "AUTOMOTIVE"
    assert "pred_qty" in data
    assert isinstance(data["pred_qty"], (float, int))
