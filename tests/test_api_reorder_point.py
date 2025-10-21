from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from service.app import MODELS_DIR, app


def _write_dummy_model(store: int = 1, family: str = "AUTOMOTIVE") -> Path:
    """Создаёт простую sklearn‑модель и сохраняет её в models/.

    Модель обучается на 2 фичах: a, b. Добавляем feature_names_in_, чтобы API знало порядок.
    """
    X = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [3.0, 3.0]], dtype=float)
    y = np.array([0.0, 2.0, 2.0, 4.5], dtype=float)
    mdl = LinearRegression().fit(X, y)
    mdl.feature_names_in_ = np.array(["a", "b"], dtype=object)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    safe_family = str(family).replace(" ", "_")
    path = MODELS_DIR / f"{int(store)}__{safe_family}.joblib"
    joblib.dump(mdl, path)
    return path


@pytest.mark.unit
def test_reorder_point_happy_path(monkeypatch):
    # Отключаем авторизацию для тестов
    monkeypatch.setenv("DISABLE_AUTH", "true")

    # Готовим фиктивную базовую модель
    _ = _write_dummy_model(store=1, family="AUTOMOTIVE")

    client = TestClient(app)
    body = {
        "store_nbr": 1,
        "family": "AUTOMOTIVE",
        "features": {"a": 1.0, "b": 2.0},
        "lead_time_days": 2,
        "service_level": 0.95,
    }
    r = client.post("/reorder_point", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    # Проверяем наличие ключевых полей
    for k in [
        "daily_mean",
        "sigma_daily",
        "lead_time_days",
        "service_level_z",
        "safety_stock",
        "reorder_point",
        "used_features",
        "quantiles_used",
    ]:
        assert k in data, f"missing {k} in response"
    assert data["lead_time_days"] == 2
    assert isinstance(data["reorder_point"], (float, int))
