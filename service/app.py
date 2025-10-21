import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Метрики Prometheus
if TYPE_CHECKING:  # только для mypy
    pass

prom_module: Optional[Any]
try:
    import prometheus_client as prom_module
except Exception:  # pragma: no cover - зависимость опциональна
    prom_module = None

# Загружаем .env для конфигурации путей и настроек
load_dotenv()

app = FastAPI(
    title="AZS+Retail — Forecast API",
    version="1.0.0",
    description=(
        "Сервис прогнозирования спроса per‑SKU. Эндпоинты для метрик, моделей и инференса.\n\n"
        "См. теги: health, models, inference."
    ),
    terms_of_service="https://example.com/tos",
    contact={"name": "AZS+Retail Team", "email": "ml@example.com"},
    license_info={"name": "Proprietary"},
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_parameters={
        "displayRequestDuration": True,
        "persistAuthorization": True,
        "defaultModelsExpandDepth": -1,
        "faviconUrl": "/static/favicon.svg",
    },
)

app.openapi_tags = [
    {"name": "health", "description": "Сервисные эндпоинты"},
    {"name": "models", "description": "Информация о моделях и метриках"},
    {"name": "inference", "description": "Онлайн‑прогнозы и квантили"},
]

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
WAREHOUSE_DIR = Path(os.getenv("WAREHOUSE_DIR", "data_dw"))
MODELS_DIR.mkdir(exist_ok=True, parents=True)
WAREHOUSE_DIR.mkdir(exist_ok=True, parents=True)

# Статика для Swagger favicon/логотипов
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# CORS для UI (Streamlit)
_cors_env = os.getenv("CORS_ORIGINS", "").strip()
if _cors_env:
    origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
else:
    origins = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://ui:8501",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Простая авторизация по ключу (можно отключить через DISABLE_AUTH=true)

# Схема безопасности для Swagger (добавляет поле X-API-Key в UI)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False, description="Секретный ключ API")


def _check_api_key(api_key: str | None = Security(api_key_scheme)):
    # Читаем флаги из окружения на каждый вызов, чтобы тесты могли подменять os.environ
    disable_auth = os.getenv("DISABLE_AUTH", "true").lower() in {"1", "true", "yes", "y"}
    if disable_auth:
        return
    api_key_cfg = os.getenv("API_KEY", "")
    if not api_key_cfg or not api_key or api_key != api_key_cfg:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")


# Простейшие метрики Prometheus
REQ_COUNT: Optional[Any]
REQ_LATENCY: Optional[Any]
if prom_module is not None:
    REQ_COUNT = prom_module.Counter(
        "api_requests_total",
        "Количество запросов",
        ["method", "path", "status"],
    )
    REQ_LATENCY = prom_module.Histogram(
        "api_request_latency_seconds",
        "Задержка запросов",
        buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
        labelnames=("method", "path"),
    )
else:
    REQ_COUNT = None
    REQ_LATENCY = None


@app.middleware("http")
async def _metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    # Метрики можно отключить, если отсутствует зависимость
    path = request.url.path
    method = request.method
    if REQ_LATENCY is None:
        response = await call_next(request)
        return response
    assert REQ_LATENCY is not None  # для mypy
    with REQ_LATENCY.labels(method=method, path=path).time():
        response = await call_next(request)
    try:
        assert REQ_COUNT is not None
        REQ_COUNT.labels(method=method, path=path, status=str(response.status_code)).inc()
    except Exception:
        pass
    return response


class PredictRequest(BaseModel):
    store_nbr: int
    family: str
    features: Dict[str, float] = Field(default_factory=dict)
    date: Optional[str] = None
    model_config = {
        "json_schema_extra": {
            "example": {
                "store_nbr": 1,
                "family": "AUTOMOTIVE",
                "features": {"year": 2017, "month": 8, "sales_lag_7": 5.0},
                "date": "2017-08-15",
            }
        }
    }


class PredictResponse(BaseModel):
    store_nbr: int
    family: str
    pred_qty: float
    used_features: List[str]
    model_config = {
        "json_schema_extra": {
            "example": {
                "store_nbr": 1,
                "family": "AUTOMOTIVE",
                "pred_qty": 6.4,
                "used_features": ["year", "month", "sales_lag_7"],
            }
        }
    }


class PredictQuantilesResponse(BaseModel):
    store_nbr: int
    family: str
    quantiles: Dict[str, float]
    used_features: List[str]
    model_config = {
        "json_schema_extra": {
            "example": {
                "store_nbr": 1,
                "family": "AUTOMOTIVE",
                "quantiles": {"0.5": 6.0, "0.9": 9.0},
                "used_features": ["year", "month", "sales_lag_7"],
            }
        }
    }


def _model_path(store_nbr: int, family: str) -> Path:
    safe_family = str(family).replace(" ", "_")
    return MODELS_DIR / f"{store_nbr}__{safe_family}.joblib"


def _model_key(store_nbr: int, family: str) -> str:
    safe_family = str(family).replace(" ", "_")
    return f"{store_nbr}__{safe_family}"


def _quantile_model_path(store_nbr: int, family: str, q: float) -> Path:
    safe_family = str(family).replace(" ", "_")
    qname = int(round(q * 100))
    return MODELS_DIR / f"{store_nbr}__{safe_family}__q{qname}.joblib"


# Кеш моделей в памяти (LRU), чтобы не читать файлы при каждом запросе
@lru_cache(maxsize=512)
def _load_model_cached(key: str):
    path = MODELS_DIR / f"{key}.joblib"
    return joblib.load(path)


def _load_model(store_nbr: int, family: str):
    path = _model_path(store_nbr, family)
    if not path.exists():
        return None
    try:
        return _load_model_cached(_model_key(store_nbr, family))
    except Exception:
        return None


def _get_feature_names(model) -> List[str]:
    names = None
    if hasattr(model, "feature_name_") and isinstance(model.feature_name_, list):
        names = model.feature_name_
    elif hasattr(model, "feature_names_in_"):
        try:
            names = list(model.feature_names_in_)
        except Exception:
            names = None
    elif hasattr(model, "booster_") and hasattr(model.booster_, "feature_name"):
        names = list(model.booster_.feature_name())
    return names or []


def _coerce_feature_value(v) -> float:
    """Приведение входного признака к float, чтобы не падать на строках.

    Комментарии — на русском. Имена функций/переменных — на английском по стандарту Python.
    Правила:
    - None/NaN/пустые → 0.0
    - bool → 0.0/1.0
    - int/float → float(v)
    - строки → попытка float, иначе 0.0 (для категориальных колонок без кодировки)
    """
    try:
        if v is None:
            return 0.0
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
        # строки и прочие типы
        return float(str(v))
    except Exception:
        return 0.0


@app.get(
    "/health",
    tags=["health"],
    summary="Статус сервиса",
    description="Проверка доступности API",
)
def health():
    """Проверка доступности сервиса."""
    return {"status": "ok"}


@app.get(
    "/live",
    tags=["health"],
    summary="Liveness probe",
    description="Проверка живости приложения",
)
def liveness():
    return {"status": "alive"}


@app.get(
    "/ready",
    tags=["health"],
    summary="Readiness probe",
    description="Готовность к обслуживанию (наличие данных/моделей)",
)
def readiness():
    metrics_ok = (WAREHOUSE_DIR / "metrics_per_sku.csv").exists()
    any_model = any(MODELS_DIR.glob("*.joblib"))
    ready = metrics_ok or any_model
    return {"ready": ready, "metrics_csv": metrics_ok, "has_models": any_model}


@app.get(
    "/version",
    tags=["health"],
    summary="Версия и окружение",
    description="Версия приложения и основные пути",
)
def version():
    return {
        "version": app.version,
        "models_dir": str(MODELS_DIR),
        "warehouse_dir": str(WAREHOUSE_DIR),
        "disable_auth": os.getenv("DISABLE_AUTH", "true").lower() in {"1", "true", "yes", "y"},
        "git_sha": os.getenv("GIT_SHA", "unknown"),
    }


@app.get(
    "/models",
    tags=["models"],
    summary="Список моделей",
    description="Возвращает список файлов моделей в каталоге MODELS_DIR",
)
def list_models():
    """Список моделей в каталоге MODELS_DIR (store_nbr, family, путь)."""
    files = []
    for p in MODELS_DIR.glob("*.joblib"):
        try:
            store, fam = p.stem.split("__", 1)
            files.append({"store_nbr": int(store), "family": fam.replace("_", " "), "path": str(p)})
        except ValueError:
            files.append({"store_nbr": None, "family": p.stem, "path": str(p)})
    return {"count": len(files), "models": files}


@app.get(
    "/quantiles_available",
    tags=["models"],
    summary="Доступные квантили для пары",
    description="Список доступных квантильных моделей для пары (store_nbr, family)",
)
def quantiles_available(store_nbr: int, family: str, _: None = Depends(_check_api_key)):
    """Список доступных квантильных моделей для пары (store_nbr, family)."""
    safe_family = str(family).replace(" ", "_")
    prefix = f"{store_nbr}__{safe_family}__q"
    qs: List[float] = []
    for p in MODELS_DIR.glob(f"{prefix}*.joblib"):
        try:
            stem = p.stem
            qstr = stem.split("__q")[-1]
            q = float(int(qstr) / 100.0)
            qs.append(q)
        except Exception:
            continue
    qs = sorted(set(qs))
    return {"store_nbr": store_nbr, "family": family, "quantiles": qs}


@app.get(
    "/feature_names",
    tags=["models"],
    summary="Имена признаков модели",
    description="Возвращает имена признаков обученной модели для пары (store_nbr, family)",
    responses={404: {"description": "Model not found"}},
)
def feature_names(store_nbr: int, family: str, _: None = Depends(_check_api_key)):
    """Возвращает имена признаков, на которых обучена модель для пары (store_nbr, family)."""
    model = _load_model(store_nbr, family)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    names = _get_feature_names(model)
    return {"store_nbr": store_nbr, "family": family, "feature_names": names}


@app.get(
    "/metrics",
    tags=["models"],
    summary="Метрики обучения по SKU",
    description="Возвращает таблицу метрик обучения per-SKU из warehouse (metrics_per_sku.csv)",
    responses={404: {"description": "metrics_per_sku.csv not found"}},
)
def get_metrics(_: None = Depends(_check_api_key)):
    """Возвращает таблицу метрик обучения per-SKU из файла warehouse (metrics_per_sku.csv)."""
    path = WAREHOUSE_DIR / "metrics_per_sku.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="metrics_per_sku.csv not found")
    df = pd.read_csv(path)
    # Заменим NaN/Inf на None, чтобы JSON кодировался без ошибок
    df = df.replace({np.nan: None})
    return {"columns": df.columns.tolist(), "rows": df.to_dict(orient="records")}


@app.get(
    "/stock_plan",
    tags=["models"],
    summary="План запасов (бейслайн)",
    description=(
        "Возвращает таблицу плана запасов (daily_mean, sigma, Safety Stock, ROP)\n"
        "Источник: data_dw/stock_plan.csv (формируется скриптом train_stock.py или make stock)"
    ),
    responses={404: {"description": "stock_plan.csv not found"}},
)
def get_stock_plan(_: None = Depends(_check_api_key)):
    path = WAREHOUSE_DIR / "stock_plan.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="stock_plan.csv not found")
    df = pd.read_csv(path)
    return {"columns": df.columns.tolist(), "rows": df.to_dict(orient="records")}


@app.post(
    "/predict_demand",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Прогноз спроса (одна пара)",
    description="Прогноз спроса по одной паре (store_nbr, family) на основе переданных фич",
    responses={
        400: {"description": "Некорректный формат входных данных"},
        503: {"description": "Модель отсутствует для пары"},
        500: {"description": "Ошибка предсказания"},
    },
)
def predict(req: PredictRequest, _: None = Depends(_check_api_key)):
    """Прогноз спроса по одной паре (store_nbr, family) на основе переданных фич (features)."""
    model = _load_model(req.store_nbr, req.family)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not found for this (store_nbr, family). Train or adjust TOP_N_SKU.",
        )

    feat_names = _get_feature_names(model)
    if not feat_names:
        raise HTTPException(
            status_code=500, detail="Model has no feature names; retrain with named columns."
        )

    x = []
    for name in feat_names:
        x.append(_coerce_feature_value(req.features.get(name, 0.0)))
    import numpy as np

    X = np.array(x, dtype=float).reshape(1, -1)

    try:
        pred = float(model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictResponse(
        store_nbr=req.store_nbr, family=req.family, pred_qty=pred, used_features=feat_names
    )


@app.post(
    "/predict_bulk",
    tags=["inference"],
    summary="Массовый прогноз",
    description="Принимает список запросов PredictRequest и возвращает результат по каждому",
)
def predict_bulk(requests_list: List[PredictRequest], _: None = Depends(_check_api_key)):
    """Массовое прогнозирование: принимает список запросов PredictRequest и возвращает результат по каждому из них."""
    responses = []
    for req in requests_list:
        model = _load_model(req.store_nbr, req.family)
        if model is None:
            responses.append(
                {"store_nbr": req.store_nbr, "family": req.family, "error": "Model not found"}
            )
            continue
        feat_names = _get_feature_names(model)
        x = [_coerce_feature_value(req.features.get(name, 0.0)) for name in feat_names]
        import numpy as np

        X = np.array(x, dtype=float).reshape(1, -1)
        try:
            pred = float(model.predict(X)[0])
            responses.append(
                {
                    "store_nbr": req.store_nbr,
                    "family": req.family,
                    "pred_qty": pred,
                    "used_features": feat_names,
                }
            )
        except Exception as e:
            responses.append(
                {
                    "store_nbr": req.store_nbr,
                    "family": req.family,
                    "error": f"Prediction failed: {e}",
                }
            )
    return {"results": responses}


@app.post(
    "/predict_demand_quantiles",
    response_model=PredictQuantilesResponse,
    tags=["inference"],
    summary="Прогноз по квантилям",
    description="Возвращает прогноз по запрошенным квантилям (например, 0.5 и 0.9)",
    responses={
        400: {"description": "Некорректный параметр qs"},
        404: {"description": "Базовая или квантильные модели не найдены"},
    },
)
def predict_quantiles(
    req: PredictRequest, qs: Optional[str] = None, _: None = Depends(_check_api_key)
):
    """Прогноз по квантилям (например, 0.5 и 0.9). Квантили передаются через параметр `qs`, разделённый запятыми.

    Пример: qs=0.5,0.9
    Если модели квантилей отсутствуют, вернётся ошибка 404 по соответствующему квантилю.
    """
    # Парсим список квантилей
    if qs:
        try:
            quants = [float(x.strip()) for x in qs.split(",") if x.strip()]
        except Exception:
            raise HTTPException(status_code=400, detail="Некорректный формат параметра qs")
    else:
        quants = [0.5, 0.9]

    # Определяем порядок фич
    base_model = _load_model(req.store_nbr, req.family)
    if base_model is None:
        raise HTTPException(
            status_code=404, detail="Базовая модель не найдена. Сначала обучите модель."
        )
    feat_names = _get_feature_names(base_model)
    x = [_coerce_feature_value(req.features.get(name, 0.0)) for name in feat_names]
    import numpy as np

    X = np.array(x, dtype=float).reshape(1, -1)

    results: Dict[str, float] = {}
    for q in quants:
        path = _quantile_model_path(req.store_nbr, req.family, q)
        if not path.exists():
            continue
        try:
            mdl_q = joblib.load(path)
            pred_q = float(mdl_q.predict(X)[0])
            results[str(q)] = pred_q
        except Exception:
            continue

    if not results:
        raise HTTPException(status_code=404, detail="Модели для указанных квантилей не найдены.")
    return PredictQuantilesResponse(
        store_nbr=req.store_nbr, family=req.family, quantiles=results, used_features=feat_names
    )


# ---------------------- БИЗНЕС-МЕТРИКИ: ROP/SS ----------------------
class ReorderPointRequest(BaseModel):
    store_nbr: int
    family: str
    features: Dict[str, float] = Field(default_factory=dict)
    lead_time_days: int = 2
    service_level_z: Optional[float] = None
    service_level: Optional[float] = Field(
        default=None, description="Уровень сервиса в [0,1], напр. 0.95"
    )


class ReorderPointResponse(BaseModel):
    store_nbr: int
    family: str
    daily_mean: float
    sigma_daily: float
    lead_time_days: int
    service_level_z: float
    safety_stock: float
    reorder_point: float
    used_features: List[str]
    quantiles_used: bool


def _z_from_service_level(p: Optional[float]) -> Optional[float]:
    """Приближение Z по уровню сервиса (без SciPy)."""
    if p is None:
        return None
    table = {
        0.80: 0.8416,
        0.90: 1.2816,
        0.95: 1.6449,
        0.975: 1.9600,
        0.99: 2.3263,
    }
    closest = min(table.keys(), key=lambda x: abs(x - p))
    return table[closest]


@app.post(
    "/reorder_point",
    tags=["inference"],
    summary="Расчёт ROP/SS на основе прогноза",
    description=(
        "Возвращает ежедневное среднее, сигму, safety stock и reorder point. "
        "Использует квантильные модели (P50/P90), если доступны."
    ),
)
def reorder_point(req: ReorderPointRequest, _: None = Depends(_check_api_key)):
    base_model = _load_model(req.store_nbr, req.family)
    if base_model is None:
        raise HTTPException(status_code=404, detail="Базовая модель не найдена.")

    feat_names = _get_feature_names(base_model)
    if not feat_names:
        raise HTTPException(
            status_code=500, detail="Model has no feature names; retrain with named columns."
        )

    import numpy as np

    X = np.array(
        [_coerce_feature_value(req.features.get(n, 0.0)) for n in feat_names], dtype=float
    ).reshape(1, -1)

    try:
        daily_mean = float(base_model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Попытка оценить сигму через квантили (если доступны модели P50/P90)
    quantiles_used = False
    sigma_daily = None
    p50_path = _quantile_model_path(req.store_nbr, req.family, 0.5)
    p90_path = _quantile_model_path(req.store_nbr, req.family, 0.9)
    if p50_path.exists() and p90_path.exists():
        try:
            mdl_p50 = joblib.load(p50_path)
            mdl_p90 = joblib.load(p90_path)
            q50 = float(mdl_p50.predict(X)[0])
            q90 = float(mdl_p90.predict(X)[0])
            # Нормальное приближение: P90 ~ mu + 1.2816*sigma; P50 ~ median ~ mu
            sigma_daily = max((q90 - q50) / 1.2816, 0.0)
            daily_mean = q50  # медианный прогноз как базовый спрос
            quantiles_used = True
        except Exception:
            sigma_daily = None

    # Фоллбек на простую эвристику, если квантилей нет
    if sigma_daily is None:
        sigma_daily = max(0.25 * daily_mean, 0.0)

    z = (
        req.service_level_z
        if (req.service_level_z is not None)
        else _z_from_service_level(req.service_level or 0.95)
    )
    if z is None:
        z = 1.6449  # 95%

    L = max(int(req.lead_time_days), 1)
    safety_stock = z * sigma_daily * (L**0.5)
    rop = daily_mean * L + safety_stock

    return ReorderPointResponse(
        store_nbr=req.store_nbr,
        family=req.family,
        daily_mean=daily_mean,
        sigma_daily=sigma_daily,
        lead_time_days=L,
        service_level_z=z,
        safety_stock=safety_stock,
        reorder_point=rop,
        used_features=feat_names,
        quantiles_used=quantiles_used,
    )


@app.get(
    "/metrics-prom",
    tags=["health"],
    summary="Prometheus метрики",
    description="Метрики приложения в формате Prometheus (text/plain)",
)
def metrics_prom(_: None = Depends(_check_api_key)):
    if prom_module is None:
        raise HTTPException(status_code=503, detail="prometheus_client не установлен")
    data = prom_module.generate_latest()
    return Response(content=data, media_type=prom_module.CONTENT_TYPE_LATEST)
