"""
Экспорт per‑SKU модели (joblib) в ONNX с опциональной INT8‑квантизацией.

Пример:
  python scripts/export_onnx.py --store 1 --family AUTOMOTIVE --quantize

Файлы:
  models/1__AUTOMOTIVE.onnx
  models/1__AUTOMOTIVE.int8.onnx  (если --quantize и доступен onnxruntime)
  models/1__AUTOMOTIVE.onnx.meta.json  (метаданные: список фич, их число)

Зависимости (опционально, устанавливаются вручную):
  pip install skl2onnx onnx onnxruntime onnxmltools lightgbm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import joblib

from train_forecast import _sanitize_family_name


def _load_optional_modules():
    try:
        import skl2onnx  # noqa: F401
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as e:
        raise SystemExit(
            "Отсутствуют зависимости для экспорта ONNX. Установите: pip install skl2onnx onnx onnxruntime\n"
            f"Ошибка: {e}"
        )
    try:
        import onnx  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "Библиотека onnx не установлена. Установите: pip install onnx\n" f"Ошибка: {e}"
        )
    # Квантизация — опционально
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore
    except Exception:
        quantize_dynamic = None  # type: ignore
        QuantType = None  # type: ignore

    # LightGBM конвертер — через onnxmltools (если установлен)
    try:
        import lightgbm as lgb  # noqa: F401
        from onnxmltools import convert_lightgbm  # type: ignore
    except Exception:
        convert_lightgbm = None  # type: ignore

    return convert_sklearn, FloatTensorType, quantize_dynamic, QuantType, convert_lightgbm


def _infer_feature_names(model) -> List[str]:
    # LightGBM sklearn API
    if hasattr(model, "feature_name_") and isinstance(model.feature_name_, list):
        return list(model.feature_name_)
    # XGBoost sklearn API
    if hasattr(model, "booster_") and hasattr(model.booster_, "feature_name"):
        try:
            return list(model.booster_.feature_name())
        except Exception:
            pass
    # Scikit models
    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)  # type: ignore[arg-type]
        except Exception:
            pass
    return []


def main() -> None:
    p = argparse.ArgumentParser("Export per-SKU model to ONNX")
    p.add_argument("--store", type=int, required=True)
    p.add_argument("--family", type=str, required=True)
    p.add_argument("--models_dir", type=str, default="models")
    p.add_argument(
        "--quantize",
        action="store_true",
        help="Сохранить также динамически квантизованную INT8-версию",
    )
    args = p.parse_args()

    convert_sklearn, FloatTensorType, quantize_dynamic, QuantType, convert_lightgbm = (
        _load_optional_modules()
    )

    safe_family = _sanitize_family_name(args.family)
    stem = f"{int(args.store)}__{safe_family}"
    model_path = Path(args.models_dir) / f"{stem}.joblib"
    if not model_path.exists():
        raise SystemExit(f"Модель не найдена: {model_path}")

    model = joblib.load(model_path)

    # Пытаемся найти сохранённый список фич рядом с моделью
    feats_path = model_path.with_suffix("").with_suffix(".features.json")
    if feats_path.exists():
        try:
            feature_names = json.loads(feats_path.read_text(encoding="utf-8"))
        except Exception:
            feature_names = _infer_feature_names(model)
    else:
        feature_names = _infer_feature_names(model)

    if not feature_names:
        raise SystemExit(
            "Не удалось определить список признаков. Сохраните *.features.json при обучении или используйте модель со списком фич."
        )

    n_features = len(feature_names)
    initial_type = [("input", FloatTensorType([None, n_features]))]
    # LightGBM требует отдельного конвертера
    onx = None
    try:
        import lightgbm as lgb  # type: ignore

        if isinstance(model, lgb.LGBMRegressor):
            if convert_lightgbm is None:
                raise SystemExit(
                    "Для экспорта LGBM установите: pip install onnxmltools lightgbm\n"
                    "Либо экспортируйте XGBoost/Sklearn модели через skl2onnx."
                )
            onx = convert_lightgbm(model.booster_, initial_types=initial_type)
    except Exception:
        pass
    if onx is None:
        onx = convert_sklearn(model, initial_types=initial_type)

    out_onnx = model_path.with_suffix(".onnx")
    out_onnx.write_bytes(onx.SerializeToString())

    # Метаданные
    meta = {
        "store_nbr": int(args.store),
        "family": args.family,
        "n_features": n_features,
        "feature_names": feature_names,
        "source_model": str(model_path.name),
    }
    out_meta = model_path.with_suffix(".onnx.meta.json")
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: экспортирован ONNX → {out_onnx}")

    if args.quantize:
        if quantize_dynamic is None:
            print(
                "onxruntime.quantization недоступен — пропускаю квантизацию. Установите onnxruntime."
            )
            return
        out_int8 = model_path.with_suffix(".int8.onnx")
        try:
            quantize_dynamic(
                model_input=str(out_onnx),
                model_output=str(out_int8),
                weight_type=QuantType.QInt8 if QuantType else None,
            )
            print(f"OK: квантизованная INT8 модель → {out_int8}")
        except Exception as e:
            print(f"Не удалось выполнить квантизацию: {e}")


if __name__ == "__main__":
    main()
