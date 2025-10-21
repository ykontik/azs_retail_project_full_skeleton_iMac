"""
Утилита для оценки базовой производительности моделей.

Скрипт загружает все *.joblib файлы из каталога моделей,
измеряет время их десериализации и (по возможности) время
вызова predict на синтетическом входе. Результаты можно
сохранить в CSV/JSON и построить простой график.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd


@dataclass
class ModelBenchmark:
    model_name: str
    model_type: str
    file_size_kb: float
    load_time_ms: float
    predict_time_ms: Optional[float]
    notes: str = ""


def _measure_model(path: Path, pred_rows: int = 64) -> ModelBenchmark:
    model_name = path.name
    start = time.perf_counter()
    try:
        model = joblib.load(path)
    except Exception as exc:  # pragma: no cover - крайне маловероятно
        load_ms = (time.perf_counter() - start) * 1000.0
        return ModelBenchmark(
            model_name=model_name,
            model_type="load_error",
            file_size_kb=path.stat().st_size / 1024.0,
            load_time_ms=load_ms,
            predict_time_ms=None,
            notes=f"load_failed: {exc}",
        )

    load_ms = (time.perf_counter() - start) * 1000.0
    model_type = type(model).__name__

    predict_ms: Optional[float] = None
    note = ""

    n_features = getattr(model, "n_features_in_", None)
    feature_names = getattr(model, "feature_names_in_", None)

    if n_features is None and isinstance(feature_names, (list, tuple, np.ndarray)):
        n_features = len(feature_names)

    if n_features and hasattr(model, "predict"):
        try:
            if isinstance(feature_names, (list, tuple, np.ndarray)):
                columns = [str(c) for c in feature_names]
            else:
                columns = [f"f{i}" for i in range(int(n_features))]

            X = pd.DataFrame(np.zeros((pred_rows, len(columns)), dtype=float), columns=columns)
            pred_start = time.perf_counter()
            _ = model.predict(X)
            predict_ms = (time.perf_counter() - pred_start) * 1000.0
        except Exception as exc:  # pragma: no cover - зависит от модели
            note = f"predict_failed: {exc}"

    return ModelBenchmark(
        model_name=model_name,
        model_type=model_type,
        file_size_kb=path.stat().st_size / 1024.0,
        load_time_ms=load_ms,
        predict_time_ms=predict_ms,
        notes=note,
    )


def _write_reports(results: List[ModelBenchmark], output_dir: Path, as_json: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(m) for m in results])

    csv_path = output_dir / "benchmark_report.csv"
    df.to_csv(csv_path, index=False)

    if as_json:
        json_path = output_dir / "benchmark_report.json"
        json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8")


def _write_plot(results: List[ModelBenchmark], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib уже в зависимостях проекта
        print(f"[benchmark] Не удалось построить график: {exc}", file=sys.stderr)
        return

    df = pd.DataFrame([asdict(m) for m in results])
    df = df.sort_values("load_time_ms", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4)))
    ax.barh(df["model_name"], df["load_time_ms"], color="#4c72b0")
    ax.set_xlabel("Время загрузки, мс")
    ax.set_ylabel("Модель")
    ax.set_title("Время загрузки моделей (joblib)")
    ax.invert_yaxis()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "benchmark_load_time.png", dpi=200)
    plt.close(fig)


def parse_args(raw_args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark для моделей (.joblib).")
    parser.add_argument("--models-dir", default="models", help="Каталог с joblib моделями.")
    parser.add_argument("--output-dir", default="data_dw", help="Куда сохранить отчёт/график.")
    parser.add_argument("--generate-report", action="store_true", help="Сохранить CSV/JSON отчёт.")
    parser.add_argument("--generate-plot", action="store_true", help="Сохранить PNG график.")
    parser.add_argument("--json", action="store_true", help="Дополнительно сохранить JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число моделей.")
    parser.add_argument(
        "--test-size",
        type=int,
        default=64,
        help="Размер синтетического батча для predict (по умолчанию 64).",
    )
    return parser.parse_args(raw_args)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"[benchmark] Каталог моделей не найден: {models_dir}", file=sys.stderr)
        return 0

    model_paths = sorted(models_dir.glob("*.joblib"))
    if args.limit:
        model_paths = model_paths[: args.limit]

    if not model_paths:
        print("[benchmark] Модели (.joblib) не найдены.")
        return 0

    results = [_measure_model(path, pred_rows=args.test_size) for path in model_paths]

    print(
        "Модель".ljust(35),
        "Тип".ljust(18),
        "Размер (KB)".rjust(12),
        "Load (ms)".rjust(10),
        "Predict (ms)".rjust(12),
    )
    for r in results:
        predict_disp = f"{r.predict_time_ms:.2f}" if r.predict_time_ms is not None else "-"
        print(
            r.model_name.ljust(35),
            r.model_type.ljust(18),
            f"{r.file_size_kb:10.1f}",
            f"{r.load_time_ms:10.2f}",
            f"{predict_disp:>12}",
            sep=" ",
        )
        if r.notes:
            print(f"  ↳ {r.notes}")

    output_dir = Path(args.output_dir)
    if args.generate_report:
        _write_reports(results, output_dir, as_json=args.json)
        print(f"[benchmark] Отчёт сохранён в {output_dir}")
    if args.generate_plot:
        _write_plot(results, output_dir)
        print(f"[benchmark] График сохранён в {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
