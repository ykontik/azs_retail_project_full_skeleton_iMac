"""
Сценарий быстрого демо для презентации.

Запускает минимальную последовательность команд для подготовки артефактов:
- Проверка/генерация данных (toy, если нет data_raw/*)
- Валидация → ETL → Фичи
- Экспорт EDA-графиков в docs/
- SHAP-отчёт для одной пары (по умолчанию store=1, family=AUTOMOTIVE)

Примеры:
  python scripts/demo_presentation.py --quick
  python scripts/demo_presentation.py --store 1 --family AUTOMOTIVE

Заметка: требуется установленное окружение (см. README → Установка) и зависимости
из requirements.txt. Если команды недоступны, скрипт выведет подсказки.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    try:
        return subprocess.run(cmd, cwd=ROOT).returncode
    except FileNotFoundError as e:
        print(f"Не найдена команда: {e}")
        return 127


def has_data() -> bool:
    dr = ROOT / "data_raw"
    needed = [
        dr / "train.csv",
        dr / "transactions.csv",
        dr / "oil.csv",
        dr / "holidays_events.csv",
        dr / "stores.csv",
    ]
    return all(p.exists() and p.stat().st_size > 0 for p in needed)


def main() -> None:
    p = argparse.ArgumentParser("demo_presentation")
    p.add_argument("--quick", action="store_true", help="минимальный прогон (без обучения)")
    p.add_argument("--store", type=int, default=int(os.getenv("STORE", 1)))
    p.add_argument("--family", type=str, default=os.getenv("FAMILY", "AUTOMOTIVE"))
    args = p.parse_args()

    print("== AZS+Retail: подготовка демо ==")
    # 0) Данные
    if not has_data():
        print("Данные не найдены, генерирую toy-набор...")
        rc = run([sys.executable, "scripts/generate_toy_data.py"])
        if rc != 0:
            print("Ошибка генерации toy-данных. Проверьте зависимости.")
            sys.exit(rc)

    # 1) Валидация → ETL → Фичи
    for step in ("validation.py", "etl.py", "features.py"):
        rc = run([sys.executable, step])
        if rc != 0:
            print(f"Шаг {step} завершился с кодом {rc}. Останов.")
            sys.exit(rc)

    # 2) EDA-графики → docs/
    print("Экспорт EDA-графиков...")
    rc = run([sys.executable, "scripts/export_eda_cli.py"])
    if rc != 0:
        print("Не удалось экспортировать EDA-графики. Убедитесь, что установлены matplotlib/seaborn.")

    # 3) SHAP для одной пары (требует установленный shap и наличие модели пары)
    print("SHAP отчёт для одной пары (если есть модель)...")
    shap_cmd = [
        sys.executable,
        "scripts/shap_report.py",
        "--store",
        str(args.store),
        "--family",
        str(args.family),
        "--train",
        "data_raw/train.csv",
        "--transactions",
        "data_raw/transactions.csv",
        "--oil",
        "data_raw/oil.csv",
        "--holidays",
        "data_raw/holidays_events.csv",
        "--stores",
        "data_raw/stores.csv",
    ]
    rc = run(shap_cmd)
    if rc != 0:
        print("SHAP не сформирован. Проверьте, что есть модель в models/{store}__{family}.joblib и установлен shap.")

    # 4) Куда смотреть в презентации
    docs = ROOT / "docs"
    wh = ROOT / "data_dw"
    print("\nГотово. Для слайдов используйте артефакты:")
    print("- docs/eda_seasonality_dow.png, docs/eda_seasonality_month.png, docs/eda_promo_effect.png")
    print("- data_dw/shap_summary_{STORE}__{FAMILY}.png (если создан)".format(STORE=args.store, FAMILY=str(args.family).replace(" ", "_")))
    print("- docs/model_comparison.csv (если сформирован) и data_dw/summary_metrics.txt")
    print("\nПодсказка: для установки зависимостей используется `make dev-setup` или см. README → Установка.")


if __name__ == "__main__":
    main()
