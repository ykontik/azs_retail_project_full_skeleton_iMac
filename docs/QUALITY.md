# Качество: тесты, типы, стиль

## Тесты
- Все тесты: `make test`
- С покрытием: `make test-cov`
- Быстрые: `make test-fast`
- Unit/Integration: `make test-unit` / `make test-integration`

Порог покрытия задаётся в `pytest.ini` (`--cov-fail-under`).

## Линт и формат
- Линтинг: `make lint`
- Авто-фикс: `make lint-fix`
- Типы: `make typecheck`
- Полный пакет: `make quality`

