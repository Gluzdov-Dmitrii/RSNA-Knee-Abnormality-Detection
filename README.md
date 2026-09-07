# RSNA Knee Abnormality Detection

**Текущий план R2, 2026-09-07:** [LOCAL_COMPUTE_PLAN.md](LOCAL_COMPUTE_PLAN.md).
Подготовка/обучение/OOF — на RTX 6000, RTX 3080 и A100 НГУ; Kaggle — финальные
inference/submissions. S01–S10 завершены, best `0.937`; folds и labels готовы,
следующий этап — storage/env/DINO cache для S11–S15. Общая GPU queue НГУ уже
активирована. `ops/compute_plan.json` дополняет ledger приоритетами local/submit.

Рабочий репозиторий для Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

Исторический снимок leaderboard: **2026-09-05 07:50 UTC** (не текущий):

- `3 089` команд; наш best `0.936`, rank `295`;
- лидер `0.954`, top-10 `0.949`, расчётная gold boundary — rank `16`, score `0.948`;
- сделано `20` сабмитов; почти все — проверки открытых inference notebooks;
- локально нет полного датасета, собственных весов, OOF и training pipeline;
- Kaggle CLI авторизован, пять приватных kernel slots и оркестраторы уже есть.

Главное ограничение code competition: обычный kernel output содержит лишь
трёхстрочный visible placeholder. При submit Kaggle выполняет notebook на скрытом test,
а полный hidden `submission.csv` участнику не возвращает. Поэтому старые hidden
prediction vectors нельзя скачать, сравнить или смешать офлайн. Любой scored ensemble
должен пересчитать все компоненты **внутри одного notebook** за лимит `< 9 h`.

Начинать здесь:

1. [`STRATEGY_WEEK_1.md`](STRATEGY_WEEK_1.md) — задача, gold path, ресурсы и S01–S35.
2. [`ops/experiment_ledger.csv`](ops/experiment_ledger.csv) — машинный статус и зависимости.
3. [`ops/LEDGER_SCHEMA.md`](ops/LEDGER_SCHEMA.md) — контракт ledger и evidence sidecar.
4. [`ops/EXPERIMENT_SPECS.md`](ops/EXPERIMENT_SPECS.md) — locked choices для S11–S25.
5. [`ops/promotion_registry.csv`](ops/promotion_registry.csv) — shared assets и promoted winners.
6. [`ops/do_not_retry.csv`](ops/do_not_retry.csv) — дубли и запрещённые тупиковые workflows.
7. [`LIGHT_MODEL_PROMPT.md`](LIGHT_MODEL_PROMPT.md) — ежедневный промпт исполнителя.
8. [`ops/baseline_submissions_2026-09-05.csv`](ops/baseline_submissions_2026-09-05.csv) — receipts 20 сабмитов, не hidden predictions.
9. [`ops/DAILY_REPORT_TEMPLATE.md`](ops/DAILY_REPORT_TEMPLATE.md) — дневной отчёт.

`tmp/` — архив скачанных public notebooks и старых оркестраторов. Это полезная
исходная база, но не собственная validation/training система.
