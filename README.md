# RSNA Knee Abnormality Detection

## Итог проекта (закрыт 2026-10-22)

Команда **Kneerds** (dmitriigluzdov + arsgorynich). Финальные сабмиты выбирает команда в интерфейсе Kaggle;
основной кандидат — **A**. Private LB на момент закрытия неизвестен.

| Сабмит | Public | Что это |
|---|---:|---|
| **A** `rsna-team-a-ensemble` v1 (57267685) | **0.957** | Raptor CoAtNet-384/224 (nartaa) 0.5 + R6 с flip TTA (heliosli) и S6 all-data (nartaa) 0.5; веса заданы правилом до запуска, без подбора по LB; защиты от сбоев. [experiments/TEAMA](experiments/TEAMA/README.md) |
| B `rsna-team-b-ensemble` v1 (57294542) | 0.956 | A с R6 без TTA (−2 ч) + MaViT-288 (hengck23) с весом 0.15; MaViT оказался похож на Raptor (Spearman 0.945) |
| C `rsna-team-c-goldmedal-noprobe` v1 (57359803) | 0.941 | Форк линии lavinwins/Mattia без LB-проб; 0.958 у источника был лучшим скором другой версии. [experiments/TEAMC](experiments/TEAMC/README.md) |
| 079 (arsgorynich, 57228091) | 0.957 | Копия публичного heliosli 0.9576 (R6-TTA + Raptor 384/224) |
| Наши собственные модели (CF01) | 0.886 | CoAtNet + 3D-CNN на 224-px кэше |

Главные выводы:
- Всё на 0.956–0.958 — одна линия публичных чекпойнтов (Raptor nartaa + compact ConvNeXt heliosli); разница внутри — шум public LB (~±0.001 между близкими сабмитами).
- 58 экспертных исследований не различают ансамбли (95% ДИ ±0.025) и использованы почти всеми публичными моделями для отбора; метки из отчётов расходятся с экспертными систематически (синовит, выпот) — см. [experiments/META/FINDINGS.md](experiments/META/FINDINGS.md).
- Перед сабмитом тяжёлого графа нужна репетиция на train-исследованиях через подменённый test root: видимые 3 исследования не ловят ни время, ни редкие сбои.
- При форке публичного ноутбука проверять score именно форкаемой версии, а не bestPublicScore ноутбука.

Данные соревнования, производные от них таблицы и приватный код тиммейта в репозиторий не выкладываются.

---

**Текущий план R2, 2026-09-07:** [LOCAL_COMPUTE_PLAN.md](LOCAL_COMPUTE_PLAN.md).
Подготовка/обучение/OOF — на RTX 3080 и A100 НГУ (быстрый цикл); Quadro 6000 для
DINO/24 GB; локальная RTX 2060 только smoke. Kaggle — финальные
inference/submissions. S01–S10 завершены, best `0.937`; folds и labels готовы,
следующий этап — env/DINO cache для S11–S15; **PIXEL_CACHE_V1 готов** (11.12 GiB
uint8, без скачивания ~500 GB DICOM). Общая GPU queue НГУ уже активирована.
`ops/compute_plan.json` дополняет ledger приоритетами local/submit.

Рабочий репозиторий для Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

Исторический снимок leaderboard: **2026-09-05 07:50 UTC** (не текущий):

- `3 089` команд; наш best `0.936`, rank `295`;
- лидер `0.954`, top-10 `0.949`, расчётная gold boundary — rank `16`, score `0.948`;
- сделано `20` сабмитов; почти все — проверки открытых inference notebooks;
- локально есть PIXEL_CACHE_V1 (11.12 GiB uint8 train pixels); нет DINO cache, собственных весов и OOF;
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
10. [`ops/assets/PIXEL_CACHE_V1/ORIGIN.md`](ops/assets/PIXEL_CACHE_V1/ORIGIN.md) — локальный 11.12 GiB uint8 train cache.

`tmp/` — архив скачанных public notebooks и старых оркестраторов. Это полезная
исходная база, но не собственная validation/training система.
