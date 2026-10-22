# SPRINT943 — точная V34 с документацией и проверками

Основной артефакт: [`kaggle/rsna-knee-v34-documented-strict.ipynb`](kaggle/rsna-knee-v34-documented-strict.ipynb).
Приватный запуск: [RSNA Knee V34 Documented Strict](https://www.kaggle.com/code/dmitriigluzdov/rsna-knee-v34-documented-strict),
**V1**, начат 22 сентября в 12:45:40 UTC после освобождения GPU-слота.
**V1 COMPLETE:** 346.995 секунды на T4×2; полный граф подтверждён, CSV побайтно
совпал с V34. Отправка **56464896**, 12:53:35 UTC, **PENDING** на момент проверки.
Это автономный inference notebook, подготовленный из exact Speedy Raptors V34,
scriptVersionId **351665011**. Авторский public score **0.943**. Текущий статус
нашего запуска/отправки находится в `RESULT.json`; score автора нельзя приписывать нам.

## Почему выбрана эта основа

Недавний анализ показал подтверждённый разрыв между нашей веткой CF01 (0.886) и
воспроизведённым публичным PUB941 (0.941). За час выбрана наиболее сильная доступная
scored-основа, без нового поиска коэффициентов по LB и добавления слабых моделей.
Предыдущий результат PUB941 сохранён как резерв.

V34 отличается от скачанного latest: **четыре**, а не три CoAt-reader
(ResGated, Global96, D4, Repair-v1); outer weight 0.60 обычно вместо 0.65.
Repair-v1 находится внутри D4 dataset v2, в подпапке `repairv1/`.
Исторический CLI pull возвращал ошибку; source получен через официальный immutable
export URL и закреплён SHA256. Все успешные вычисления V34 сохранены.

## Навигация

| Путь | Назначение |
|---|---|
| `kaggle/*.ipynb` | Основной ноутбук: описание графа, 23 стадии, запуск и дальнейшая разработка |
| `kaggle/kernel-metadata.json` | Приватный Kaggle kernel, T4×2, Internet OFF, зависимости |
| `runtime/*.py` | Редактируемые стадии, соответствуют разделам notebook |
| `build_notebook.py` | Собирает notebook из стадий; обычный запуск сохраняет edits |
| `source_v34/` | Неизменённый исходник, author log, fetch receipt |
| `input_lock.json` | Проверенные версии подключений и происхождение |
| `harden_source.py`, `test_hardening.py` | Fail-closed изменения и проверки |
| `evaluate_oof.py`, `VALIDATION.md` | Критерий будущих тренировочных экспериментов |
| `visible/`, `RESULT.json` | Результат нашего запуска, проверки, submission ref |
| `preview.html` | Статический удобный просмотр notebook с оглавлением и раскрываемым кодом |
| `author_v34_outputs/` | Малые публичные outputs exact V34 и независимые проверки схем/арифметики |

## Пересборка и запуск

Из корня проекта:

```powershell
python experiments/SPRINT943/build_notebook.py
python -m pytest experiments/SPRINT943/test_hardening.py -q
kaggle kernels push -p experiments/SPRINT943/kaggle
kaggle kernels status dmitriigluzdov/rsna-knee-v34-documented-strict
```

`--refresh-from-source` у builder **перезаписывает runtime/*.py** закреплённым
исходником с safety patches. Без этого флага локальные изменения стадий сохраняются.
Не редактировать `source_v34/`: для нового source завести отдельный эксперимент.

После COMPLETE скачать только нужные outputs, затем:

```powershell
python experiments/SPRINT943/fetch_visible.py --version 1
python experiments/SPRINT943/validate_visible.py experiments/SPRINT943/visible
```

Code submission отправляет конкретную сохранённую Kaggle version. Номер берётся
из подтверждённого push, а не предполагается. В историю записываются version,
submission ref, hash исходника и output. Не отправлять обычный трёхстрочный CSV:
hidden inference выполняется Kaggle заново.

## Что улучшено и что ещё неизвестно

Добавлены проверки полного состава CoAt, конечности raw predictions, NPZ UID/labels,
полной потери acquisition, actual receipts и atomic final output. Исходный код мог
скрыто отбросить ветку и всё равно написать COMPLETE. Обработка допустимого частичного
восстановления DICOM сохраняется и журналируется.

Авторский visible run ~6 минут на трёх исследованиях не доказывает hidden runtime.
Полный OOF нового графа отсутствует. Target-specific weights и calibrator с protocol
features могут зависеть от public/site состава. Поэтому private-выигрыш не обещается.
Материал для следующего шага — воспроизводимый контроль и evaluator с legacy/known-only
AUC, coverage и fold metrics. Новое собственное обучение за этот спринт не заявляется.

До нашего GPU-запуска дополнительно выполнен `replay_author_blend.py`: финальный blend
из четырёх авторских probability NPZ, Raptor CSV и parent NPZ даёт **побайтно тот же
visible submission.csv**, SHA256 `8f0c5e7b2b561538dae55439a4d2e9173141096da3defd9bc0a4bac351eef37a`.
Это регрессия арифметики на трёх studies, не повторный neural inference и не OOF.
Новый release gate прошёл на реальных author receipts. D4 per-study JSONL из source
outputs получить не удалось; этот дополнительный проход здесь не проверен.

См. `VALIDATION.md`: существующий R(2+1)D18 имеет legacy 0.810603 и known-only 0.838955,
но это разные label subsets и **не OOF публичного ансамбля**.

Основной source: Mattia Angeli, Apache-2.0; полные credits сохранены в notebook.
