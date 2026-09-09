# План R2: подготовка на НГУ, Kaggle для inference и submissions

Действует с **2026-09-07**. Этот документ заменяет compute budget, календарную
очередь и правила отбора оставшихся S11–S35 из `STRATEGY_WEEK_1.md`.
Гипотезы и source versions сохраняются в `ops/EXPERIMENT_SPECS.md`.
Выполненные S01–S10 и их receipts не переписываются.

## Что установлено

- S01–S07 дали `0.937`, S08–S10 — `0.936`; явного улучшения над S01 нет.
  Всего в сохранённой истории 30 сабмитов. Дальнейшие public-only weight sweeps
  имеют низкий приоритет.
- FOLDS_V1 и три teacher label sets готовы. Fold SHA:
  `3086df3341333f44adb883292da386857c3230eaa2d501514ddf827a2da11b1a`.
- Готов PIXEL_CACHE_V1: 4407 studies, 11.12 GiB uint8, без сырого DICOM.
- Не готовы: DINO cache, pinned ML/torch env, собственные heads/OOF. Raw ~500 GB DICOM по умолчанию не скачивается.
- Свежий `kaggle quota` 2026-09-07 около 12:10 UTC: **20.90/30.00 h использовано,
  9.10 h осталось**, reset `2026-09-12T00:00:00Z`. Это общая квота аккаунта.
- SSH identity подтверждены в этой сессии через Windows-клиент пользователя:
  `nsu-quadro → prepost`, `nsu-a100 → ngpu01`, `nsu-pc → desktop-7t0uo8i\user`.
  Sandbox под другим Windows-аккаунтом не видит SSH aliases; обычное разрешённое
  выполнение настроенного клиента работает. Это не неисправность VPN/НГУ.
- В 12:07 UTC общая очередь уже активирована: `initialized=true`,
  `mode=DIRECT_USER_AUTHORIZED`, `ALLOWED_WITH_RESOURCE_RESERVATION`,
  `slurm_required=false`. Старые сообщения `BLOCKED_COORDINATION` в отчёте Day 3
  и SETUP_STATUS устарели. Перед новым запуском читать живой control record.
- У RSNA remote manifest пока старый запрет `No reports, DICOMs ... in this tree`.
  Он не был изменён этой актуализацией. Перенос требует согласовать его с новым
  выбранным storage/data policy до копирования изображений.

Источники: `ops/reports/2026-09-06_day2.md`, `ops/reports/2026-09-07_day3.md`,
живые SSH/quota проверки; общий промпт:
`C:\Users\Dmitry\Desktop\Kaggle\Kaggle Agents\external-resources\AGENT_PROMPT.md`.

## Размещение задач

Цикл «гипотеза → 5-fold OOF → отбор» идёт на NSU, не на ноутбуке.

| Ресурс | Роль RSNA | Начальный предел одного job |
|---|---|---|
| HASEE RTX 2060 6 GB | только local import/syntax smoke (batch крошечный). Не 5-fold, не OOF, не лестница | 1 GPU; если свободно <4 GB — CPU-only smoke |
| CPU Linux, 56 cores / 112 threads, 251 GiB RAM | DICOM decode, physical sort/crop, shards, label merge, OOF metrics по готовым предсказаниям | 8 CPU threads, 32 GiB RAM; увеличивать после I/O pilot и проверки общей нагрузки |
| RTX 3080 10 GB, `nsu-pc` | **default** для S16–S23 2.5D/small CNN, когда PIXEL_CACHE_V1 лежит на локальном диске PC | 1 GPU, 4–8 CPU threads, ≤20 GiB RAM; оставлять VRAM для Windows |
| A100 80 GB, `nsu-a100` | **самый быстрый OOF**, пока кэш уже на Linux NFS; также high-res/MIL/3D, distillation, live DINO | 1 GPU/job; TMPDIR=/tmp и DataLoader workers>0 (NFS path ломает AF_UNIX) |
| Quadro RTX 6000 24 GB, `nsu-quadro` | frozen DINO cache; jobs, которым нужны 24 GB; overflow если 3080/A100 заняты | 1 GPU, 8 CPU threads, 32–64 GiB RAM, chunk ≤2 h |
| Вторая A100 80 GB | второй независимый fold/seed, если другие проекты не ждут | отдельная allocation; не считать две карты общей памятью 160 GB |
| Kaggle GPU | короткий Save & Run финального notebook и hidden grading | 1 активная RSNA session; train/cache/search запрещены |

S22 ResNet-18 2.5D на batch 16 занял ~1.6 GB VRAM (~29 s/epoch на Quadro, GPU ~40% из-за NFS и `workers=0`). Это **влезает в 3080** с запасом. 2060 (~5 GB свободно) для этого цикла не используем: мало памяти, слабый throughput, это не accelerator гипотез.

Пока 11.12 GiB cache не скопирован на `nsu-pc`, следующий 2.5D OOF запускать на **A100** (тот же NFS cache, без второй копии 11 GiB). После локальной копии на PC — 3080, если job влезает в 10 GB; A100, если нужен более крупный batch, 3D, или свободная карта и очередь пуста. Quadro не является default для small CNN.

Это предпочтения задач, а не постоянное закрепление GPU за проектом. Small heads
сначала пробовать на CPU: 4 407 строк frozen embeddings могут не требовать GPU.
Под конкуренцией — один GPU-job RSNA; свободная вторая A100 доступна только по общей
очереди. DDP на двух A100 не нужен для маленьких heads; независимые folds проще.

Внешнее обучение не отменяет Kaggle code competition: hidden test поступает только
в Kaggle notebook. Его нужно читать и предсказывать заново в scoring run.
[Kaggle Code Competitions](https://www.kaggle.com/docs/competitions).

## Хранилище и перенос

Один Linux project root для обоих серверов:
`/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection`.
Windows root: `C:\Users\User\kaggle\projects\rsna-knee-abnormality-detection`.
Окружения host-specific, но Linux data/cache общие: второй раз копировать на A100
те же файлы не нужно. `/home` и `/data` — NFS, их нельзя считать локальным NVMe.

В live `df` свободно около 7 TiB на `/home` и 39 TiB на `/data`. Это объём всего
export, **не подтверждённая персональная квота**. Выбрать разрешённый project subtree
и записать индивидуальный storage budget; просто обнаружить свободный `/data` мало.

Плановые, а не измеренные бюджеты:

- полный raw input: около 570 GB; один raw + compact cache + weights + запас —
  ориентир **0.8–1.0 TB**. При одновременном хранении архива и распаковки —
  **1.3–1.5 TB**. Размер зависит от того, что скачивается/распаковывается;
- bootstrap subset: 32 studies для correctness, затем 128 для timings, в пределах
  зафиксированного pilot budget; после успеха — полный train;
- uint8 cache PIXEL_CACHE_V1: 4407×6×9×224² = **11.12 GiB** pixels, downloaded
  from `dmitriigluzdov/rsna-knee-uint8-224-9-c130`. Это default train-pixel corpus.
  336² ≈25.0 GiB остаётся оценкой, не скачанным артефактом. Сырой DICOM ~500 GB
  по умолчанию не скачивается.
- pooled FP16 embeddings 4407×6×768×2 bytes ≈39 MiB. Slice/token-level features и
  несколько encoders могут занимать гигабайты. Сохранять нужный head layout;
- на RTX 3080 передавать только требуемые shards/embeddings/weights; полный raw не нужен.

Для NFS собрать sharded cache (например, последовательные shards 1–4 GiB с UID index),
чтобы каждая эпоха не открывала сотни тысяч DICOM. Bounded prefetch, измеренный I/O,
cache-key = data manifest + code + geometry + intensity rules + encoder hash + dtype.
Folds хранятся отдельно. При смене geometry/encoder cache-key меняется.

Данные скачивать существующим авторизованным Kaggle-клиентом на HASEE частями и
передавать с resume/hash-check в зарегистрированное NSU-хранилище. Kaggle credentials
на НГУ не копировать. Для 570 GB идеальное время передачи: ~12.7 h при 100 Mbit/s,
~1.3 h при 1 Gbit/s; VPN, small files и диск могут существенно увеличить время.
При 200 GB свободного локального SSD использовать shards/порционные файлы, а не
полный архив с распаковкой. Сначала проверить доступные download units API.

В НГУ для текущих S11–S15 достаточно разрешённых изображений, label tables и
metadata без `Report`; исходные reports не нужны. Raw/caches доступны только владельцу
и разрешённой команде. Правила внешних ресурсов и
[competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
проверяются перед staging. Proposed manifest policy: private competition train DICOM,
sanitized metadata и derived artifacts допустимы внутри зарегистрированного RSNA
subtree; credentials и raw reports по-прежнему исключены. Сохранить старую policy
и rationale изменения; не менять общий policy других проектов.

## Этап M0 — один раз подготовить перенос

1. Прочитать внешние README/ACCESS/WORKFLOW/RESOURCE_POLICY/SETUP_STATUS и живой
   `_control/COORDINATION_STATUS.json`. Текущая очередь уже существует; не создавать
   второй dispatcher или отдельную локальную очередь.
2. Проверить storage/quota и RSNA manifest, записать разрешённые data classes и peak
   budget; обновить только RSNA manifest в рамках подготовки переноса.
3. Создать отдельный dependency-pinned ML env, сохранив `py3.11-stdlib-v1`.
   Имеющиеся envs содержат только stdlib, без pip/torch. Нужны torch/torchvision,
   timm, numpy/pandas, pydicom и codecs, OpenCV, sklearn, safetensors по совместимым
   версиям A0. Зафиксировать wheel/runtime lock после local import smoke.
4. Выделить shared Python modules для preprocessing, feature extraction, heads и
   inference. Корень данных/weights/output задаётся аргументами; Kaggle launcher
   использует те же функции. Exact code commit и weights pin обязательны.
5. Провести CPU decode pilot 32 studies, затем GPU benchmark 128 studies с allocation.
   Проверить UID/order, missing slots, geometry, elapsed/VRAM/RAM и resume.
6. Сделать full DINO cache один раз, immutable UID index и manifest. На shared NFS
   extractor с чанками по study; завершённые hash-verified shards переиспользовать.

Пункты 1–4 и перенос не занимают GPU. Запрос GPU только после готовых inputs/env.
Протокол очереди — установленный
`nsu-quadro:/home/scientists/gluz_d_s/kaggle/_control/resource_queue.py`:
`request → RESERVED → started(PID,start) → heartbeat ≤60 s → release` после
подтверждённого завершения процесса. Все production queue calls через `nsu-quadro`.
Token держать в приватном run state. Потеря SSH или stale heartbeat не разрешает
повторный запуск. Под нагрузкой — chunks ≤2 h, ожидание 2/5/10 минут с CPU-работой.

M0 является планом подготовки, а не утверждением, что перенос/env/train уже выполнены.

## Валидация перед отбором

- Сохранить FOLDS_V1; split по StudyInstanceUID. Image fine-tune выполняется отдельно
  внутри каждого fold. Features от модели, обученной на held-out studies, не OOF.
- Для единого frozen cache разрешён только encoder без fitting на competition
  labels/images либо доказанно fold-safe extraction. Проверить A0 feature tap и
  checkpoint provenance. Если A0 tap обучен на всём train, cache/OOF для него
  блокируется; нужен fold-specific encoder, а не переименование train predictions.
- Все S11–S15 сравнивать на **одном фиксированном evaluation label table**, не каждый
  на своих teacher targets. Для non-expert studies: Pilkwang V1 бинаризация `≥0.5`,
  masked missing. Фиксировать UID mask и label hash. Это **weak-label OOF proxy**,
  а не достоверная оценка image AUC; дополнительно считать teacher disagreement.
- Expert-58: только genuinely held-out predictions. Для S15 с expert weight ×6
  нельзя оценивать checkpoint на тех же 58, на которых он обучен. Использовать
  по каждому case его fold-out prediction; обозначать `gold58_crossfit`, не sealed
  holdout. Сохранять отдельные данные weak OOF и expert crossfit.
- Нельзя использовать public A0 train predictions как OOF reference. OOF comparison
  — свои models на одинаковых folds. Candidate selection/blend tuning вложены в
  train-fold, или selection bias явно отделён от unbiased reporting.
- Для каждого candidate: 12 AUC с числом positives/negatives; undefined AUC = null;
  seed/folds, Δ к control, per-site/protocol stress, correlations и ресурсные метрики.

## Оставшиеся 25 гипотез: локальный поиск и очередь сабмитов

Дни теперь означают **пакеты**, не требование завершить этап за 24 часа. Machine
routing и submission policy каждого S-ID — в `ops/compute_plan.json`.

| Пакет | Расчёты вне Kaggle | Что подавать первым |
|---|---|---|
| S11–S15 | один shared DINO cache; 4 label heads ×5 folds; затем S15 winner + expert weights | S11 control, лучший S12–S14, затем S15; ещё 2 только при независимой полезной гипотезе |
| S16–S20 | все geometry variants на одном pilot fold; полный OOF только лучших двух | control/candidate с лучшим tradeoff AUC/runtime; максимум 2–3 |
| S21–S25 | DINOv2-S, ResNet18, ConvNeXt-Tiny, r3d18 screening; full folds двух лидеров | лучший DINO, лучшая независимая ветка, подтверждённый S25 ensemble |
| S26–S30 | все 5 target-group replacements оценить по OOF без Kaggle | только группы, прошедшие promotion; остальные `local_evaluated/excluded` |
| S31–S35 | компактный Q, TTA, high-res, shrink, hedge; trim/distill если нужно | S31, S34/S35; S32/S33 только при реальном запасе runtime |

Первые 24–48 часов после готового storage/transfer: M0, DINO cache, S11–S14 heads и
OOF. Срок условный, benchmark/transfer может его сдвинуть. S15 готовится после local
выбора winner; ждать public scores S12–S14 для выбора labels не требуется.

Следующая команда «засабмить следующие решения» означает: максимум 5 готовых
неподанных кандидатов из самого раннего подготовленного пакета, с приоритетами
выше. **Ожидаемый рабочий объём — 2–3 информативных сабмита на подготовленный пакет**,
а не пять любых каждый день. Недостаточно доказательств — продолжить local preparation.
После local screening неподанные кандидаты могут получить `local_evaluated` и
`decision=defer_submit`; это terminal для обхода пакетов, но не `scored`.
Перевод обратно в очередь — явное изменение dispatch plan с причиной, без нового S-ID.

## Экономия именно Kaggle quota

На Kaggle запрещены full train, train DICOM decode/cache, train feature extraction,
OOF search, grid search и длинные репетиции. Всё это выполняется на НГУ.
В inference bundle загрузить только выбранные weights, config, код и offline
dependencies. Train data/caches/OOF/results на Kaggle для inference не нужны.
Private Dataset/Model должен оставаться private и version-pinned.

Сейчас A0 notebook монолитный и может выполнять подготовительные расчёты во время
inference. **Сначала аудит и перенос всех train-dependent стадий**: calibration,
heads/scalers, statistics, label reads, train features. Сохранять fitted artifacts;
hidden cohort ranks/test-dependent normalization вычислять на текущем test. Проверить
на 32/128 train-as-test cases parity старого/нового graph и unchanged columns.
Во время rehearsal задать train mount недоступным: если inference падает из-за
неэкспортированного train artifact, пакет ещё не готов.

A100 training mixed precision допустим, но Kaggle T4 требует переносимого inference:
FP32 weights + FP16 autocast по протестированному пути, без обязательных BF16,
FlashAttention/A100-only kernels или предположения о 80 GB VRAM. Не менять pipeline
для visible 3 rows и hidden; разница только во входном размере.

Каждая финальная версия пишет **ровно `submission.csv`**. Day 2 документирует HTTP 400
для `-f submission_Sxx.csv`. Можно тестировать несколько recipes локально на одном
component cache, но каждый Kaggle submission — отдельная immutable version.
Нет гарантии reuse скрытого run или бесплатного переиспользования пяти CSV.

Учёт разделять:

1. `kaggle quota` — недельные accelerator hours аккаунта;
2. `competitions submission-limits` — дневные RSNA submissions;
3. visible Save & Run — измеренные wall minutes;
4. hidden grading — отдельный run, лимит competition и его observed/unknown billing.

Документация [GPU quota](https://www.kaggle.com/docs/efficient-gpu-usage) и
[Notebook runtime](https://www.kaggle.com/docs/notebooks) не доказывает, что каждый
час hidden grading списывается из weekly quota. Не умножать 35×9h как расход квоты
и не обещать hidden grading бесплатно. Фиксировать quota до/после; при параллельных
проектах attribution неизвестен. Submit для проверки биллинга запрещён.

До следующего reset RSNA: **0 h Kaggle на подготовку**, soft ceiling **1.5 h на
visible validation**, контрольная точка после первых 0.5 h. Это cap RSNA, не
резервирование оставшихся 9.10 h и не распределение квоты других проектов.
Если средний visible validation стоит q quota-hours, пакет n оценивается как n×q
плюс измеренный retry allowance; только фактические quota readings калибруют q.
Отдельно оценивать hidden runtime: safety gate `<8.5 h`, engineering target `<6 h`
на Kaggle hardware с RAM/VRAM margin. Скорость A100 не экстраполировать на T4 напрямую.

## Что передать следующему исполнителю

Обновлённый `LIGHT_MODEL_PROMPT.md` задаёт prepare и submit modes. Prepare запускает
M0/local compute по разрешённому workflow, submit выбирает прошедшие local gates
готовые версии. Разрешение на ежедневный пакет до пяти уже задано пользователем;
чтение внешнего one-shot template не сокращает явно разрешённый пакет до одного.
Одна попытка submit на candidate; pending/неясный ответ проверяется по receipts.

Записывать в RESULT: host/GPU, queue request/job/PID identity, env lock, cache hashes,
data paths/size, GPU wall hours, CPU/RAM/VRAM peaks, weak OOF reference и expert
crossfit, transfer times/bytes, export manifest, Kaggle visible runtime/quota deltas,
hidden estimate/observed runtime, submission ref, score/rank. Все данные для
следующей аналитики сохраняются даже у candidate без public submit.
