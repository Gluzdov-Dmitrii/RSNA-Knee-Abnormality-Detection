# Исполнитель RSNA — R2, локальная подготовка и экономные submissions

Скопируйте этот промпт в задачу исполнителя. Команды:

- «Подготовь следующий пакет на НГУ по R2» — подготовка данных, env, cache, train,
  OOF, export; Kaggle GPU и submissions не запускаются этой командой.
- «Засабмить следующие решения по R2, до пяти» — отправить готовые отобранные
  кандидаты в пределах квоты, дождаться scoring, обновить журнал. Если они не
  готовы, продолжить внешнюю подготовку и указать реальный blocker.

---

Ты исполняешь план RSNA в
`C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection`.
Competition: `rsna-knee-abnormality-detection`.

Прочитай полностью перед работой:

1. `LOCAL_COMPUTE_PLAN.md` — действующая R2-стратегия; она заменяет прежний compute
   budget/календарь/порядок подачи из `STRATEGY_WEEK_1.md`.
2. `ops/compute_plan.json`, `ops/experiment_ledger.csv`, `ops/LEDGER_SCHEMA.md`.
3. `ops/EXPERIMENT_SPECS.md`, `ops/promotion_registry.csv`, `ops/do_not_retry.csv`.
4. Последний дневной отчёт, `ops/DAILY_REPORT_TEMPLATE.md`, правила проекта.
5. Общий NSU-промпт
   `C:\Users\Dmitry\Desktop\Kaggle\Kaggle Agents\external-resources\AGENT_PROMPT.md`
   и перечисленные в нём README/SETUP_STATUS/ACCESS/WORKFLOW/RESOURCE_POLICY.

User authorization на команду «до пяти» покрывает весь явно названный пакет; общие
one-shot defaults не превращают его в один submission. Если текущая команда —
только plan/review/prepare, не отправляй submissions. Не покупай ресурсы, не выбирай
final submissions, не меняй команды и не публикуй private artifacts публично.

## Проверка состояния

Сохраняй UTC, HEAD/status, последние receipts, `kaggle quota`, дневной
`competitions submission-limits`, running/pending jobs аккаунта. CLI:
`C:\Users\Dmitry\.venvs\kg\Scripts\kaggle.exe`.
Исторические числа не выдавай за текущие. Ничего не выводи из credential store.

S01–S10 уже scored, не повторяй. FOLDS_V1, PIXEL_CACHE_V1 и три label assets уже
готовы; сверяй hashes, а не перестраивай их. Следующий пакет S11–S15 зависит от
DINO cache/heads. Pixel-path (S22 и др.) учится на 11.12 GiB uint8 cache, не на
~500 GB DICOM. Не скачивай официальный train zip по умолчанию.

SSH через существующий Windows-клиент пользователя и aliases `nsu-quadro`,
`nsu-a100`, `nsu-pc`. Sandbox alias failure не означает, что сервера недоступны:
используй штатный permissions flow для настроенного клиента, не копируй ключи.
В 2026-09-07 12:07 UTC direct coordinator активирован; перепроверь живой
`/home/scientists/gluz_d_s/kaggle/_control/COORDINATION_STATUS.json`.
Не воспроизводи устаревший BLOCKED_COORDINATION/Slurm blocker из старых заметок.

## Prepare mode — все тяжёлые расчёты вне Kaggle

Следуй M0 в R2. Проверить RSNA storage manifest/quota, согласовать data classes с
переносом, передавать разрешённые train images и labels без credentials/raw reports.
Не копировать полный архив, если peak disk budget не помещается. Linux root общий
NFS для двух hosts, Windows root другой; читать точные пути из SETUP_STATUS.
ML env создавай отдельно с pinned dependencies; stdlib-only env не готов к torch.

Вынеси train-dependent операции A0 из inference: train features, fitted heads,
calibration, statistics должны стать weights/config artifacts. Препроцессинг
train/inference должен использовать общие функции. Offline export не должен
требовать train mount. Test-dependent ranks считаются на текущем test.

Начни с CPU decode 32 studies, затем GPU pilot 128; запиши throughput/VRAM/RAM/I/O.
На основании pilot оцени duration/space полного cache. Выполняй resumable shards,
проверяй hashes и UID index, не пересчитывай завершённые shards после разрыва SSH.

Placement: HASEE RTX 2060 — только local smoke, не 5-fold OOF. CPU для labels и
метрик по готовым предсказаниям. RTX 3080 10 GB — default small 2.5D/CNN, когда
PIXEL_CACHE_V1 локален на `nsu-pc`. A100 — самый быстрый OOF при NFS cache (уже
лежит на Linux) и для high-res/MIL/3D/live DINO. RTX 6000 24 GB — DINO cache,
24 GB jobs, overflow. Не ставить small CNN на 2060 и не делать Quadro default,
если 3080 или свободный A100 подходят. Вторую A100 брать для независимого fold
только при allocation и без ожидающих проектов. Две A100 не являются одной
памятью 160 GB. На Linux DataLoader: TMPDIR=/tmp, не длинный NFS path.

Используй только общую очередь
`nsu-quadro:/home/scientists/gluz_d_s/kaggle/_control/resource_queue.py`.
Проверь actual CLI/schema перед request. Inputs/env должны быть готовы до
reservation. Под конкуренцией — один GPU-job RSNA, chunks ≤2 h, heartbeat ≤60 s,
checkpoint/resume, release только после подтверждённого выхода своих процессов.
Не вытесняй чужие jobs и не заводи вторую очередь. Пока ждёшь 2/5/10 минут, делай
CPU-работу. Lease tokens храни только в приватном run state.

## Отбор S11–S35

Гипотезы/архитектуры/label sources брать из EXPERIMENT_SPECS. На всех вариантах
используй locked folds, одинаковую fixed evaluation table и её mask/hash.
Weak-label OOF — proxy; отдельно сохраняй expert-58 crossfit. Не оценивай S15 на
58 случаях тем checkpoint, который на них учился. Encoder для общего frozen cache
не должен содержать fitting на validation folds. A0 train predictions не являются OOF.

Сначала local screen, затем 5-fold OOF лидеров; второй seed только у winners.
Операционный LABEL/GEOMETRY winner — max валидный OOF, tie-break runtime/S-ID;
включение target group в Q — по promotion rule с учётом reference/proxy limitations.
Public LB не используется как основной выбор labels/model/blend.

Выбирай из самого раннего подготовленного пакета по `submit_priority` и gates
`ops/compute_plan.json`: сначала S11, затем лучший S12–S14, затем S15. Дополнительные
два кандидата только если есть самостоятельная полезная гипотеза и бюджет.
Обычный пакет — 2–3 submissions, максимум 5; повторный sweep public weights не нужен.

Candidate с полными local metrics, отложенный по compute policy, получает
`status=local_evaluated`, `decision=defer_submit`, пустые submission ref/score.
Он закрывает local evidence gates и не мешает следующему пакету. При провале
dependency — `excluded` с причиной/нулевым downstream weight; missing data пока
исправимо — `blocked`, не выдуманное завершение. Scored никогда не повторяй.

## Submit mode — минимальный Kaggle runtime

До Kaggle подготовь export bundle: code, chosen weights, config, offline wheels,
manifest/SHA. Через локальный авторизованный CLI можно создать/версионировать private
Kaggle asset для этого пакета; это подготовка явно запрошенного submit. Не загружай
train cache, raw reports, DICOM, OOF или credentials в inference bundle.

На НГУ проверить parity с A0 на одинаковом 32/128-study input, geometry, missing
series, column order и changed targets. Checkpoint экспортировать переносимо: T4
FP16/FP32 path, без обязательного A100 BF16/FlashAttention и предположений о 80 GB.

Каггл выполняет только inference по текущему mounted test. Один immutable
kernel/version на candidate, выход **`submission.csv`**. Именованные
`submission_Sxx.csv` уже дали HTTP 400; не повторять этот эксперимент. Hidden
predictions не доступны участнику, local/старый hidden CSV использовать нельзя.

До submit: code/config/weights hashes distinct; internet off; 13 колонок в official
sample order, exact UID order/set, unique UID, finite `[0,1]` predictions. Встрой
assertions для произвольного test size. Проводить один короткий visible Save & Run
для выбранной версии; full rehearsal/train/cache на Kaggle запрещены.

Kaggle budget R2 до 12 сентября: 0 h preparation, soft ceiling 1.5 h visible
validation RSNA, checkpoint после первых 0.5 h. Переснимать account quota; этот
ceiling не резервирует ресурсы других проектов. Hidden estimate должен быть
`<8.5 h`, engineering target `<6 h`; не экстраполировать A100 runtime прямо на T4.
Billing hidden grading записывать observed/unknown, не считать автоматически
равным weekly GPU debit. При параллельных чужих jobs quota attribution неизвестен.

Для каждой разрешённой подачи — один submit call с S-ID/hash в description.
Сразу записать ref. `submitted` означает только poll, `ready_to_run=no`; после
timeout сначала read-only reconciliation. Не повторять запрос ради пустого stdout.
После score записать status, score/rank, time, top-1/top-10/gold boundary/team count.

## Журнал и завершение

Для каждого S-ID сохраняй immutable code/recipe/ORIGIN, RESULT.json/RESULT.md:
source versions, model/data/cache/folds/env hashes, compute host/GPU/allocation,
run/PID identity, runtime/resource peaks, weak OOF и expert crossfit, correlations,
export/parity, quota delta, kernel/version, submission receipt или local-only reason.
В ledger — status; в compute plan — placement/priority. Новые public результаты
записывать отдельно от local metrics. Не backfill-ить старые записи догадками.

Отчёт по шаблону включает готовность следующего пакета, jobs, blocker, GPU-hours
НГУ/Kaggle отдельно, transfer/cache footprint, resource release/cleanup receipt.
Сохраняй latest resumable и best validated checkpoints, остальные удаляй только
после проверки отсутствия зависимостей и экспорта результатов согласно shared policy.

Проверь изменения, сделай commit по правилам проекта. Push — только при выполнении
его условий; чужие scratch files не включать. Итог пользователю: подготовлено,
submitted/scored/local_evaluated, scores, quota, следующий concrete task и blocker.

---
