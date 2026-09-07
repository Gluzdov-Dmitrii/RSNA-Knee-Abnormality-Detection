# RSNA Knee Abnormality Detection: неделя 1

**Актуализация R2 от 2026-09-07:** compute budget, календарь и порядок подачи
оставшихся S11–S35 заменены [LOCAL_COMPUTE_PLAN.md](LOCAL_COMPUTE_PLAN.md).
Подготовка/обучение/OOF выполняются на НГУ, Kaggle — только финальный inference.
S01–S10 уже scored; лучший `0.937`. Ниже сохранена исходная стратегия и гипотезы;
актуальные статусы — ledger, приоритеты подачи — `ops/compute_plan.json`.

Снимок leaderboard сделан **2026-09-05 07:50 UTC**: `3 089` команд, наш best
`0.936` / rank `295`, лидер `0.954`, top-10 `0.949`, расчётная gold boundary —
rank `16` / `0.948`. Эти числа динамические; перед каждым пакетом нужен новый снимок.

## О чём competition

Для каждого MRI-исследования колена нужно предсказать 12 вероятностей:

`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`,
`PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.

Метрика — macro ROC-AUC по 12 колонкам. В train 4 407 исследований и 24 371 серия.
Экспертные image-derived метки есть только у 58 исследований; у остальных 4 349 —
мультиязычные radiology reports с пустыми target-колонками. В test reports нет:
финальная модель обязана работать только по изображениям. Полный input — около
569.76 GB и 820 тысяч DICOM-файлов.

Это [Code Competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements):
submit только из Notebook, internet off, CPU/GPU runtime `< 9 h`, выходной CSV
(обычно `submission.csv`). Лимит — 5 сабмитов в день, финально выбираются 2. Deadline —
2026-10-22 23:59 UTC, team merger — 2026-10-15. Призовой фонд $77 000: main top-10
и efficiency top-3. Официальные страницы: [overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview),
[data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data),
[evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation),
[rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).

## Что уже есть и чего нет

Есть 20 завершённых сабмитов: `0.936 × 12`, `0.935 × 4`, `0.934 × 2`,
`0.928 × 1`, `0.927 × 1`; 69 локальных notebooks (49 уникальных); публичные ветки
DINOv2/DINOv3, Raptor/CoAtNet, RadImageNet, MedicalNet и slot heads; Kaggle CLI;
пять приватных kernel slots и оркестраторы с контролем квоты/двух GPU jobs.

Нет локального raw train, собственных folds/OOF/weights, weak-label registry,
training pipeline и скрытых prediction vectors. Текущие 20 запусков — разведка
public code, а не воспроизводимая validation-система.

Свежий открытый anchor —
[Renta 0.937 meniscus residual](https://www.kaggle.com/code/renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid).
Pinned source: kernel ID `133076816`, version `1`, scriptVersionId `347142162`.
Его Hyakumanben-repro
byte-identical: это не независимая модель. Ветка
`prvsiyan/the-bee-s-knees-final-rsna-push` использует тот же граф и добавляет
пятиклассовую V6-рецептуру (kernel ID `133138832`, version `6`, scriptVersionId
`347412699`); её локальная
Gold-58 проверка перспективна, но в исходнике нет подтверждённого official LB
receipt для V6.

## Что потребуется для золота

По [Kaggle Progression](https://www.kaggle.com/progression/competitions) при 1 000+
команд gold — top-10 плюс примерно 0.2% участников: сейчас около top-16. Для запаса
нужен кандидат уровня `0.950+` и устойчивость на private LB, а не попадание ровно в
сегодняшние `0.948`.

Разрыв `0.012` нельзя надёжно закрыть весами одного public ensemble. Нужны:

1. Weak labels с явными `positive / negative / unknown`, negation, severity и
   confidence. Экспертные image labels авторитетнее report labels; расхождения
   ожидаемы, что подтверждает [host](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826).
2. DICOM preprocessing по физической геометрии: slice position/orientation, physical
   FOV, laterality, series/plane/contrast selection. Filename order использовать нельзя.
3. Target-aware multi-view/MIL: разным диагнозам нужны разные planes, series и pooling.
   Увеличение encoder само по себе дало лишь небольшой прирост в публичной
   [абляции](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154).
4. Study-level multilabel 5-fold OOF плюс site/scanner/protocol stress test. Случайные
   folds оптимистичны; metadata-shortcut слаб в
   [публичной проверке](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517).
   Gold-58 — только sealed sanity с bootstrap, не CV и не набор для подбора десятков весов.
5. Вторая действительно независимая модель и per-target OOF blending. Команда из
   2–5 участников с комплементарными моделями существенно повышает шанс на золото.

Host разрешил hosted LLM API для извлечения labels из reports в
[официальном ответе](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965).
Нужно фиксировать provider/model/version/prompt hash/cost и соблюдать правила данных;
raw reports нельзя публиковать.

## Где считать и сколько ресурсов нужно

Текущая машина: RTX 2060 6 GB, i7-9750H (6C/12T), 31.8 GB RAM; свободно 204.6 GB
на NVMe C:, 272.1 GB на SATA SSD D:, 1.415 TB на USB HDD E:.

Raw data поместятся только на E:, но 820 тысяч мелких DICOM на USB HDD сделают I/O
узким местом. RTX 2060 годится для smoke tests, frozen embeddings/heads и маленьких
224px runs; для full multi-view/3D fine-tune она тесна.

| Режим | VRAM | RAM | Быстрое свободное место | Что реально делать |
|---|---:|---:|---:|---|
| Kaggle/cloud-first | локально 6 GB достаточно | 32 GB | 50–200 GB | код, metadata, OOF, hashes; raw/caches/weights в private Kaggle assets |
| Минимальный local train | 16 GB | 32–64 GB | 1 TB NVMe | 224–288px 2D/2.5D, малые batches |
| Комфортный | 24 GB | 64 GB | 2 TB NVMe | 336px, multi-view, folds |
| 3D/high-res | 40–48 GB | 64–128 GB | 2–4 TB NVMe | volumetric branches и большие slice windows |

Полностью облачная разработка возможна и для золота: собственные weak labels,
caches, train, OOF и weights можно создавать в Kaggle/cloud. Локально обязательно
вести код, provenance и аналитику, но скачивать 570 GB не нужно. Только запускать
чужие public notebooks недостаточно: их plateau сейчас около `0.936–0.937`.

Текущий A0 сам по себе тяжёлый: около 3.85 GB входных weights/assets, примерно 30 GB
system RAM и расчёт на T4×2; в исходнике стоит 8-часовой watchdog. Visible 3-study
run (~203 s) нельзя линейно экстраполировать на hidden. V6-рецепты почти бесплатны
после получения компонентных predictions, но весь A0 graph имеет повышенный риск
подойти близко к 9-часовому лимиту; T4×1 для него не планировать.

## Критическое ограничение hidden test

Обычный output notebook содержит только visible placeholder из 3 исследований. При
submit Kaggle повторно выполняет notebook на скрытом test (~1 300 исследований), но
полный hidden `submission.csv` участнику **не возвращает**. Следовательно:

- нельзя скачать 20 старых hidden predictions, посчитать их correlations или собрать
  из них офлайн-бленд;
- SHA visible 3-row CSV не является SHA скрытого сабмита;
- любой scored blend обязан загрузить weights и пересчитать все компоненты внутри
  одного internet-off notebook с hidden runtime `< 9 h`;
- сохраняем recipe/code/config/model hashes, visible output, OOF predictions,
  submission receipt и score; hidden-vector hash/correlation недоступны.

Это обязательный контракт для всех экспериментов ниже.

## Бюджет недели и gates

`35` — верхняя очередь слотов, а не обещание сжечь все слоты. Пять submit/day —
предел, не KPI. Если weights/caches отсутствуют, unified graph не укладывается в
`8.5 h` rehearsal или эксперимент не distinct, S-ID остаётся `blocked/ready`, а
слот не заменяется случайным fork.

Точные research choices для дней 3–5 закреплены в
[`ops/EXPERIMENT_SPECS.md`](ops/EXPERIMENT_SPECS.md). Ежедневный исполнитель не
выбирает другой teacher/backbone «по ситуации»; отсутствующий approved label asset
является blocker, а не разрешением импровизировать.

R2 compute envelope: **0 Kaggle GPU-h на train/cache/OOF**, все подготовительные
стадии на НГУ. До reset 12 сентября soft cap RSNA — 1.5 h visible validation,
пересмотр после первых 0.5 h; фактический расход измеряется через `kaggle quota`.
35 S-ID остаются исследовательской очередью; после локального отбора подавать
обычно 2–3, максимум 5 готовых кандидатов за день. Полный runtime/data/queue budget
и правила `local_evaluated` указаны в R2.

### Неизменяемый anchor и promotion rule

`A0` — **не CSV**, а pinned runnable inference graph Renta/Bee со всеми точными
upstream versions и weights. В исходной рецептуре Transformer/Raptor имеют веса
`0.40/0.60`; для `Medial Meniscus` используется specialist route
`0.30 Transformer + 0.60 Raptor + 0.10 bag`. Порядок rank/merge копируется из
источника буквально. Bag-head технически выдаёт оба meniscus targets, но
опубликованный и проверенный residual применяет его только к Medial; Lateral bag
нельзя включать без отдельного S-ID и OOF evidence.

`A0` остаётся неизменным контрольным графом всю неделю. Новый branch входит в
target-wise composite `Q`, только если на locked folds:

- group ΔOOF AUC `≥ +0.003` относительно **leakage-free control той же собственной
  model family**, обученного на тех же folds/data (для label ablation — S11; для
  geometry — default-geometry head S15; для architecture — S21 как DINO control);
  знак положителен минимум в 4 из 5 folds;
- второй seed подтверждает знак;
- отдельный target не падает больше чем на `0.005`, иначе его weight обнуляется;
- Gold-58 bootstrap не показывает убедительно отрицательный эффект;
- public LB используется лишь как secondary check, не как единственный критерий.

У public A0 нет доказанно leakage-free OOF: его train predictions нельзя выдавать за
OOF и использовать как reference. Correlations для promotion считаются только между
собственными leakage-free OOF; A0 остаётся submission control по public receipt.

Для продолжения pipeline `LABEL_WINNER` и `GEOMETRY_WINNER` выбираются автоматически
после evaluation set: максимальный валидный OOF macro на locked folds; tie-break —
меньший runtime, затем меньший S-ID. Registry status `ready` означает operational
control следующего этапа, но не `promoted` в Q. Исполнитель сам применяет эти
детерминированные правила; ежедневный senior checkpoint не требуется.

## План: 7 пакетов × 5 S-ID

### День 1 — exact frontier и факторизация V6

Один pinned graph, все неуказанные targets бит-в-бит равны A0.

- **S01:** exact A0 control — воспроизвести подтверждённый Renta `0.937` graph.
- **S02:** exact V6 — Raptor outer weights: ACL `.80`, Lateral OA `.35`, PF OA `.35`,
  Synovitis `.35`, Baker's `.375`; Medial Meniscus сохраняет bag route.
- **S03:** только ACL `.80`; остальные targets A0.
- **S04:** только четыре non-ACL V6 targets: Lateral OA/PF OA/Synovitis `.35`,
  Baker's `.375`; ACL и остальные A0.
- **S05:** midpoint/shrink V6: ACL `.70`, Lateral OA/PF OA/Synovitis `.475`,
  Baker's `.4875`; Medial Meniscus сохраняет `.30/.60/.10`.

Контроль дублей: Renta и Hyakumanben byte-identical, поэтому второй exact fork не
сабмитить. Для каждого варианта сохранять `recipe.json` и code/config SHA.
Проверка Day 2: именованные `submission_Sxx.csv` получили HTTP 400. Для каждого
кандидата нужна immutable kernel version с выходом `submission.csv`; локальные
OOF/component caches можно переиспользовать, hidden outputs Kaggle — нельзя.

### День 2 — оставшиеся target-group directions в том же графе

Это диагностические, заранее фиксированные пробы вокруг базового Raptor weight `.60`:

- **S06:** только MCL, Raptor `.80`.
- **S07:** только Lateral Meniscus, Raptor `.80`; Medial Meniscus остаётся A0 specialist.
- **S08:** только Medial OA, Raptor `.35`.
- **S09:** только Effusion, Raptor `.35`.
- **S10:** Contusion и Fracture, Raptor `.80`.

Разница public score `0.001` считается шумным сигналом и не создаёт новый anchor.

### День 3 — weak-label ablation на одном frozen representation

Одинаковые folds, cache, backbone, head, seed и unified inference. Новые heads читают
уже вычисляемые A0 DINO embeddings, поэтому второй encoder не запускается. Внутри
notebook: `0.75 rank(A0) + 0.25 rank(Li)` по каждой target-колонке.

- **S11:** Pilkwang V1 `report_labels_v2.csv` — точный public-label control.
- **S12:** median consensus независимых Pilkwang V1, Steven V6/v2 и Lixin V1.
- **S13:** arithmetic-mean consensus тех же трёх raw label families.
- **S14:** median consensus + disagreement/unknown mask по locked threshold `0.50`.
- **S15:** лучший из предыдущих четырёх по locked OOF + confidence weights и expert-58
  sample weight `×6`; зависит от готовых результатов первых четырёх.

Если head нельзя встроить с переиспользованием features или unified A0+Li не
укладывается в 8.5 h, experiments блокируются; standalone Li не подставляется,
потому что это уже другая гипотеза.

### День 4 — geometry при фиксированных labels/model family

Каждый вариант обучается на тех же folds. Чтобы не считать почти 8-часовой A0 дважды,
`Gi` **заменяет только Transformer-side компонент** в исходной fusion-рецептуре;
Raptor, bag, target weights и остальные части графа остаются неизменными. Обычная
доля заменяемого компонента `.40`, для Medial Meniscus `.30` из-за `.10` bag.

- **S16:** physical sort, 130 mm FOV, central 60%, 9 real slices, 224px.
- **S17:** physical sort, 160 mm FOV, central 75%, 12 real slices, 224px.
- **S18:** physical sort, 120 mm FOV, 9 slices, 336px.
- **S19:** top-2 series per plane/contrast slot + attention/MIL.
- **S20:** dual-scale 120/180 mm target-gated fusion.

До full fold сначала smoke + one-fold/subset screening; плохой вариант не получает
дорогой train и submission только ради заполнения дня.

### День 5 — независимая архитектурная ошибка

Лучшие labels/geometry фиксированы. `Mi` заменяет Transformer-side компонент A0, а
Raptor/bag и fusion weights сохраняются. Это component-replacement test, не второй
полный A0 graph; каждый вариант обязан пройти runtime gate.

- **S21:** DINOv2-S diagnosis-specific slot attention/MIL.
- **S22:** ResNet-18 2.5D independent plane heads + learned fusion.
- **S23:** ConvNeXt-Tiny 2.5D branch.
- **S24:** torchvision `r3d_18`, 16-slice 3D branch.
- **S25:** two-seed rank ensemble двух лучших OOF-веток; зависит от первых четырёх.

### День 6 — target-wise composite, только подтверждённые группы

Для группы берётся один winner из дней 3–5 по promotion rule. Он заменяет A0
Transformer-side prediction только в изменяемых targets с долей не выше `.40`;
Raptor/bag считаются один раз, прочие targets — точный A0. Shared-feature heads
предпочтительны; независимый encoder допускается только после runtime rehearsal.

- **S26:** ACL + MCL.
- **S27:** Medial + Lateral Meniscus.
- **S28:** Medial OA + Lateral OA + PF OA.
- **S29:** Effusion + Synovitis + Baker's.
- **S30:** Contusion + Fracture.

Если для группы нет OOF winner, соответствующий S-ID получает terminal status
`excluded` с evidence (нулевой weight в Q), а не превращается в public weight sweep.

### День 7 — robust candidates и private hedge

- **S31:** composite Q из прошедших promotion rule веток, 5-fold ensemble; target weight
  нового сигнала не выше `.45`. До readiness разные group winners должны использовать
  один shared encoder либо быть distilled в один multi-head student.
- **S32:** S31 + deterministic crop/window TTA, только при hidden estimate `< 8.5 h`.
- **S33:** high-resolution branch только для meniscus/OA/fracture, остальные targets S31.
- **S34:** conservative shrink всего лучшего S31–S33 residual на 50% обратно к A0;
  target scope остаётся `all`.
- **S35:** private hedge: `.55 rank(A0) + .30 rank(best-own-DINO) +
  .15 rank(best-independent-conv-or-3D)` с нулевым weight для не прошедших target groups.

Два финальных кандидата после недели: лучший robust OOF/public и наиболее
декоррелированный conservative hedge. Выбор final submissions — отдельное решение.

## Что должно остаться после недели

- 35 строк ledger с честными `scored/ready/blocked/excluded`, а не обязательно 35 score receipts;
- pinned recipes, source/version IDs, hashes и visible 3-row outputs;
- immutable folds, OOF predictions и 12 per-target AUC для собственных веток;
- OOF raw/rank correlation matrix и do-not-retry registry;
- два воспроизводимых unified inference notebooks с estimate/runtime `< 9 h`;
- ясный ответ, где появился новый сигнал: labels, geometry, architecture или target group.
