# Locked experiment specifications

R2 (2026-09-07): обучение и все preprocessing/OOF выполняются по
`../LOCAL_COMPUTE_PLAN.md` на НГУ; `../ops/compute_plan.json` задаёт приоритеты
submission. Полезные local-only результаты фиксируются без обязательного сабмита.
Frozen cache должен иметь encoder provenance без fold leakage. Все label variants
оцениваются на одной фиксированной weak-label reference; expert-58 у S15 — crossfit.

Этот файл снимает исследовательские развилки с ежедневного исполнителя. Менять
семейство модели, label source, geometry или blend rule без решения старшей модели
нельзя. Library/runtime версии, появившиеся при materialization, фиксируются в
`SPEC.json` и после первого validated artifact становятся immutable.

## Shared label assets

- `LABEL_PILKWANG_V1`: `pilkwang/rsna-knee-llm-labels` V1,
  `report_labels_v2.csv`, SHA-256
  `6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920`.
- `LABEL_STEVEN_V6`: `stevenleehans/rsna-knee-llm-report-labels` V6,
  `llm_labels_v2.csv`. `llm_labels_v4_blend.csv` запрещено считать независимым
  teacher: это уже Steven+Lixin blend.
- `LABEL_LIXIN_V1`: `lixin73/rsna-knee-llm-report-labels-sol56` V1,
  `labels_llm_gpt56sol.csv`. Этот dataset нужно подключить отдельно: в Tony bundle
  есть только его производный вклад.

Пока эти registry keys не `ready`, связанные S-ID не готовы к train/submit.

## S11–S15: labels

- `S11`: `LABEL_PILKWANG_V1`; общий DINOv2 ViT-S/14 frozen cache; фиксированный
  2-layer head: LayerNorm → Linear(D,256) → GELU → Dropout(0.1) → Linear(256,12).
  Одинаковый head для S11–S15; seed 2026, locked FOLDS_V1.
- `S12`: soft median consensus ровно трёх raw families: Pilkwang V1, Steven V6/v2,
  Lixin V1. Unknown/NaN не участвует в median/loss.
- `S13`: arithmetic mean тех же трёх raw families; unknown/NaN исключается из
  знаменателя. Никакой готовый `v4_blend` не используется.
- `S14`: median S12 + disagreement mask: loss weight `0`, если валидных teachers
  меньше двух или `max(label)-min(label) > 0.50`; иначе confidence weight
  `1 - (max-min)`.
- `S15`: OOF winner из S11–S14 + confidence-weighted loss; expert-58 sample weight
  ровно `6.0`. Остальное идентично winner.

Все пять heads используют уже вычисляемые A0 DINO features. Inference recipe:
`.75 rank(A0) + .25 rank(head)`; второй encoder запрещён.

## S16–S20: geometry

Backbone/head/labels/folds/seed берутся из S15. Меняется ровно geometry/series rule:

- `S16`: physical FOV 130 mm; central 60%; 9 real slices; 224×224.
- `S17`: 160 mm; central 75%; 12 real slices; 224×224.
- `S18`: 120 mm; 9 real slices; 336×336.
- `S19`: top-2 series на каждый plane/contrast slot; один attention-MIL pool.
- `S20`: два inputs 120/180 mm; один target-gated linear fusion layer.

Каждый вариант заменяет Transformer-side component A0, не добавляет второй A0.

## S21–S25: architecture

Labels/folds/geometry/seed фиксируются по winner предыдущих этапов:

- `S21`: DINOv2 ViT-S/14, six slot tokens, target-specific attention-MIL. Это control.
- `S22`: torchvision ResNet-18 2.5D, три независимых plane heads, learned 12-target
  linear fusion. Никакой замены backbone по усмотрению исполнителя.
- `S23`: timm ConvNeXt-Tiny 2.5D, тот же slice sampler и MIL head. CoAtNet не допускается.
- `S24`: torchvision `r3d_18`, 16-slice 3D windows, shallow 12-target attention pool.
  Raptor не допускается как подмена.
- `S25`: rank-mean двух лучших S21–S24 по locked OOF, seed set `{2026, 2027}`;
  если нет двух прошедших candidates — `excluded`.

Все exact package/library/pretrained-weight versions фиксируются до `ready`. Любая
невместившаяся в compute/runtime архитектура получает `excluded`, а не незаметную
замену меньшей/другой моделью.
