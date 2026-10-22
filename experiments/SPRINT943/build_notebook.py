"""Build a standalone, documented Kaggle notebook from editable stage sources.

First run: --refresh-from-source applies audited safety patches to pinned V34.
Later runs read runtime/*.py; they never overwrite your edits without that flag.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_v34/bend-the-knee-to-speedy-raptors-the-original.ipynb"
SOURCE_SHA = "a025fcd5aa25373ac380f60746ba5ca17e70e97f714fdc17e01ce414ec2163b5"
KERNEL = "dmitriigluzdov/rsna-knee-v34-documented-strict"
NAME = "rsna-knee-v34-documented-strict.ipynb"

SECTIONS = [
    ("artifact_contracts", "1. Контракты внешних моделей", "Проверки манифестов, версий библиотек и CoAtNet runtime. Эти функции выполняются позже, когда подключены GPU и входы. Численная рецептура каждой модели сохранена; механизмы загрузки выделены в отдельный раздел."),
    ("dense_sampling", "2. Выбор физических срезов", "Capacity-aware sampling распределяет доступные срезы между анатомическими слотами. Используются реальные срезы; upsampling не считается дополнительной информацией. Разные семьи сохраняют собственные crop, orientation и normalization."),
    ("environment_and_audit", "3. Среда, конфигурация и журнал", "Требуются **GPU T4 ×2**, Internet OFF. Проверяем обе CUDA-карты. Фиксируем seed, бюджет 8 часов, schema из 12 targets и функции проверки UID, конечности и диапазона. Значения учебных EPOCHS/LR в унаследованном коде не запускают обучение; активные DINO-настройки загружаются из manifest."),
    ("input_paths", "4. Входы соревнования", "Используются test.csv, test_series.csv и test_series/. Train pixels и reports не участвуют в прогнозе; train.csv читается исходным кодом только для sizing памяти. Проверяется однозначность путей и наличие generic DINOv2-S."),
    ("dicom_metadata", "5. Метаданные DICOM и сторона колена", "Заголовки описывают геометрию, laterality и тип последовательности. Fat-suppression включает распознавание SMART FAT/Dixon water. Это часть checkpoint-specific preprocessing; изменение требует отдельного эксперимента."),
    ("slot_routing", "6. Раскладка последовательностей по слотам", "Для каждого исследования выбираются серии нужной плоскости и контраста. Реально отсутствующий тип серии разрешён маской. Ошибку чтения существующей серии нельзя незаметно объявлять отсутствием."),
    ("dicom_decode", "7. Порядок и декодирование срезов", "Физический порядок, выбранное окно, crop и intensity processing остаются согласованными с исходной моделью. Допустимое восстановление отдельных повреждённых срезов записывается в журнал; полная потеря нужной серии блокирует финальную публикацию."),
    ("laterality", "8. Нормализация laterality", "Правая сторона отражается по оси, соответствующей плоскости. Не следует применять одно общее отражение ко всем семействам: веса обучались с разными контрактами."),
    ("pixel_cache", "9. Кэш пикселей", "Повторные чтения сокращаются через общий cache и memo порядка. Ключ содержит серию и правила; это ускорение не добавляет новые модели. Проверки охвата предотвращают потерю исследований."),
    ("dino_slot_head", "10. DINO: attention по слотам", "SlotHead строит отдельное attention для каждого finding. Маски исключают отсутствующие acquisition slots; в некоторых checkpoints используются анатомические priors."),
    ("dino_model", "11. DINO: backbone и признаки", "Срезы нормализуются ImageNet mean/std. CLS и pooled patch features поступают в slot head. Размер изображения и pooling определяются конфигурацией конкретного checkpoint."),
    ("dino_factory", "12. DINO: создание моделей", "Generic DINOv2-S задаёт архитектуру. Каждый competition checkpoint загружается строго и проверяется fingerprint. CPU templates сокращают повторное создание архитектуры."),
    ("dino_ensemble", "13. DINO: 20 моделей и общий prefix", "Все **20/20** members обязательны. Первые шесть блоков разделяются только после SHA-проверки и численного сравнения. Window pooling зависит от target; итоговая ветвь усредняет percentile ranks. Вспомогательный soft-pool не является дополнительной активной ветвью."),
    ("dino_submission_contract", "14. Контракт промежуточной таблицы", "Строки выравниваются по StudyInstanceUID; порядок 12 столбцов фиксирован. Промежуточный результат сохраняется как _pipeline_stage.csv, не как готовая отправка."),
    ("run_dino", "15. Запуск DINO", "Первый вычислительный этап: декодирование, fingerprint каждого member, inference и rank aggregation. По завершении освобождаются временные CPU templates."),
    ("a5_configuration", "16. A5: пять fold-моделей", "A5 использует шесть plane × fat-suppression слотов, 16 срезов, 336 px, crop 130 mm и окно 12–88%. Все пять fold-checkpoints обязательны. Это отдельная семья с собственным preprocessing."),
    ("a5_decode", "17. A5: подготовка исследований", "Сохраняются исходные InstanceNumber ordering и per-slice intensity. Наличие подходящих acquisitions проверяется отдельно от ошибок декодирования."),
    ("a5_model", "18. A5: модель и strict loading", "Backbone, pooling и head восстанавливаются из cfg checkpoint. Проверяются все encoder/head tensors, а не только несколько совпавших ключей."),
    ("run_a5", "19. Запуск A5 на двух T4", "Сохраняются исходный microbatch и режим precision. Допустим существующий повтор в FP32 при нечисловом результате, после него невалидные прогнозы блокируют отправку. Реальное отсутствие acquisition обрабатывается отдельно."),
    ("blend_a5", "20. DINO + A5", "При доступности A5 сохраняется смесь 0.48 DINO + 0.52 A5. При подтверждённом отсутствии нужных данных остаётся DINO. Веса не подбирались в этом спринте."),
    ("radimagenet", "21. RadImageNet и сохранённый calibrator", "Используются ResNet50 RadImageNet, reference/E13 heads и зафиксированные E10/E13/E11 layouts. Calibrator включает protocol features: это риск переноса между клиниками. Его оставляем для воспроизведения V34; ни полный OOF, ни private-выигрыш calibrator здесь не доказаны. Сжатый payload — коэффициенты исходного calibrator, не скрытое обучение."),
    ("raptor_coat_and_blend", "22. Raptor + четыре CoAtNet-reader", "Raptor: MaxSpan forward/reverse и Native384/Dense, веса 0.60/0.10/0.20/0.10 в исходном порядке arms. CoAt: **ResGated, Global96, D4, Repair-v1**. Для CoAt усредняются вероятности членов, затем берётся rank; inner blend = 0.60 Raptor + 0.40 CoAt без промежуточного rerank. Outer CoAt/Raptor weight: 0.60 обычно, ACL/Lateral OA/Fracture 0.75, Medial Meniscus 0.80, Lateral Meniscus 1.00. Это сохранённые source choices, а не новый OOF-оптимум."),
    ("release", "23. Проверка полного графа и submission.csv", "Файл submission.csv появляется только после проверки обязательных семей, receipts, покрытия UID и finite predictions. Atomic write защищает от частично записанного CSV. Строгие gates меняют поведение при ошибке; успешные формулы V34 сохранены."),
]

INTRO = """# RSNA Knee — V34: сильная база для дальнейшей разработки

**Назначение:** воспроизводимый inference полного открытого ансамбля Speedy Raptors V34
с объяснением этапов и строгими проверками. Это использование опубликованных весов,
а не наше новое обучение.

| Что известно | Доказательство |
|---|---|
| Авторская scored-версия | [Mattia Angeli, V34 / 351665011](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-speedy-raptors-the-original?scriptVersionId=351665011), public **0.943** |
| Наш подтверждённый резерв | Rice/PUB941, public **0.941**, submission 56243894 |
| Наш score этого нового пакета | Проверять RESULT.json; авторский 0.943 не является нашим результатом |
| Полный OOF данного графа | Недоступен; private качество неизвестно |

## Как запустить и как читать

1. На Kaggle подключить inputs из kernel-metadata.json; **T4 ×2**, **Internet OFF**.
2. Выполнить **Run All**. Обычный запуск видит лишь 3 placeholder studies. Он проверяет
   работоспособность и численную эквивалентность, но не качество или hidden runtime.
3. Успех: submission.csv, btkd_v559_complete.json, sprint943_run_manifest.json,
   diagnostics/ и receipts всех четырёх CoAt-ветвей.
4. Для leaderboard нужен **code submission конкретной версии**: Kaggle повторно запускает
   код на скрытых данных. Локальный трёхстрочный CSV не содержит hidden predictions.

Ноутбук автономен: все исполняемые стадии включены ниже. В репозитории те же стадии
лежат в `runtime/*.py`. Их можно менять и запускать `python build_notebook.py`;
`--refresh-from-source` явно сбрасывает стадии к закреплённому V34 с safety patches.
`source_v34/` — неизменённое происхождение; `harden_source.py` — проверяемый diff поведения
при сбоях; `evaluate_oof.py` — отдельный критерий для будущих train-экспериментов.

## Карта вычислений

```text
test DICOM + metadata
  ├─ DINOv2-S ×20 → A5 ×5 → RadImageNet heads + saved calibrator ─┐
  └─ Raptor ×4 views + CoAt (ResGated / Global96 / D4 / Repair-v1) ┤
                 UID alignment → fixed target weights → rank ──┘
                                  ↓ integrity gates
                              submission.csv
```

Качество на private не гарантируется. Наша защита в этом пакете — сохранение сильного
scored-графа и checkpoint preprocessing, полный состав, отсутствие нового поиска по LB
и явный протокол независимой валидации ниже. Эти меры сами по себе не доказывают прирост AUC.
"""

NEXT = """## 24. Как развивать решение после этого спринта

**Сначала сохранить контроль.** Код, source SHA, mounted source IDs, manifests и output
SHA входят в receipts. Менять один содержательный фактор за эксперимент и сохранять
member predictions, чтобы отличать новый сигнал от случайного ранжирования.

**Валидация до нового blend.** `evaluate_oof.py` считает legacy weak-AUC и known-only
YES/NO AUC, покрытие каждого target, fold metrics и проверяет UID/fold integrity.
27.32% старых weak labels — UNK; их нельзя молча объявлять истинными отрицательными.
Known-only тоже имеет selection bias и показывается вместе с coverage.
Study-level folds не доказывают независимость пациентов/клиник: проверить patient/site
и дубликаты до расширения обучения. Gold58 — ограниченный дополнительный аудит.

**Следующая модель.** Frozen generic DINOv2-S + небольшой slot-attention head — минимальная
обучаемая ветка. Сначала pilot на двух заранее выбранных folds с одинаковым бюджетом:
confidence/UNK policy или более плотный native input. При устойчивом результате — 5-fold
OOF. Competition checkpoints из этого notebook на train не становятся OOF.

**Ансамблирование.** Подбор весов проводится внутри meta-validation/cross-fit; результат
подбора на тех же OOF labels имеет оптимизм. Добавлять старые CNN/3D только после
проверки дополнительного сигнала. Публичные forks одного графа не независимы.

**Риски, которые нужно измерить.** Target-specific outer weights, protocol-aware calibrator,
различие частот патологий между public/private, site/scanner shift, редкие acquisition
layouts и hidden runtime. Снятие calibrator или shrinkage весов — отдельная гипотеза;
автоматически объявлять такой вариант лучше на private нельзя.

## 25. Происхождение и благодарности

Основной код: Mattia Angeli, **Speedy Raptors V34**, Apache-2.0. Оригинальные credits:
Jiwei Liu, Pilkwang, Sofia Anjenje, Antoine G., prvsiyan, Marwan Mahmoud,
dreaddevelopment / Roman Tamrazov, renta.k и Anvith Pothula.
Права и условия каждого input asset учитываются отдельно от лицензии notebook.
Наш вклад: закрепление scored source, документация, проверка зависимостей,
fail-closed release gates и воспроизводимый evaluator для следующих экспериментов.
"""


def cell(kind, source, name):
    result = {"cell_type": kind, "id": name, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(refresh=False):
    source_bytes = SOURCE.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_SHA
    original = json.loads(source_bytes)
    runtime = ROOT / "runtime"
    runtime.mkdir(exist_ok=True)
    if refresh:
        from harden_source import harden_notebook
        hardened = harden_notebook(original)
        code_cells = [c for c in hardened["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) == len(SECTIONS), (len(code_cells), len(SECTIONS))
        for i, (item, (slug, _, _)) in enumerate(zip(code_cells, SECTIONS), 1):
            (runtime / f"{i:02d}_{slug}.py").write_text("".join(item["source"]), encoding="utf-8")
    cells = [cell("markdown", INTRO, "overview")]
    cells.append(cell("code", "import json, os\nfrom pathlib import Path\n"
        "os.environ['HF_HUB_OFFLINE'] = '1'\nos.environ['TRANSFORMERS_OFFLINE'] = '1'\n"
        "os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'\n"
        f"SPRINT943_SOURCE = {repr({'version':34, 'script_version_id':351665011, 'sha256':SOURCE_SHA})}\n"
        "print('Pinned source:', SPRINT943_SOURCE)\n", "provenance"))
    hashes = {}
    for i, (slug, title, description) in enumerate(SECTIONS, 1):
        path = runtime / f"{i:02d}_{slug}.py"
        code = path.read_text(encoding="utf-8")
        compile(code, str(path), "exec")
        hashes[path.name] = hashlib.sha256(code.encode()).hexdigest()
        cells.append(cell("markdown", f"## {title}\n\n{description}\n\nРедактируемый исходник: `runtime/{path.name}`.\n", f"section-{i:02d}"))
        item = cell("code", code, f"stage-{i:02d}")
        if len(code) > 15000:
            item["metadata"]["jupyter"] = {"source_hidden": True}
        cells.append(item)
    cells.append(cell("code", "# Machine-readable run evidence; no prediction transformation.\n"
        "import importlib.metadata as _pkg\n"
        "_sprint_versions = {}\n"
        "for _name in ('torch','torchvision','numpy','pandas','pydicom','timm','transformers','opencv-python-headless'):\n"
        "    try: _sprint_versions[_name] = _pkg.version(_name)\n"
        "    except _pkg.PackageNotFoundError: _sprint_versions[_name] = None\n"
        "rsna_json('/kaggle/working/sprint943_run_manifest.json', {\n"
        "    'source': SPRINT943_SOURCE, 'libraries': _sprint_versions,\n"
        "    'test_studies': len(_RSNA_TEST_IDS), 'outer_weights': _coatnet_weight,\n"
        "    'submission_sha256': rsna_sha('/kaggle/working/submission.csv'),\n"
        "    'elapsed_seconds': time.time()-T0, 'full_graph_oof_available': False,\n"
        "    'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})\n"
        "print('Submission and run evidence saved. Hidden score requires a code submission.')\n", "run-evidence"))
    cells.append(cell("markdown", NEXT, "development"))
    metadata = {k:v for k,v in original["metadata"].items() if k in ("kernelspec","language_info","kaggle")}
    metadata["kaggle"]["isGpuEnabled"] = True
    metadata["kaggle"]["isInternetEnabled"] = False
    metadata["sprint943"] = {"source_sha256":SOURCE_SHA,"source_version_id":351665011,"runtime_sha256":hashes}
    notebook = {"cells":cells,"metadata":metadata,"nbformat":4,"nbformat_minor":5}
    package = ROOT / "kaggle"
    package.mkdir(exist_ok=True)
    out = package / NAME
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n",encoding="utf-8")
    base = json.loads((ROOT/"runtime_metadata_template.json").read_text(encoding="utf-8"))
    base.pop("id_no",None)
    base.update(id=KERNEL,title="RSNA Knee V34 Documented Strict",code_file=NAME,
                is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,machine_shape="NvidiaTeslaT4")
    lockpath=ROOT/"input_lock.json"
    if lockpath.exists():
        lock=json.loads(lockpath.read_text(encoding="utf-8-sig"))
        # Verified CLI-ready overrides are optional; never guess version syntax.
        for key,value in lock.get("metadata_overrides",{}).items():
            if key in ("dataset_sources","kernel_sources","model_sources"):
                base[key]=value
    (package/"kernel-metadata.json").write_text(json.dumps(base,indent=2)+"\n",encoding="utf-8")
    receipt={"notebook":str(out),"notebook_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
             "source_sha256":SOURCE_SHA,"code_cells":sum(c["cell_type"]=="code" for c in cells),
             "markdown_cells":sum(c["cell_type"]=="markdown" for c in cells),"runtime_sha256":hashes}
    (ROOT/"build_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in receipt.items() if k!='runtime_sha256'}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-from-source",action="store_true")
    build(parser.parse_args().refresh_from_source)
