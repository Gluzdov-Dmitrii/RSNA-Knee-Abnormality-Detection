# Experiment ledger contract

`experiment_ledger.csv` — единственный источник **очереди, статуса, readiness и
dependencies**. Полное доказательство результата хранится в машинно-читаемом
`experiments/Sxx/RESULT.json`, путь на него обязателен в `evidence_path`.

## Lifecycle

Допустимые `status`:

`planned → preparing → ready → submitted → scored → promoted|rejected`

`blocked` допустим из любого незавершённого состояния; после устранения причины он
возвращается в `preparing`. `excluded` — terminal status для заранее запланированного
слота, чья dependency доказанно не прошла gate; нужен `RESULT.json` с причиной и
нулевым downstream weight. `ready_to_run=yes` разрешено только при выполненных
dependencies, наличии immutable artifact и успешном preflight. `submitted` означает,
что есть Kaggle submission ref и `ready_to_run=no`: такой S-ID разрешено только
poll-ить, но не resubmit. `scored` — что public score уже получен. После terminal
`FAILED/CANCELLED` причина фиксируется; возврат в `preparing` допустим только для
исправленного нового artifact.

`artifact_status`: `source_identified`, `source_pinned`, `recipe_defined`, `missing_training_assets`,
`waiting_dependency`, `building`, `validated`, `runtime_failed` или `invalid`.

Допустимые `decision`: `pending`, `needs_replication`, `promote`, `reject`, `exclude`.
`needs_replication` не является status: строка остаётся `scored`, пока старшая модель
не примет terminal decision.

## Dependencies

Несколько зависимостей разделяются `;`. Формат `NAME@gate`, например:

- `S01@artifact` — immutable artifact и recipe hash готовы;
- `S11@oof` — OOF evidence записан и прошёл schema validation;
- `S31@scored` — Kaggle score получен;
- `GROUP_OA@promoted` — в registry выбран winner по promotion rule.
- `LABEL_ABLATIONS@evaluated` — каждый обязательный candidate имеет OOF evidence
  либо явно зафиксирован как terminal `rejected/excluded` с причиной.
- `S26@evidence` — S26 имеет валидный RESULT независимо от исхода `scored`,
  `rejected` или `excluded`; `blocked` без terminal evidence gate не закрывает.

Исполнитель не имеет права очистить `blocker` и поставить `ready_to_run=yes`, пока
каждый gate не подтверждён ссылкой/hash в `RESULT.json` или registry.

Именованные shared assets и group winners хранятся в `promotion_registry.csv`;
известные дубли и запрещённые повторные workflows — в `do_not_retry.csv`.

### Promotion registry

Допустимые registry `status`: `missing`, `waiting`, `building`, `ready`, `evaluated`,
`promoted`, `rejected`. `missing/waiting/building/rejected` не закрывают положительный
gate. Правила:

- `@ready`: status `ready` или `promoted`; обязательны `artifact_ref`, `evidence_ref`,
  `updated_at_utc`;
- `@evaluated`: status `evaluated` или `promoted`; обязателен evidence со списком всех
  candidates и их terminal/OOF outcomes;
- `@promoted`: только status `promoted`; обязательны `experiment_id`, immutable
  `artifact_ref`, `evidence_ref`, timestamp;
- registry transition без обновления timestamp и evidence запрещён.

Колонка `weight` всегда числовая. Смысл задаёт `weight_rule`; например
`raptor_outer_weight`, `new_branch_rank_weight`, `residual_fraction`.

## Обязательный RESULT.json

Файл должен быть валидным JSON и содержать как минимум:

```json
{
  "schema_version": 1,
  "experiment_id": "Sxx",
  "hypothesis": "",
  "only_intended_change": "",
  "parent_graph": "",
  "execution_mode": "unified_hidden_graph",
  "dependencies": [],
  "recipe": {},
  "source_refs": [{"ref": "owner/slug", "version": ""}],
  "hashes": {
    "code_config_sha256": "",
    "data_sha256": "",
    "labels_sha256": "",
    "cache_sha256": "",
    "folds_sha256": "",
    "model_sha256": "",
    "visible_output_sha256": ""
  },
  "validation": {
    "oof_macro_auc": null,
    "oof_reference": "",
    "per_target_auc": {},
    "fold_deltas": [],
    "second_seed_delta": null,
    "gold58_macro_auc": null,
    "gold58_bootstrap_interval": [],
    "oof_raw_correlations": {},
    "oof_rank_correlations": {}
  },
  "runtime": {
    "visible_minutes": null,
    "hidden_estimate_minutes": null,
    "peak_vram_gb": null
  },
  "submission": {
    "ref": "",
    "submitted_at_utc": "",
    "public_score": null,
    "rank_before": null,
    "rank_after": null,
    "leader_score": null,
    "top10_score": null,
    "gold_cutoff_rank": null,
    "gold_cutoff_score": null,
    "team_count": null
  },
  "decision": "pending",
  "blocker": "",
  "notes": []
}
```

Для inference-only экспериментов OOF-поля могут быть `null`, но это должно быть
объяснено в `notes`. Для S11+ `folds`, `model`, OOF macro, 12 per-target AUC и OOF
correlations обязательны до submit.

Hidden prediction vector и его SHA/correlation здесь отсутствуют намеренно: Kaggle
их участнику не раскрывает. `visible_output_sha256` относится только к трёхстрочному
placeholder и служит проверкой локальной воспроизводимости, не дедупликацией hidden
предсказаний. Дубли до submit определяются по code/config/recipe/model hashes.
