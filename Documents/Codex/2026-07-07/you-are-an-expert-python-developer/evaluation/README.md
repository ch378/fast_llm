# Evaluation

Dossier reserve a l'evaluation automatique des sorties JSON et CSV.

Structure prevue :

- `input/` : sorties a evaluer, organisees par provider.
- `validation/` : controles de schema, champs, source et doublons.
- `labels/` : labels automatiques ou labels de reference.
- `model/` : entrainement et prediction XGBoost.
- `metrics/` : calcul des scores.
- `reports/` : rapports JSON et CSV.

L'evaluation est independante de `extractor/` et lit uniquement le CSV fourni.

Exemples :

```powershell
python evaluation/evaluate.py --provider mastercard --input evaluation/input/mastercard
python evaluation/evaluate.py --provider cb --input evaluation/input/cb
python evaluation/evaluate.py --provider visa --input evaluation/input/visa
```

Les rapports sont ecrits dans `evaluation/reports/`.

Dans Dagster, l'asset `evaluation_report` s'execute apres `csv_exports`.
Le job complet suit donc :

```text
PDF -> Markdown -> Chunks -> LLM -> CSV -> Evaluation
                                           +-> PostgreSQL
```

Le job `csv_evaluation_only` est totalement independant de PostgreSQL.
Il lit uniquement `evaluation/input/<provider>/` et ecrit dans
`evaluation/reports/<provider>/`.
