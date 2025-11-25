# SDM Prompt Iterator

Outil CLI pour automatiser l'itération et l'optimisation des prompts de catégorisation SDM.

## Fonctionnalités

- **Extraction de vérité terrain** depuis des jobs validés par des humains
- **Évaluation automatique** des prompts avec métriques (accuracy, recouvrement)
- **Itération assistée par Claude** pour améliorer les prompts
- **Mode auto-split** pour éviter l'overfitting (séparation train/eval)
- **Historique des runs** pour suivre l'évolution des performances

## Installation

```bash
cd ~/Programs/akeneo/tools/sdm-prompt-iterator
pip install -r requirements.txt
```

## Configuration

Créer un fichier `.env` à la racine du projet :

```bash
SDM_USER=user@akeneo.com
SDM_PASSWORD=your_password
ANTHROPIC_API_KEY=sk-ant-xxx
```

Ces variables peuvent aussi être passées en arguments CLI si besoin.

## Modes d'utilisation

### Mode Manuel

Utilise un job de test existant pour itérer sur le prompt.

```bash
# 1. Initialiser l'expérience
python main.py init \
  --experiment "category_optimization" \
  --step-id 123 \
  --field category \
  --match-keys "product_name,brand" \
  --test-job abc-123-def \
  --truth-jobs job1,job2,job3

# 2. Extraire la vérité terrain
python main.py extract-truth --experiment category_optimization

# 3. Évaluer le prompt actuel
python main.py evaluate --experiment category_optimization

# 4. Itérer avec Claude
python main.py iterate --experiment category_optimization --iterations 3
```

### Mode Auto-Split (Recommandé)

Crée automatiquement des jobs train/eval à partir des données de vérité pour éviter l'overfitting.

```bash
# 1. Initialiser avec split automatique (75% train, 25% eval)
python main.py init-auto \
  --experiment "category_v2" \
  --step-id 123 \
  --field category \
  --match-keys "product_name,brand" \
  --truth-jobs job1,job2,job3 \
  --train-ratio 0.75 \
  --seed 42

# 2. Extraire la vérité terrain
python main.py extract-truth --experiment category_v2

# 3. Itérer sur le job de train
python main.py iterate --experiment category_v2 --iterations 5

# 4. Évaluation finale sur le holdout set
python main.py final-eval --experiment category_v2
```

## Commandes

### `init`
Crée une nouvelle expérience en mode manuel.

| Option | Description |
|--------|-------------|
| `--experiment` | Nom de l'expérience |
| `--email` | Email SDM (ou `SDM_USER` env var) |
| `--password` | Mot de passe SDM (ou `SDM_PASSWORD` env var) |
| `--step-id` | ID de l'étape de classification |
| `--prev-step-id` | ID de l'étape précédente (auto-détecté si omis) |
| `--field` | Nom du champ de classification (ex: `category`) |
| `--match-keys` | Colonnes pour matcher les lignes (ex: `product_name,brand`) |
| `--test-job` | ID du job de test |
| `--truth-jobs` | IDs des jobs contenant la vérité terrain (séparés par virgule) |

### `init-auto`
Crée une expérience avec split automatique train/eval.

| Option | Description |
|--------|-------------|
| `--train-ratio` | Ratio pour le train set (défaut: 0.75) |
| `--seed` | Graine aléatoire pour le split (défaut: 42) |
| *(autres)* | Mêmes options que `init` sauf `--test-job` |

### `extract-truth`
Extrait la vérité terrain depuis les jobs validés.

```bash
python main.py extract-truth --experiment <name> --email <email> --password <pwd>
```

### `evaluate`
Évalue le prompt actuel contre la vérité terrain.

| Option | Description |
|--------|-------------|
| `--skip-rerun` | Ne pas relancer le job, utiliser les résultats actuels |
| `--use-eval-job` | Utiliser le job d'évaluation (mode auto-split) |

### `iterate`
Itère sur le prompt avec l'aide de Claude.

| Option | Description |
|--------|-------------|
| `--claude-api-key` | Clé API Anthropic (ou `ANTHROPIC_API_KEY` env var) |
| `--iterations` | Nombre d'itérations (défaut: 3) |
| `--auto-apply` | Appliquer automatiquement les suggestions |

### `final-eval`
Évaluation finale sur le holdout set (mode auto-split uniquement).

### `history`
Affiche l'historique des runs d'une expérience.

```bash
python main.py history --experiment <name>
```

### `list`
Liste toutes les expériences.

```bash
python main.py list
```

## Métriques

Le script calcule deux métriques sur trois ensembles de données :

### Métriques
- **Accuracy** : % de matchs exacts (toutes les catégories identiques)
- **Coverage (Recouvrement)** : % de lignes avec au moins une catégorie commune

### Ensembles
- **all** : Toutes les lignes
- **automated** : Lignes classifiées automatiquement (haute confiance)
- **to_check** : Lignes marquées "à vérifier" (basse confiance)

## Structure des fichiers

```
sdm-prompt-iterator/
├── main.py              # CLI principal
├── sdm_client.py        # Client API SDM
├── evaluator.py         # Calcul des métriques
├── claude_advisor.py    # Intégration Claude
├── storage.py           # Gestion des expériences
├── config.py            # Configuration
├── requirements.txt     # Dépendances
└── experiments/         # Données des expériences
    └── {experiment_name}/
        ├── config.json       # Configuration
        ├── ground_truth.json # Vérité terrain
        └── runs/             # Historique des runs
            └── {run_id}/
                ├── prompt.txt
                └── metrics.json
```

## Format de la vérité terrain

```json
{
  "extracted_at": "2024-01-15T10:00:00Z",
  "source_jobs": ["job1", "job2"],
  "field_name": "category",
  "rows": [
    {
      "match_key": {"product_name": "Widget A", "brand": "Acme"},
      "source_data": {"product_name": "Widget A", "brand": "Acme", "description": "..."},
      "categories": [
        ["Electronics", "Gadgets"],
        ["Home", "Tools"]
      ],
      "status": "validated",
      "source_job": "job1"
    }
  ]
}
```

## Workflow recommandé

1. **Collecter des jobs validés** : Avoir plusieurs jobs où un humain a validé/corrigé les classifications
2. **Utiliser le mode auto-split** : Évite l'overfitting en séparant train et eval
3. **Itérer 3-5 fois** : Laisser Claude analyser les erreurs et suggérer des améliorations
4. **Évaluer sur le holdout** : Vérifier que les améliorations généralisent bien
5. **Garder l'historique** : Comparer les performances entre les runs

## Notes

- L'URL de production SDM est utilisée par défaut : `https://sdm.akeneo.cloud`
- Le polling du job attend max 10 minutes avec intervalles de 10 secondes
- Les credentials sont lus depuis les variables d'environnement (`SDM_USER`, `SDM_PASSWORD`, `ANTHROPIC_API_KEY`) ou le fichier `.env`
- La clé API Claude est optionnelle (mode évaluation seule possible)
- Le champ `--prev-step-id` est auto-détecté si non fourni
