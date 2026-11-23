# Prédicteur de Rentabilité d'Investissement — PoC ML

Proof-of-concept de machine learning qui prédit, **avant tout engagement
financier**, si un projet d'investissement sera rentable sur son horizon
de placement.

Projet réalisé à Albert School × Mines Paris PSL, à partir du template
[ml-poc-project](https://github.com/CiABAMBA/ml-poc-project).

---

## Contexte métier

Une mauvaise allocation du capital entraîne soit une perte directe en capital
(investir dans un projet perdant), soit un manque à gagner (rejeter un projet
rentable). Ce modèle aide à objectiver cette décision à partir de données
financières.

**Variable cible :** `rentable`
- `1` — rendement net positif sur l'horizon de placement
- `0` — perte nette

---

## Structure du dépôt

```
.
├── data/                          # Données (générées au 1er run)
│   ├── investissements_brut.csv
│   └── investissements_traite.csv
├── logs/
├── models/                        # Modèles sérialisés
│   ├── gradient_boosting.joblib
│   ├── xgboost.joblib
│   └── random_forest.joblib
├── notebooks/
│   ├── 01_exploration_donnees.ipynb
│   └── 02_modelisation_evaluation.ipynb
├── results/
│   └── model_metrics.csv
├── scripts/
│   ├── main.py      ← fichier ORIGINAL du template (ne pas modifier)
│   └── train.py     ← entraîne et sauvegarde les modèles
├── src/
│   ├── config.py    ← chemins + registre MODELS + variables Streamlit
│   ├── data.py      ← load_dataset_split()
│   ├── metrics.py   ← compute_metrics()
│   ├── model_io.py  ← load_model() / save_model()
│   ├── results.py   ← write_metrics()
│   └── app.py       ← build_app()
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Modules requis par le template

| Fichier | Fonction exposée | Rôle |
|---|---|---|
| `src/config.py` | — | Variables `PROJECT_ROOT`, `SRC_DIR`, `APP_ENTRYPOINT`, `ENV_FILE`, `MODELS`, `STREAMLIT_HOST`, `STREAMLIT_PORT` |
| `src/data.py` | `load_dataset_split()` | Retourne `(X_train, X_test, y_train, y_test)` |
| `src/metrics.py` | `compute_metrics(y_true, y_pred)` | Retourne un `dict[str, float]` non vide |
| `src/model_io.py` | `load_model(path)` | Charge un `.joblib` et retourne l'estimateur |
| `src/results.py` | `write_metrics(rows)` | Sauvegarde le CSV et retourne le DataFrame |
| `src/app.py` | `build_app()` | Point d'entrée Streamlit |

---

## Modèles comparés

| Clé | Nom | Description |
|---|---|---|
| `gradient_boosting` | Gradient Boosting | Boosting séquentiel sklearn |
| `xgboost` | XGBoost | Boosting extrême avec régularisation L1/L2 |
| `random_forest` | Forêt Aléatoire | Ensemble de 400 arbres (bagging) |

---

## Démarrage rapide

```bash
# 1. Cloner le dépôt
git clone https://github.com/matheopradels/bienvestir.git
cd bienvestir

# 2. Environnement virtuel
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate

# 3. Dépendances
pip install -r requirements.txt

# 4. Entraîner les modèles
python scripts/train.py

# 5. Évaluer + lancer l'app (script du template)
python scripts/main.py
```

L'application s'ouvre sur **http://localhost:8501**.

---

## Métriques rapportées

| Clé CSV | Nom français | Interprétation |
|---|---|---|
| `exactitude` | Exactitude | Taux global de bonne classification |
| `f1_pondere` | F1 pondéré | Équilibre précision / rappel |
| `precision_rentable` | Précision (rentable) | Qualité des alertes positives |
| `rappel_rentable` | Rappel (rentable) | Couverture des vrais positifs |
| `auc_roc` | AUC-ROC | Pouvoir discriminant sur tous les seuils |
| `precision_moyenne` | Précision moyenne | AUC de la courbe Précision-Rappel |
| `coeff_matthews` | Coeff. Matthews | Métrique synthétique équilibrée |

---

## Notes importantes

- `scripts/main.py` est le fichier **original du template** — ne pas le modifier.
- Les noms de fonctions `load_dataset_split`, `compute_metrics`, `load_model`,
  `write_metrics` et `build_app` sont imposés par le template.
- Supprimer `models/*.joblib` pour forcer un ré-entraînement.
