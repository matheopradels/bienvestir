"""
config.py
=========
Configuration centrale du projet « Prédicteur de Rentabilité d'Investissement ».

Ce fichier expose EXACTEMENT les variables attendues par scripts/main.py :
  - PROJECT_ROOT      : racine du projet
  - SRC_DIR           : répertoire des sources
  - APP_ENTRYPOINT    : chemin vers app.py (pour Streamlit)
  - ENV_FILE          : chemin vers le fichier .env
  - MODELS            : registre des modèles (dict avec clés "name", "path")
  - STREAMLIT_HOST    : adresse d'écoute Streamlit
  - STREAMLIT_PORT    : port d'écoute Streamlit

⚠  Ne pas renommer ces variables — elles sont importées directement
   par scripts/main.py du template.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins racine
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR      = PROJECT_ROOT / "src"
ENV_FILE     = PROJECT_ROOT / ".env"

# ---------------------------------------------------------------------------
# Point d'entrée Streamlit (attendu par le template sous APP_ENTRYPOINT)
# ---------------------------------------------------------------------------
APP_ENTRYPOINT = SRC_DIR / "app.py"

# ---------------------------------------------------------------------------
# Serveur Streamlit
# ---------------------------------------------------------------------------
STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# ---------------------------------------------------------------------------
# Sous-répertoires du projet
# ---------------------------------------------------------------------------
DATA_DIR    = PROJECT_ROOT / "data"
MODELS_DIR  = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR   = PROJECT_ROOT / "plots"
LOGS_DIR    = PROJECT_ROOT / "logs"

for _d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Registre des modèles
# Chaque entrée DOIT contenir "path" (requis par le template).
# "name" et "description" sont des métadonnées affichées dans l'app.
# ---------------------------------------------------------------------------
MODELS = {
    "gradient_boosting": {
        "name":        "Gradient Boosting",
        "description": (
            "Boosting séquentiel (sklearn). Corrige itérativement les erreurs "
            "du modèle précédent. Robuste au bruit, peu sensible aux valeurs aberrantes."
        ),
        "path": MODELS_DIR / "gradient_boosting.joblib",
    },
    "xgboost": {
        "name":        "XGBoost",
        "description": (
            "Extreme Gradient Boosting avec régularisation L1/L2. "
            "Très rapide, meilleure AUC sur l'ensemble de test."
        ),
        "path": MODELS_DIR / "xgboost.joblib",
    },
    "random_forest": {
        "name":        "Forêt Aléatoire",
        "description": (
            "Ensemble de 400 arbres entraînés en parallèle (bagging). "
            "Stable et fournit une estimation fiable de l'importance des variables."
        ),
        "path": MODELS_DIR / "random_forest.joblib",
    },
}
