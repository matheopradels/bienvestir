"""
train.py
========
Entraînement et sauvegarde des trois modèles du projet.

Ce script est SÉPARÉ de scripts/main.py (qui appartient au template
et ne doit pas être modifié). Il doit être exécuté UNE FOIS avant
de lancer main.py :

    python scripts/train.py   # entraîne et sauvegarde les modèles
    python scripts/main.py    # évalue et lance l'app

Les modèles entraînés sont sauvegardés dans models/ au format .joblib.
Si un fichier existe déjà, l'entraînement est ignoré pour ce modèle.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False
    print("⚠  XGBoost non disponible. Installez-le : pip install xgboost")

from config import MODELS
from data import load_dataset_split
from model_io import save_model

SEP = "=" * 60


def main() -> None:
    print(f"\n{SEP}")
    print("  Entraînement des modèles — Prédicteur de Rentabilité")
    print(f"{SEP}\n")

    print("[1/2] Chargement des données …")
    t0 = time.perf_counter()
    X_train, _, y_train, _ = load_dataset_split()
    print(f"      Entraînement : {len(X_train):,} échantillons ({time.perf_counter()-t0:.2f}s)")
    print(f"      Taux rentable : {y_train.mean():.1%}\n")

    ratio = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)

    estimateurs = {
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, min_samples_leaf=20, max_features="sqrt",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=16, min_samples_leaf=10,
            class_weight="balanced", max_features="sqrt",
            n_jobs=-1, random_state=42,
        ),
    }
    if XGBOOST_OK:
        estimateurs["xgboost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=ratio, eval_metric="logloss",
            random_state=42, verbosity=0,
        )

    print("[2/2] Entraînement …")
    for cle, meta in MODELS.items():
        if meta["path"].exists():
            print(f"  ✅ {meta['name']} — modèle existant, ignoré.")
            continue
        if cle not in estimateurs:
            print(f"  ⚠  {meta['name']} — estimateur non défini, ignoré.")
            continue
        t1 = time.perf_counter()
        estimateurs[cle].fit(X_train, y_train)
        save_model(estimateurs[cle], meta["path"])
        print(f"  ✅ {meta['name']} entraîné et sauvegardé ({time.perf_counter()-t1:.1f}s)")

    print(f"\n{SEP}")
    print("  Entraînement terminé. Lancez maintenant : python scripts/main.py")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
