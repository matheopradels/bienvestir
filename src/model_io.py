"""
model_io.py
===========
Chargement et sauvegarde des modèles sérialisés.

Ce module est importé par scripts/main.py via :
    load_model = model_io_module.load_model

Contrat public imposé par le template
--------------------------------------
    load_model(path: Path) -> Any
        Charge et retourne un estimateur scikit-learn depuis un fichier .joblib.
        L'objet retourné doit exposer une méthode .predict(X).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


# ---------------------------------------------------------------------------
# Contrat public (NE PAS renommer)
# ---------------------------------------------------------------------------

def load_model(path: Path) -> Any:
    """
    Charge un modèle sérialisé depuis le disque.

    Paramètres
    ----------
    path : Path
        Chemin vers le fichier .joblib contenant le modèle entraîné.

    Retourne
    --------
    Any
        Estimateur scikit-learn (ou compatible) exposant .predict() et
        optionnellement .predict_proba().

    Lève
    ----
    FileNotFoundError
        Si le fichier n'existe pas à l'emplacement indiqué.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier modèle introuvable : {path}\n"
            "Entraînez les modèles en exécutant : python scripts/train.py"
        )
    return joblib.load(path)


def save_model(model: Any, path: Path, compress: int = 3) -> None:
    """
    Sérialise un modèle entraîné sur le disque.

    Paramètres
    ----------
    model : Any
        Estimateur scikit-learn entraîné.
    path : Path
        Chemin de destination du fichier .joblib.
    compress : int
        Niveau de compression zlib (0 = aucune, 9 = maximale). Défaut : 3.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=compress)
