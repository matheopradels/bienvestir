"""
results.py
==========
Persistance des métriques d'évaluation.

Ce module est importé par scripts/main.py via :
    write_metrics = results_module.write_metrics

Contrat public imposé par le template
--------------------------------------
    write_metrics(rows: list[dict]) -> pd.DataFrame
        Reçoit une liste de dictionnaires (un par modèle) contenant
        les clés model_key, model_name, model_path et les métriques,
        les sauvegarde dans results/model_metrics.csv et retourne
        le DataFrame correspondant.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import RESULTS_DIR

# Nom du fichier de sortie attendu par le template
FICHIER_METRIQUES = RESULTS_DIR / "model_metrics.csv"


# ---------------------------------------------------------------------------
# Contrat public (NE PAS renommer)
# ---------------------------------------------------------------------------

def write_metrics(rows: list[dict]) -> pd.DataFrame:
    """
    Sauvegarde les métriques de tous les modèles dans un fichier CSV.

    Paramètres
    ----------
    rows : list[dict]
        Liste de dictionnaires, un par modèle. Chaque dictionnaire contient
        au minimum les clés « model_key », « model_name », « model_path »
        ainsi que les métriques numériques retournées par compute_metrics().

    Retourne
    --------
    pd.DataFrame
        DataFrame avec une ligne par modèle et une colonne par métrique.
        Également sauvegardé dans results/model_metrics.csv.
    """
    if not rows:
        raise ValueError("La liste `rows` est vide — aucune métrique à sauvegarder.")

    df = pd.DataFrame(rows)

    # Colonnes de métadonnées en tête, métriques ensuite
    cols_meta     = ["model_key", "model_name", "model_path"]
    cols_metriques = [c for c in df.columns if c not in cols_meta]
    df = df[cols_meta + cols_metriques]

    FICHIER_METRIQUES.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FICHIER_METRIQUES, index=False, encoding="utf-8")

    return df
