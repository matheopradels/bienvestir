"""
metrics.py
==========
Calcul des métriques d'évaluation pour le projet
« Prédicteur de Rentabilité d'Investissement ».

Contrat public imposé par le template
--------------------------------------
    compute_metrics(y_true, y_pred) -> dict[str, float]
        Retourne un dictionnaire NON VIDE avec des valeurs numériques.
        Les clés doivent être IDENTIQUES pour tous les modèles comparés
        (le template itère sur elles pour construire le CSV).

Métriques choisies et justification
-------------------------------------
Dans un contexte d'investissement, les erreurs ont des coûts asymétriques :
  - Faux positif  → investissement dans un projet perdant (perte en capital)
  - Faux négatif  → rejet d'un projet rentable (manque à gagner)

L'AUC-ROC mesure le pouvoir discriminant global sans dépendre d'un seuil.
Le coefficient de Matthews synthétise les 4 cases de la matrice de confusion.
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Contrat public (NE PAS renommer ni modifier la signature)
# ---------------------------------------------------------------------------

def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """
    Calcule les métriques de classification sur l'ensemble de test.

    Paramètres
    ----------
    y_true : array-like de forme (n,)
        Étiquettes réelles. 0 = non rentable, 1 = rentable.
    y_pred : array-like de forme (n,)
        Étiquettes prédites par model.predict(X_test).

    Retourne
    --------
    dict[str, float]
        Dictionnaire {nom_metrique: valeur} arrondi à 4 décimales.
        Toutes les clés sont identiques quel que soit le modèle appelant.
    """
    return {
        "exactitude":         _r(accuracy_score(y_true, y_pred)),
        "f1_pondere":         _r(f1_score(y_true, y_pred, average="weighted",  zero_division=0)),
        "precision_rentable": _r(precision_score(y_true, y_pred, pos_label=1,  zero_division=0)),
        "rappel_rentable":    _r(recall_score(y_true, y_pred,    pos_label=1,  zero_division=0)),
        "auc_roc":            _r(roc_auc_score(y_true, y_pred)),
        "precision_moyenne":  _r(average_precision_score(y_true, y_pred)),
        "coeff_matthews":     _r(matthews_corrcoef(y_true, y_pred)),
    }


def _r(v: float, d: int = 4) -> float:
    """Arrondit à `d` décimales."""
    return round(float(v), d)
