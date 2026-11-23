"""
data.py - Bienvestir
Chargement LendingClub (reel) avec fallback synthetique.
load_dataset_split() -> (X_train, X_test, y_train, y_test)
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from config import DATA_DIR

GRAINE    = 42
TEST_SIZE = 0.20
N_MAX_LC  = 150_000
LC_NOMS   = ["lending_club.csv", "accepted_2007_to_2018Q4.csv", "loan.csv"]
FICHIER_PROC = DATA_DIR / "investissements_traite.csv"
FICHIER_BRUT = DATA_DIR / "investissements_brut.csv"
COLONNES_LC  = [
    "loan_amnt","funded_amnt","int_rate","installment",
    "annual_inc","dti","delinq_2yrs",
    "fico_range_low","fico_range_high",
    "open_acc","pub_rec","revol_bal","revol_util","total_acc",
    "mort_acc","pub_rec_bankruptcies",
    "term","grade","home_ownership","purpose","loan_status",
]


def load_dataset_split() -> tuple:
    """
    Charge LendingClub si disponible, sinon donnees synthetiques.
    Retourne (X_train, X_test, y_train, y_test).
    """
    if FICHIER_PROC.exists():
        df = pd.read_csv(FICHIER_PROC)
        return _split(df)

    fichier_lc = next((DATA_DIR / n for n in LC_NOMS if (DATA_DIR / n).exists()), None)

    if fichier_lc:
        print(f"[data] Fichier LendingClub : {fichier_lc.name}")
        df = _charger_lending_club(fichier_lc)
        df.to_csv(FICHIER_BRUT, index=False)
    else:
        print("[data] LendingClub absent -> synthetique")
        print("[data] Vraies donnees : https://www.kaggle.com/datasets/wordsforthewise/lending-club")
        df = _generer_synthetique(n=20_000, graine=GRAINE)
        df.to_csv(FICHIER_BRUT, index=False)

    df_proc = _preparer_features(df)
    df_proc.to_csv(FICHIER_PROC, index=False)
    return _split(df_proc)


def _charger_lending_club(fichier, n_max: int = N_MAX_LC) -> pd.DataFrame:
    """Charge, filtre et nettoie le dataset LendingClub reel."""
    cols_dispo = pd.read_csv(fichier, nrows=0).columns.tolist()
    cols = [c for c in COLONNES_LC if c in cols_dispo]
    df = pd.read_csv(fichier, usecols=cols, low_memory=False)
    print(f"[data] Lignes brutes : {len(df):,}")

    statuts_ok  = {"Fully Paid"}
    statuts_def = {"Charged Off","Default",
                   "Does not meet the credit policy. Status:Charged Off"}
    mask = df["loan_status"].isin(statuts_ok | statuts_def)
    df   = df[mask].copy()
    df["rentable"] = df["loan_status"].isin(statuts_ok).astype(int)
    df.drop(columns=["loan_status"], inplace=True)
    print(f"[data] Apres filtrage : {len(df):,} | Remb: {df['rentable'].mean():.1%}")

    if len(df) > n_max:
        df = (df.groupby("rentable", group_keys=False)
                .apply(lambda x: x.sample(n_max // 2, random_state=GRAINE))
                .reset_index(drop=True))

    for col in ["int_rate","revol_util"]:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.replace("%","",regex=False)
                              .pipe(pd.to_numeric, errors="coerce"))

    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.extract(r"(\d+)")[0].astype(float)

    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
        df.drop(columns=["fico_range_low","fico_range_high"], inplace=True, errors="ignore")

    df.rename(columns={
        "loan_amnt":"montant_investi","int_rate":"taux_interet",
        "annual_inc":"revenu_annuel","dti":"dti_ratio",
        "delinq_2yrs":"incidents_2ans","open_acc":"comptes_ouverts",
        "revol_util":"utilisation_revolving","term":"duree_mois",
    }, inplace=True)

    df.dropna(thresh=int(len(df.columns)*0.7), inplace=True)
    print(f"[data] Final : {len(df):,} lignes")
    return df


def _generer_synthetique(n: int, graine: int) -> pd.DataFrame:
    """Donnees synthetiques inspirees de LendingClub (fallback)."""
    rng = np.random.default_rng(graine)
    montant_investi       = np.exp(rng.normal(10.8,1.1,n)).clip(5000,10_000_000).astype(int)
    taux_interet          = rng.normal(13.5,5.0,n).clip(5.0,30.0).round(2)
    revenu_annuel         = np.exp(rng.normal(11.0,0.7,n)).clip(20000,500000).astype(int)
    dti_ratio             = rng.uniform(0,40,n).round(2)
    incidents_2ans        = rng.integers(0,5,n)
    fico_score            = rng.normal(700,55,n).clip(580,850).astype(int)
    comptes_ouverts       = rng.integers(3,30,n)
    utilisation_revolving = rng.uniform(0,100,n).round(1)
    duree_mois            = rng.choice([36,60],n)
    grade      = rng.choice(["A","B","C","D","E","F","G"],n,p=[0.20,0.25,0.22,0.15,0.10,0.05,0.03])
    home_own   = rng.choice(["RENT","OWN","MORTGAGE","OTHER"],n,p=[0.40,0.10,0.45,0.05])
    purpose    = rng.choice(
        ["debt_consolidation","credit_card","home_improvement","other","small_business"],
        n, p=[0.45,0.25,0.12,0.12,0.06]
    )
    grade_num = np.array([{"A":7,"B":6,"C":5,"D":4,"E":3,"F":2,"G":1}[g] for g in grade])
    log_odds  = (
        -0.5 - 0.15*taux_interet - 0.04*dti_ratio
        + 0.01*(fico_score-700) - 0.30*(incidents_2ans>0).astype(float)
        + 0.15*grade_num + 0.003*(revenu_annuel/1000)
        - 0.30*(purpose=="small_business").astype(float)
        + rng.normal(0,0.5,n)
    )
    rentable = (rng.random(n) < 1/(1+np.exp(-log_odds))).astype(int)
    return pd.DataFrame({
        "montant_investi":montant_investi,"taux_interet":taux_interet,
        "revenu_annuel":revenu_annuel,"dti_ratio":dti_ratio,
        "incidents_2ans":incidents_2ans,"fico_score":fico_score,
        "comptes_ouverts":comptes_ouverts,"utilisation_revolving":utilisation_revolving,
        "duree_mois":duree_mois,"grade":grade,"home_ownership":home_own,
        "purpose":purpose,"rentable":rentable,
    })


def _preparer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables derivees et encodage ordinal."""
    df = df.copy()
    if "montant_investi" in df.columns and "revenu_annuel" in df.columns:
        df["ratio_pret_revenu"] = (df["montant_investi"]/(df["revenu_annuel"]+1)).round(4)
    if "taux_interet" in df.columns:
        df["log_taux"] = np.log1p(df["taux_interet"]).round(4)
    if "fico_score" in df.columns:
        df["fico_normalise"] = ((df["fico_score"]-580)/(850-580)).round(4)
    if "montant_investi" in df.columns:
        df["log_montant"] = np.log1p(df["montant_investi"]).round(4)
    if "dti_ratio" in df.columns and "utilisation_revolving" in df.columns:
        df["score_endettement"] = (
            df["dti_ratio"]*0.6 + df["utilisation_revolving"].fillna(0)*0.4/100
        ).round(4)
    cat_cols = [c for c in ["grade","home_ownership","purpose"] if c in df.columns]
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,dtype=int)
        df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))
    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def _split(df: pd.DataFrame) -> tuple:
    X = df.drop(columns=["rentable"])
    y = df["rentable"]
    return tuple(train_test_split(X, y, test_size=TEST_SIZE, random_state=GRAINE, stratify=y))
