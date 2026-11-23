"""
app.py — Bienvestir
Portfolio tracker avec vraies données de marché.
Design aéré, modern, boutons natifs Streamlit (pas de HTML caché).
"""

from __future__ import annotations
from datetime import date
from collections import defaultdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False

import requests

from config import DATA_DIR, MODELS, RESULTS_DIR

# ═══════════════════════════════════════════════════════════════════════════
# CATALOGUE D'ACTIFS
# ═══════════════════════════════════════════════════════════════════════════

ACTIFS = {
    "crypto": {
        "BTC-USD":  {"nom": "Bitcoin",    "symbole": "BTC"},
        "ETH-USD":  {"nom": "Ethereum",   "symbole": "ETH"},
        "SOL-USD":  {"nom": "Solana",     "symbole": "SOL"},
        "BNB-USD":  {"nom": "BNB",        "symbole": "BNB"},
        "XRP-USD":  {"nom": "XRP",        "symbole": "XRP"},
        "DOGE-USD": {"nom": "Dogecoin",   "symbole": "DOGE"},
    },
    "actions": {
        "AAPL":   {"nom": "Apple",         "symbole": "AAPL"},
        "MSFT":   {"nom": "Microsoft",     "symbole": "MSFT"},
        "NVDA":   {"nom": "Nvidia",        "symbole": "NVDA"},
        "TSLA":   {"nom": "Tesla",         "symbole": "TSLA"},
        "META":   {"nom": "Meta",          "symbole": "META"},
        "GOOGL":  {"nom": "Alphabet",      "symbole": "GOOGL"},
        "AMZN":   {"nom": "Amazon",        "symbole": "AMZN"},
        "MC.PA":  {"nom": "LVMH",          "symbole": "MC"},
        "AIR.PA": {"nom": "Airbus",        "symbole": "AIR"},
        "TTE.PA": {"nom": "TotalEnergies", "symbole": "TTE"},
        "SAN.PA": {"nom": "Sanofi",        "symbole": "SAN"},
        "BNP.PA": {"nom": "BNP Paribas",   "symbole": "BNP"},
    },
    "etf": {
        "SPY":     {"nom": "S&P 500 ETF",    "symbole": "SPY"},
        "QQQ":     {"nom": "Nasdaq 100 ETF", "symbole": "QQQ"},
        "IWDA.AS": {"nom": "MSCI World ETF", "symbole": "IWDA"},
        "GLD":     {"nom": "Gold ETF",       "symbole": "GLD"},
        "VT":      {"nom": "Vanguard Total", "symbole": "VT"},
    },
    "obligations": {
        "TLT": {"nom": "US Treasury 20Y", "symbole": "TLT"},
        "AGG": {"nom": "US Bond ETF",     "symbole": "AGG"},
        "BND": {"nom": "Vanguard Bonds",  "symbole": "BND"},
    },
}

CAT_META = {
    "crypto":      {"label": "Crypto",      "emoji": "🪙", "color": "#F59E0B"},
    "actions":     {"label": "Actions",     "emoji": "📈", "color": "#3B82F6"},
    "etf":         {"label": "ETF / Fonds", "emoji": "🌍", "color": "#10B981"},
    "obligations": {"label": "Obligations", "emoji": "🏦", "color": "#8B5CF6"},
    "immobilier":  {"label": "Immobilier",  "emoji": "🏠", "color": "#EC4899"},
    "autre":       {"label": "Autre",       "emoji": "🎨", "color": "#64748B"},
}

# ML encodings
ENC_SECTEUR  = {"agroalimentaire":0,"bourse":1,"energie":2,"immobilier":3,"infrastructure":4,"sante":5,"technologie":6}
ENC_ZONE     = {"Amerique_Nord":0,"Asie_Pacifique":1,"Europe_Est":2,"Europe_Ouest":3,"France":4,"Marches_Emergents":5}
ENC_ACTIF_ML = {"actions":0,"fonds_indiciels":1,"immobilier_direct":2,"matieres_premieres":3,"obligations":4,"private_equity":5}
ENC_NOTATION = {"A":0,"AA":1,"AAA":2,"B":3,"BB":4,"BBB":5,"CCC":6}
RANG_NOTATION= {"AAA":7,"AA":6,"A":5,"BBB":4,"BB":3,"B":2,"CCC":1}
TYPE_MAP     = {
    "📈 Actions / Bourse":         ("actions","bourse"),
    "🏠 Immobilier":               ("immobilier_direct","immobilier"),
    "🪙 Crypto":                   ("actions","technologie"),
    "🏦 Obligations":              ("obligations","bourse"),
    "💡 Startup / Private equity": ("private_equity","technologie"),
    "🌍 ETF / Fonds":              ("fonds_indiciels","bourse"),
    "🎨 Autre":                    ("actions","bourse"),
}
RISQUE_MAP   = {"😴 Très faible":1.0,"🙂 Faible":3.0,"😐 Modéré":5.0,"😬 Élevé":7.5,"🔥 Très élevé":9.5}

PLOTLY_BASE  = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94A3B8"),
)


# ═══════════════════════════════════════════════════════════════════════════
# DONNÉES DE MARCHÉ
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=180)
def fetch_quote(ticker: str) -> dict:
    """
    Récupère le prix via yfinance (principal) ou Yahoo Finance API (fallback).
    Retourne {"price": float|None, "change_pct": float|None, "error": str|None}
    """
    # Tentative yfinance
    if YFINANCE_OK:
        try:
            t     = yf.Ticker(ticker)
            fi    = t.fast_info
            price = fi.last_price
            prev  = fi.previous_close
            chg   = (price - prev) / prev * 100 if prev and prev > 0 else 0.0
            if price and price > 0:
                return {"price": price, "change_pct": chg, "error": None}
        except Exception:
            pass

    # Fallback : Yahoo Finance direct
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=2d"
        r   = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        if r.status_code == 200:
            meta  = r.json()["chart"]["result"][0]["meta"]
            price = meta.get("regularMarketPrice")
            prev  = meta.get("chartPreviousClose") or meta.get("previousClose")
            chg   = (price - prev) / prev * 100 if price and prev and prev > 0 else 0.0
            return {"price": price, "change_pct": chg, "error": None}
    except Exception:
        pass

    return {"price": None, "change_pct": None, "error": "Données indisponibles"}


@st.cache_data(ttl=600)
def fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    if YFINANCE_OK:
        try:
            df = yf.Ticker(ticker).history(period=period)
            return df[["Close"]].rename(columns={"Close": "prix"}) if not df.empty else None
        except Exception:
            pass
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range={period}"
        r   = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if r.status_code == 200:
            res       = r.json()["chart"]["result"][0]
            timestamps= res["timestamp"]
            closes    = res["indicators"]["quote"][0]["close"]
            df        = pd.DataFrame({"prix": closes}, index=pd.to_datetime(timestamps, unit="s"))
            return df.dropna()
    except Exception:
        pass
    return None


def enrichir(p: dict) -> dict:
    out = dict(p)
    ticker = p.get("ticker")
    if ticker:
        q = fetch_quote(ticker)
        px_now = q["price"]
        if px_now and p.get("prix_achat") and p.get("quantite"):
            val_now  = p["quantite"] * px_now
            val_ach  = p["quantite"] * p["prix_achat"]
            gain_e   = val_now - val_ach
            gain_pct = gain_e / val_ach * 100 if val_ach > 0 else 0
            out.update({"prix_now": px_now, "val_now": val_now, "val_ach": val_ach,
                        "gain_e": gain_e, "gain_pct": gain_pct,
                        "chg_jour": q["change_pct"], "live": True})
            return out
    # Fallback estimation
    m   = p.get("montant", 0)
    r   = p.get("rendement", 0)
    d   = max(p.get("duree", 1), 1)
    vn  = m * (1 + r / 100) ** d
    out.update({"prix_now": None, "val_now": vn, "val_ach": m,
                "gain_e": vn - m, "gain_pct": (vn - m) / m * 100 if m > 0 else 0,
                "chg_jour": None, "live": False})
    return out


# ═══════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════

def build_app() -> None:
    st.set_page_config(page_title="Bienvestir", page_icon="💎", layout="wide")
    _inject_css()
    _sidebar_conseiller()

    # Header
    st.markdown("""
    <div style="padding:2rem 0 1.5rem;border-bottom:1px solid #1E293B;margin-bottom:2rem">
        <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
                    background:linear-gradient(135deg,#6EE7F7,#818CF8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    letter-spacing:-.03em">
            💎 Bienvestir
        </div>
        <div style="font-size:.8rem;color:#475569;margin-top:.3rem;letter-spacing:.1em;text-transform:uppercase">
            Suivez · Analysez · Anticipez
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["💼   Mon Portfolio", "🔮   Simulateur ML"])
    with tab1: _tab_portfolio()
    with tab2: _tab_simulateur()


# ═══════════════════════════════════════════════════════════════════════════
# TAB PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════

def _tab_portfolio() -> None:
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []

    pf = [enrichir(p) for p in st.session_state.portfolio]

    # ── KPIs ──────────────────────────────────────────────────────────────
    total_ach = sum(p["val_ach"]  for p in pf)
    total_now = sum(p["val_now"]  for p in pf)
    gain_tot  = total_now - total_ach
    perf_pct  = gain_tot / total_ach * 100 if total_ach > 0 else 0
    sg        = "+" if gain_tot >= 0 else ""

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(_kpi_html("Capital investi", f"{total_ach:,.2f} €", "#6EE7F7", "💶"), unsafe_allow_html=True)
    with k2:
        st.markdown(_kpi_html("Valeur actuelle", f"{total_now:,.2f} €", "#818CF8", "📊"), unsafe_allow_html=True)
    with k3:
        c = "#34D399" if gain_tot >= 0 else "#F87171"
        st.markdown(_kpi_html("Gain / Perte", f"{sg}{gain_tot:,.2f} €", c, "💰"), unsafe_allow_html=True)
    with k4:
        c = "#34D399" if perf_pct >= 0 else "#F87171"
        st.markdown(_kpi_html("Performance totale", f"{sg}{perf_pct:.2f}%", c, "⚡"), unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    # ── Contenu ────────────────────────────────────────────────────────────
    if pf:
        col_g, col_d = st.columns([1, 1], gap="large")

        with col_g:
            st.markdown("#### Répartition du portfolio")
            _donut(pf)

        with col_d:
            st.markdown("#### Gain / Perte par position")
            _barres_positions(pf)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("#### Mes positions")
        _liste_positions(pf)

        # Courbe historique
        groupes_live = [g for g in _grouper_positions(pf) if g.get("live") and g.get("ticker")]
        if groupes_live:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown("#### Historique de prix")
            opts = {f"{g['nom']} ({g['ticker']})": g["ticker"] for g in groupes_live}
            if len(opts) > 1:
                sel_label = st.selectbox("Actif", list(opts.keys()), label_visibility="collapsed")
            else:
                sel_label = list(opts.keys())[0]
            _courbe_historique(opts[sel_label], sel_label)
    else:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;border:1px dashed #1E3A5F;
                    border-radius:20px;background:#080F1E88">
            <div style="font-size:3rem;margin-bottom:.8rem">📭</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.2rem;
                        font-weight:700;color:#F1F5F9;margin-bottom:.4rem">
                Votre portfolio est vide
            </div>
            <div style="color:#475569;font-size:.9rem">
                Ajoutez votre premier investissement ci-dessous
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.divider()

    # ── Formulaire d'ajout ─────────────────────────────────────────────────
    st.markdown("### ➕  Ajouter une position")
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    mode = st.radio("", ["🔍 Actif coté (temps réel)", "✏️ Investissement manuel"],
                    horizontal=True, label_visibility="collapsed")

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

    if mode == "🔍 Actif coté (temps réel)":
        _form_catalogue()
    else:
        _form_manuel()


def _grouper_positions(pf: list) -> list:
    """
    Regroupe les positions par ticker (ou par nom si pas de ticker).
    Calcule le prix d'achat moyen pondéré et les totaux agrégés.
    """
    groupes: dict = {}
    for p in pf:
        cle = p.get("ticker") or p.get("nom", "autre")
        if cle not in groupes:
            groupes[cle] = {
                "nom":        p.get("nom", cle),
                "ticker":     p.get("ticker"),
                "categorie":  p.get("categorie", "autre"),
                "positions":  [],
                "total_investi": 0.0,
                "total_quantite": 0.0,
                "val_now":    0.0,
                "gain_e":     0.0,
                "prix_now":   p.get("prix_now"),
                "chg_jour":   p.get("chg_jour"),
                "live":       p.get("live", False),
            }
        g = groupes[cle]
        g["positions"].append(p)
        g["total_investi"]   += p.get("val_ach", p.get("montant", 0))
        g["total_quantite"]  += p.get("quantite", 0) or 0
        g["val_now"]         += p.get("val_now", 0)
        g["gain_e"]          += p.get("gain_e", 0)
        if p.get("prix_now"):
            g["prix_now"] = p["prix_now"]
        if p.get("chg_jour") is not None:
            g["chg_jour"] = p["chg_jour"]
        if p.get("live"):
            g["live"] = True

    result = []
    for g in groupes.values():
        g["gain_pct"] = g["gain_e"] / g["total_investi"] * 100 if g["total_investi"] > 0 else 0
        # Prix moyen pondéré
        if g["total_quantite"] > 0 and g["total_investi"] > 0:
            g["prix_achat_moyen"] = g["total_investi"] / g["total_quantite"]
        else:
            g["prix_achat_moyen"] = None
        g["nb_positions"] = len(g["positions"])
        result.append(g)
    return result


def _liste_positions(pf: list) -> None:
    groupes = _grouper_positions(pf)

    for i, g in enumerate(groupes):
        cat    = CAT_META.get(g.get("categorie","autre"), CAT_META["autre"])
        gc     = "#34D399" if g["gain_e"] >= 0 else "#F87171"
        sg     = "+" if g["gain_e"] >= 0 else ""
        nb     = g["nb_positions"]
        badge_nb = f'<span style="background:#1E3A5F;color:#94A3B8;font-size:.65rem;padding:.1rem .4rem;border-radius:99px;margin-left:.3rem">{nb}×</span>' if nb > 1 else ""

        live_txt = ""
        if g.get("live") and g.get("chg_jour") is not None:
            cj = g["chg_jour"]
            cc = "#34D399" if cj >= 0 else "#F87171"
            live_txt = f'<span style="font-size:.68rem;color:{cc};background:{cc}18;padding:.12rem .4rem;border-radius:99px;border:1px solid {cc}33">{"+" if cj>=0 else ""}{cj:.2f}% auj.</span>'

        # Infos détail
        detail_parts = [cat["label"]]
        if g.get("prix_achat_moyen"):
            detail_parts.append(f"px moy. achat : {g['prix_achat_moyen']:,.2f}")
        if g.get("prix_now"):
            detail_parts.append(f"px actuel : {g['prix_now']:,.2f}")
        if g.get("total_quantite") and g["total_quantite"] > 0:
            detail_parts.append(f"{g['total_quantite']:.6g} unités")
        detail_txt = " · ".join(detail_parts)

        row_l, row_r, row_del = st.columns([6, 3, 1])
        with row_l:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:.9rem;
                        padding:1rem 1.2rem;background:#0B1628;
                        border:1px solid #0F2040;border-radius:14px">
                <div style="width:2.4rem;height:2.4rem;border-radius:10px;flex-shrink:0;
                            background:{cat["color"]}18;border:1px solid {cat["color"]}30;
                            display:flex;align-items:center;justify-content:center;font-size:1.1rem">
                    {cat["emoji"]}
                </div>
                <div style="flex:1;min-width:0">
                    <div style="font-weight:600;font-size:.95rem;color:#F1F5F9;
                                display:flex;align-items:center;gap:.4rem;flex-wrap:wrap">
                        {g["nom"]}{badge_nb} {live_txt}
                    </div>
                    <div style="font-size:.7rem;color:#475569;margin-top:.15rem;
                                white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                        {detail_txt}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with row_r:
            st.markdown(f"""
            <div style="padding:1rem 1.2rem;background:#0B1628;
                        border:1px solid #0F2040;border-radius:14px;text-align:right;height:100%;
                        display:flex;flex-direction:column;justify-content:center">
                <div style="font-family:'Syne',sans-serif;font-size:1rem;
                            font-weight:700;color:#F1F5F9">{g["val_now"]:,.2f} €</div>
                <div style="font-size:.75rem;color:{gc};margin-top:.15rem">
                    {sg}{g["gain_e"]:,.2f} € ({sg}{g["gain_pct"]:.2f}%)
                </div>
                <div style="font-size:.68rem;color:#334155;margin-top:.1rem">
                    investi : {g["total_investi"]:,.2f} €
                </div>
            </div>
            """, unsafe_allow_html=True)
        with row_del:
            st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)
            if st.button("🗑️", key=f"del_grp_{i}", use_container_width=True,
                         help=f"Supprimer toutes les positions {g['nom']}"):
                # Supprimer toutes les positions de ce groupe
                tickers_to_del = {p.get("ticker") or p.get("nom") for p in g["positions"]}
                st.session_state.portfolio = [
                    p for p in st.session_state.portfolio
                    if (p.get("ticker") or p.get("nom")) not in tickers_to_del
                ]
                st.rerun()
        st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)


def _donut(pf: list) -> None:
    totaux: dict = defaultdict(float)
    for p in pf:
        cat = CAT_META.get(p.get("categorie","autre"), CAT_META["autre"])
        totaux[cat["label"]] += p["val_now"]

    labels = list(totaux.keys())
    values = list(totaux.values())
    colors = [next((v["color"] for v in CAT_META.values() if v["label"]==l), "#64748B") for l in labels]
    total  = sum(values)

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=.62,
        marker=dict(colors=colors, line=dict(color="#060D1A", width=4)),
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>%{value:,.2f} €  ·  %{percent}<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{total:,.0f} €</b>", x=.5, y=.56,
                       font=dict(size=16, color="#F1F5F9", family="Syne"), showarrow=False)
    fig.add_annotation(text="valeur totale", x=.5, y=.44,
                       font=dict(size=12, color="#475569"), showarrow=False)
    fig.update_layout(**PLOTLY_BASE, height=300, showlegend=False,
                      margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)


def _barres_positions(pf: list) -> None:
    noms   = [p.get("nom","") for p in pf]
    gains  = [p["gain_e"] for p in pf]
    colors = ["#34D399" if g >= 0 else "#F87171" for g in gains]

    fig = go.Figure(go.Bar(
        y=noms, x=gains, orientation="h",
        marker=dict(color=colors, opacity=.85, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>%{x:+,.2f} €<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#1E293B", width=1.5))
    fig.update_layout(**PLOTLY_BASE, height=max(140, len(pf)*50),
                      xaxis=dict(tickformat="+,.0f", gridcolor="#0F172A", zeroline=False),
                      yaxis=dict(gridcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)


def _courbe_historique(ticker: str, label: str) -> None:
    """
    Graphique interactif avec sélection de temporalité.
    Inclut bougies, volume, range slider, et outils de zoom.
    """
    # Sélecteur de période
    periodes = {
        "5j": "5d", "1m": "1mo", "3m": "3mo",
        "6m": "6mo", "1a": "1y", "2a": "2y", "5a": "5y",
    }
    col_p = st.columns(len(periodes))
    if "periode_sel" not in st.session_state:
        st.session_state.periode_sel = "1a"

    for col, (label_p, val_p) in zip(col_p, periodes.items()):
        is_sel = st.session_state.periode_sel == val_p
        if col.button(
            label_p,
            key=f"per_{val_p}_{ticker}",
            type="primary" if is_sel else "secondary",
            use_container_width=True,
        ):
            st.session_state.periode_sel = val_p
            st.rerun()

    periode = st.session_state.periode_sel
    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

    with st.spinner("Chargement des données…"):
        hist = fetch_history(ticker, periode)

    if hist is None or hist.empty:
        st.caption("Historique non disponible pour cette période.")
        return

    prices   = hist["prix"]
    variation = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
    line_col  = "#34D399" if variation >= 0 else "#F87171"
    fill_col  = "rgba(52,211,153,0.06)" if variation >= 0 else "rgba(248,113,113,0.06)"

    # Stats rapides
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Actuel",     f"{prices.iloc[-1]:,.2f}")
    sc2.metric("Var. période", f"{variation:+.2f}%")
    sc3.metric("Plus haut",  f"{prices.max():,.2f}")
    sc4.metric("Plus bas",   f"{prices.min():,.2f}")

    st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)

    # Graphique principal avec range slider
    fig = go.Figure()

    # Aire + ligne principale
    fig.add_trace(go.Scatter(
        x=hist.index, y=prices,
        name=label,
        line=dict(color=line_col, width=2),
        fill="tozeroy",
        fillcolor=fill_col,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Prix : <b>%{y:,.2f}</b><extra></extra>",
    ))

    # Moyenne mobile 20j
    if len(prices) >= 20:
        ma20 = prices.rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=hist.index, y=ma20,
            name="MM20",
            line=dict(color="#818CF8", width=1.2, dash="dot"),
            hovertemplate="MM20: <b>%{y:,.2f}</b><extra></extra>",
            opacity=0.7,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#94A3B8"),
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            bgcolor="rgba(11,22,40,0.8)",
            bordercolor="#1E3A5F",
            borderwidth=1,
            x=0, y=1,
            orientation="h",
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0B1628", bordercolor="#1E3A5F", font_color="#F1F5F9"),
        xaxis=dict(
            gridcolor="#0F172A",
            showspikes=True,
            spikecolor="#334155",
            spikethickness=1,
            spikedash="dot",
            rangeslider=dict(
                visible=True,
                bgcolor="#060D1A",
                bordercolor="#0F2040",
                thickness=0.06,
            ),
        ),
        yaxis=dict(
            gridcolor="#0F172A",
            tickformat=",.2f",
            showspikes=True,
            spikecolor="#334155",
            spikethickness=1,
        ),
        dragmode="zoom",
    )

    # Boutons de zoom rapide intégrés dans Plotly
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=7,  label="7j",  step="day",   stepmode="backward"),
                dict(count=1,  label="1m",  step="month", stepmode="backward"),
                dict(count=3,  label="3m",  step="month", stepmode="backward"),
                dict(count=6,  label="6m",  step="month", stepmode="backward"),
                dict(count=1,  label="1a",  step="year",  stepmode="backward"),
                dict(step="all", label="Tout"),
            ],
            bgcolor="#0B1628",
            activecolor="#4F46E5",
            bordercolor="#1E3A5F",
            font=dict(color="#94A3B8", size=11),
            x=0, y=1.0,
        )
    )

    st.plotly_chart(fig, use_container_width=True, config={
        "scrollZoom":       True,
        "displayModeBar":   True,
        "modeBarButtonsToRemove": ["autoScale2d", "lasso2d", "select2d"],
        "displaylogo":      False,
        "toImageButtonOptions": {"format": "png", "filename": f"bienvestir_{ticker}"},
    })



# ── Formulaire catalogue ───────────────────────────────────────────────────

def _form_catalogue() -> None:
    if "cat_sel" not in st.session_state:
        st.session_state.cat_sel = "crypto"

    # Boutons catégorie — natifs Streamlit
    cats = list(ACTIFS.keys())
    cols = st.columns(len(cats))
    for col, cat_key in zip(cols, cats):
        meta    = CAT_META[cat_key]
        is_sel  = st.session_state.cat_sel == cat_key
        label   = f"{'✅ ' if is_sel else ''}{meta['emoji']} {meta['label']}"
        if col.button(label, key=f"btn_cat_{cat_key}", use_container_width=True,
                      type="primary" if is_sel else "secondary"):
            st.session_state.cat_sel = cat_key
            st.rerun()

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

    # Sélection actif + formulaire
    actifs_cat = ACTIFS[st.session_state.cat_sel]
    options    = {f"{m['nom']}  ({t})": t for t, m in actifs_cat.items()}

    with st.form("form_cat", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        chosen_lbl    = c1.selectbox("Actif", list(options.keys()))
        ticker        = options[chosen_lbl]
        montant_eur   = c2.number_input("Montant investi (€)", 1.0, 10_000_000.0, 100.0, step=10.0)
        date_achat    = c3.date_input("Date d'achat", value=date.today())

        submitted = st.form_submit_button("➕  Ajouter au portfolio", type="primary", use_container_width=True)
        if submitted:
            with st.spinner("Récupération du prix…"):
                q = fetch_quote(ticker)

            if q["price"] is None:
                st.error(f"⚠️  Prix indisponible pour **{ticker}**. "
                         "Vérifiez votre connexion internet ou réessayez dans quelques secondes.")
            else:
                px      = q["price"]
                qty     = montant_eur / px
                meta_a  = actifs_cat[ticker]
                st.session_state.portfolio.append({
                    "nom":        meta_a["nom"],
                    "ticker":     ticker,
                    "categorie":  st.session_state.cat_sel,
                    "quantite":   round(qty, 8),
                    "prix_achat": px,
                    "montant":    montant_eur,
                    "rendement":  0,
                    "duree":      1,
                    "date":       date_achat.strftime("%d/%m/%Y"),
                })
                st.success(f"✅  **{meta_a['nom']}** ajouté — {qty:.6g} unités à {px:,.2f} €/unité")
                st.rerun()

    # Prix en live
    with st.expander("📊  Voir les prix actuels"):
        rows = []
        for t, m in actifs_cat.items():
            q = fetch_quote(t)
            rows.append({
                "Actif":    m["nom"],
                "Ticker":   t,
                "Prix":     f"{q['price']:,.2f} €" if q["price"] else "—",
                "Var. 24h": f"{q['change_pct']:+.2f}%" if q["change_pct"] is not None else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _form_manuel() -> None:
    cat_opts = {
        "🏠 Immobilier":    "immobilier",
        "🍽️ Restauration":  "autre",
        "💡 Startup":       "autre",
        "🎨 Autre":         "autre",
    }
    with st.form("form_manuel", clear_on_submit=True):
        c1, c2 = st.columns(2)
        nom       = c1.text_input("Nom", placeholder="Ex: Appart Paris, Restaurant Le Bon…")
        categorie = c2.selectbox("Catégorie", list(cat_opts.keys()))
        c3, c4, c5 = st.columns(3)
        montant   = c3.number_input("Montant investi (€)", 100.0, 10_000_000.0, 10_000.0, step=100.0)
        rendement = c4.number_input("Rendement annuel estimé (%)", -50.0, 200.0, 5.0, step=0.5)
        duree     = c5.number_input("Durée (années)", 1, 30, 5)

        ok = st.form_submit_button("➕  Ajouter", type="primary", use_container_width=True)
        if ok:
            if not nom.strip():
                st.error("Donnez un nom à cet investissement.")
            else:
                st.session_state.portfolio.append({
                    "nom": nom, "ticker": None,
                    "categorie": cat_opts[categorie],
                    "quantite": None, "prix_achat": None,
                    "montant": montant, "rendement": rendement, "duree": duree,
                    "date": date.today().strftime("%d/%m/%Y"),
                })
                st.success(f"✅  **{nom}** ajouté !")
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB SIMULATEUR
# ═══════════════════════════════════════════════════════════════════════════

def _tab_simulateur() -> None:
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    st.markdown("### Analysez la rentabilité de votre projet")
    st.markdown("<p style='color:#475569;font-size:.9rem;margin-bottom:1.5rem'>Nos 3 modèles ML donnent chacun un verdict indépendant.</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        montant      = st.number_input("💶  Combien investissez-vous ? (€)", 500, 10_000_000, 10_000, step=500)
        rendement    = st.slider("📈  Rendement annuel espéré", -20.0, 60.0, 8.0, .5, format="%.1f%%")
        duree        = st.slider("⏳  Sur combien d'années ?", 1, 20, 5, format="%d ans")
    with c2:
        risque_label = st.select_slider("⚠️  Niveau de risque", options=list(RISQUE_MAP.keys()), value="😐 Modéré")
        type_label   = st.selectbox("📂  Type d'investissement", list(TYPE_MAP.keys()))

    inflation=.025; levier=1.; liquidite=.6; diversif=2; experience=3; notation="BBB"; zone_geo="France"
    with st.expander("⚙️  Paramètres avancés"):
        e1, e2, e3 = st.columns(3)
        with e1:
            inflation  = st.slider("Inflation (%)", .5, 12., 2.5, .1) / 100
            levier     = st.slider("Levier", 1., 8., 1., .1)
        with e2:
            liquidite  = st.slider("Liquidité (0→10)", 0, 10, 6) / 10
            diversif   = st.slider("Secteurs", 1, 12, 2)
        with e3:
            experience = st.slider("Expérience (ans)", 0, 35, 3)
            notation   = st.selectbox("Notation", ["AAA","AA","A","BBB","BB","B","CCC"], index=3)
            zone_geo   = st.selectbox("Zone", list(ENC_ZONE.keys()))

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    if not st.button("🚀  Lancer l'analyse", type="primary", use_container_width=True):
        return

    roi_brut = rendement / 100
    risque   = RISQUE_MAP[risque_label]
    ta_key, sec_key = TYPE_MAP[type_label]
    X        = _vecteur(montant, roi_brut, duree, risque, levier, inflation,
                        liquidite, diversif, experience, sec_key, zone_geo, ta_key, notation)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Verdict unique — XGBoost (meilleur modèle, AUC-ROC 0.645)
    _afficher_verdict_unique(X)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # KPIs
    roi_net = roi_brut - inflation
    val_fin = montant * (1 + roi_brut) ** duree
    gain    = val_fin - montant
    sg      = "+" if gain >= 0 else ""

    k1, k2, k3, k4 = st.columns(4)
    c_g = "#34D399" if gain >= 0 else "#F87171"
    with k1: st.markdown(_kpi_html("Valeur finale",       f"{val_fin:,.0f} €",          "#6EE7F7", "🎯"), unsafe_allow_html=True)
    with k2: st.markdown(_kpi_html("Gain / Perte",        f"{sg}{gain:,.0f} €",         c_g,       "💰"), unsafe_allow_html=True)
    with k3: st.markdown(_kpi_html("Rendement net/an",    f"{roi_net*100:+.1f}%",        c_g,       "📊"), unsafe_allow_html=True)
    with k4: st.markdown(_kpi_html("Ratio rend./risque",  f"{roi_brut/(risque+.1):.2f}", "#818CF8", "⚡"), unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Projection
    annees  = list(range(duree+1))
    v_brute = [montant*(1+roi_brut)**a for a in annees]
    v_nette = [montant*(1+roi_net)**a   for a in annees]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=annees, y=v_brute, name="Valeur brute",
                             line=dict(color="#818CF8",width=2.5),
                             fill="tozeroy", fillcolor="rgba(129,140,248,.06)",
                             hovertemplate="Année %{x} → <b>%{y:,.0f} €</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=annees, y=v_nette, name="Valeur réelle (hors inflation)",
                             line=dict(color="#34D399",width=2,dash="dot"),
                             hovertemplate="Année %{x} → <b>%{y:,.0f} €</b><extra></extra>"))
    fig.add_hline(y=montant, line=dict(color="#334155",width=1,dash="dash"),
                  annotation_text="Capital initial", annotation_font_color="#64748B")
    fig.update_layout(**PLOTLY_BASE, height=300,
                      legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", x=0, y=1.12),
                      xaxis=dict(title="Années", gridcolor="#0F172A"),
                      yaxis=dict(tickformat=",.0f", gridcolor="#0F172A"))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSANTS
# ═══════════════════════════════════════════════════════════════════════════
def _afficher_verdict_unique(X: "pd.DataFrame") -> None:
    """
    Affiche le verdict du meilleur modèle (XGBoost) sous forme
    d'une phrase simple et claire avec la probabilité.
    """
    from pathlib import Path
    from config import MODELS

    # Charger XGBoost (meilleur modèle)
    modele_path = MODELS["xgboost"]["path"]
    if not modele_path.exists():
        # Fallback sur n'importe quel modèle disponible
        for meta in MODELS.values():
            if meta["path"].exists():
                modele_path = meta["path"]
                break
        else:
            st.warning("Aucun modèle trouvé — lancez `python scripts/train.py`.")
            return

    modele = joblib.load(modele_path)
    proba  = float(modele.predict_proba(X)[0][1])
    pred   = int(modele.predict(X)[0])
    pct    = int(proba * 100)

    # Couleur et message selon la probabilité
    if pct >= 65:
        couleur = "#34D399"
        bg      = "#05160F"
        border  = "#34D39944"
        emoji   = "✅"
        verdict = "Investissement probablement rentable"
        detail  = f"Nos modèles estiment que cet investissement a de bonnes chances d'être intéressant."
    elif pct >= 45:
        couleur = "#F59E0B"
        bg      = "#160E02"
        border  = "#F59E0B44"
        emoji   = "⚠️"
        verdict = "Résultat incertain"
        detail  = f"Nos modèles ne parviennent pas à trancher clairement. Prudence recommandée."
    else:
        couleur = "#F87171"
        bg      = "#160505"
        border  = "#F8717144"
        emoji   = "❌"
        verdict = "Investissement risqué"
        detail  = f"Nos modèles estiment que cet investissement présente un risque élevé de ne pas être rentable."

    st.markdown(f"""
    <div style="border:2px solid {border};border-radius:20px;background:{bg};
                padding:2rem 2.5rem;text-align:center;margin:1rem 0">
        <div style="font-size:2.5rem;margin-bottom:.6rem">{emoji}</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;
                    font-weight:700;color:{couleur};margin-bottom:.8rem">
            {verdict}
        </div>
        <div style="font-size:1rem;color:#CBD5E1;margin-bottom:1.5rem">
            {detail}
        </div>
        <div style="display:inline-block;background:#0F172A;border-radius:14px;
                    padding:.8rem 2rem;margin-bottom:.5rem">
            <span style="font-size:.8rem;color:#64748B;text-transform:uppercase;
                         letter-spacing:.1em;display:block;margin-bottom:.3rem">
                Probabilité de rentabilité
            </span>
            <span style="font-family:'Syne',sans-serif;font-size:2.5rem;
                         font-weight:800;color:{couleur}">
                {pct}%
            </span>
        </div>
        <div style="background:#0F172A;border-radius:99px;height:8px;
                    margin:.8rem auto 0;max-width:320px;overflow:hidden">
            <div style="width:{pct}%;height:100%;background:{couleur};
                        border-radius:99px;transition:width .8s ease"></div>
        </div>
        <div style="font-size:.72rem;color:#334155;margin-top:.6rem">
            Basé sur XGBoost · entraîné sur données réelles LendingClub
        </div>
    </div>
    """, unsafe_allow_html=True)



def _kpi_html(label, value, color, icon) -> str:
    return f"""
    <div style="background:#0B1628;border:1px solid #0F2040;border-radius:16px;
                padding:1.1rem 1.3rem;margin-bottom:.4rem">
        <div style="display:flex;align-items:center;gap:.4rem;
                    font-size:.7rem;color:#475569;text-transform:uppercase;
                    letter-spacing:.08em;margin-bottom:.5rem">
            <span>{icon}</span><span>{label}</span>
        </div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;
                    font-weight:700;color:{color};letter-spacing:-.02em">
            {value}
        </div>
    </div>"""


def _verdict_card(nom, pred, proba) -> None:
    v   = "PROFITABLE ✅"    if pred == 1 else "NOT PROFITABLE ❌"
    col = "#34D399"           if pred == 1 else "#F87171"
    bg  = "#05160F"           if pred == 1 else "#160505"
    pct = int(proba * 100)
    st.markdown(f"""
    <div style="border:1.5px solid {col}44;border-radius:16px;background:{bg};
                padding:1.4rem 1rem;text-align:center">
        <div style="font-size:.65rem;color:#475569;text-transform:uppercase;
                    letter-spacing:.1em;margin-bottom:.6rem">{nom}</div>
        <div style="font-family:'Syne',sans-serif;font-size:1rem;
                    font-weight:700;color:{col};margin-bottom:.8rem">{v}</div>
        <div style="background:#0F172A;border-radius:99px;height:6px;margin-bottom:.6rem;overflow:hidden">
            <div style="width:{pct}%;height:100%;background:{col};border-radius:99px"></div>
        </div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:{col}">
            {pct}%
        </div>
        <div style="font-size:.72rem;color:#475569;margin-top:.3rem">probabilité de rentabilité</div>
    </div>
    """, unsafe_allow_html=True)


def _vecteur(montant, roi_brut, horizon, risque, levier, inflation,
             liquidite, diversif, experience, secteur, zone_geo, type_actif, notation) -> pd.DataFrame:
    """
    Construit le vecteur de features pour l'inference.
    Les colonnes correspondent exactement aux features du modele entraine sur LendingClub.
    
    Mapping des inputs utilisateur -> features LendingClub :
      montant_investi        = montant (euros)
      taux_interet           = roi_brut * 100 (proxy du taux d'interet attendu)
      revenu_annuel          = montant / max(risque/10, 0.01) (proxy revenu selon risque)
      dti_ratio              = levier * 10 (proxy endettement)
      incidents_2ans         = 1 si risque > 7, 0 sinon
      fico_score             = 850 - risque * 27  (580-850 selon risque)
      comptes_ouverts        = diversif + 3
      utilisation_revolving  = liquidite * 100 (inverse : faible liquidite = forte utilisation)
      duree_mois             = horizon * 12 clipe a [12, 60]
      grade                  = encode selon risque (0=A, ..., 6=G)
      home_ownership         = 2 (MORTGAGE par defaut)
      purpose                = 0 (debt_consolidation par defaut)
    """
    # Calculs derives
    taux_i   = min(roi_brut * 100, 30.0)             # taux pct
    revenu   = max(montant / max(risque / 10, 0.01), 20000)
    dti      = min(levier * 10, 40.0)
    incidents = 1 if risque > 7 else 0
    fico     = max(580, min(850, int(850 - risque * 27)))
    comptes  = diversif + 3
    util_rev = max(0, min(100, (1 - liquidite) * 100))
    duree    = max(12, min(60, horizon * 12))
    grade_enc = min(6, max(0, int(risque * 0.7)))     # 0-6 selon risque

    # Variables derivees (replicant _preparer_features)
    ratio_pr = montant / (revenu + 1)
    log_taux = np.log1p(taux_i)
    fico_n   = (fico - 580) / (850 - 580)
    log_m    = np.log1p(montant)
    score_end = dti * 0.6 + util_rev * 0.4 / 100

    return pd.DataFrame([{
        "montant_investi":       montant,
        "taux_interet":          round(taux_i, 2),
        "revenu_annuel":         int(revenu),
        "dti_ratio":             round(dti, 2),
        "incidents_2ans":        incidents,
        "fico_score":            fico,
        "comptes_ouverts":       comptes,
        "utilisation_revolving": round(util_rev, 1),
        "duree_mois":            duree,
        "grade":                 grade_enc,
        "home_ownership":        2,
        "purpose":               0,
        "ratio_pret_revenu":     round(ratio_pr, 4),
        "log_taux":              round(log_taux, 4),
        "fico_normalise":        round(fico_n, 4),
        "log_montant":           round(log_m, 4),
        "score_endettement":     round(score_end, 4),
    }])



# ═══════════════════════════════════════════════════════════════════════════
# CSS — épuré, aéré, moderne
# ═══════════════════════════════════════════════════════════════════════════

def _inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Inter:wght@300;400;500&display=swap');

    /* Base */
    html,body,[class*="css"],.stApp{
        font-family:'Inter',sans-serif!important;
        background:#060D1A!important;
        color:#CBD5E1;
    }
    .block-container{
        padding-top:.5rem!important;
        padding-bottom:3rem!important;
        max-width:1200px!important;
    }

    /* Tabs */
    button[data-baseweb="tab"]{
        font-family:'Inter',sans-serif!important;
        font-size:.9rem!important;
        font-weight:500!important;
        color:#64748B!important;
    }
    button[data-baseweb="tab"][aria-selected="true"]{
        color:#F1F5F9!important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-border"]{
        background:#1E293B!important;
    }

    /* Inputs */
    input, [data-baseweb="select"] > div{
        background:#0B1628!important;
        border:1px solid #1E3A5F!important;
        border-radius:10px!important;
        color:#F1F5F9!important;
    }
    label{color:#94A3B8!important;font-size:.82rem!important}

    /* Slider */
    [data-baseweb="slider"] [role="slider"]{
        background:#818CF8!important;
        box-shadow:0 0 0 4px #818CF820!important;
    }

    /* Buttons — primary */
    .stButton button[kind="primary"]{
        background:linear-gradient(135deg,#4F46E5,#7C3AED)!important;
        border:none!important;
        border-radius:12px!important;
        font-family:'Inter',sans-serif!important;
        font-size:.88rem!important;
        font-weight:500!important;
        padding:.65rem 1.2rem!important;
        color:#fff!important;
        transition:opacity .15s!important;
    }
    .stButton button[kind="primary"]:hover{opacity:.85!important}

    /* Buttons — secondary (catégories) */
    .stButton button[kind="secondary"]{
        background:#0B1628!important;
        border:1px solid #1E3A5F!important;
        border-radius:12px!important;
        font-size:.85rem!important;
        color:#94A3B8!important;
        transition:all .15s!important;
    }
    .stButton button[kind="secondary"]:hover{
        border-color:#4F46E5!important;
        color:#F1F5F9!important;
    }

    /* Expander */
    details{
        background:#0B1628!important;
        border:1px solid #0F2040!important;
        border-radius:14px!important;
        padding:.2rem!important;
    }
    summary{color:#94A3B8!important;font-size:.85rem!important}

    /* Form */
    [data-testid="stForm"]{
        background:#0B1628!important;
        border:1px solid #0F2040!important;
        border-radius:16px!important;
        padding:1.2rem!important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"]{border-radius:12px!important;overflow:hidden}

    /* Divider */
    hr{border-color:#0F2040!important}

    /* Metric */
    [data-testid="stMetricValue"]{
        font-family:'Syne',sans-serif!important;
        font-size:1.1rem!important;font-weight:700!important
    }
    [data-testid="stMetricLabel"]{color:#475569!important;font-size:.72rem!important}

    /* Radio */
    [data-testid="stRadio"] label{
        background:#0B1628!important;
        border:1px solid #1E3A5F!important;
        border-radius:10px!important;
        padding:.45rem .9rem!important;
        cursor:pointer!important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar{width:4px;height:4px}
    ::-webkit-scrollbar-track{background:#060D1A}
    ::-webkit-scrollbar-thumb{background:#1E3A5F;border-radius:2px}

    /* Plotly modebar */
    .modebar{background:transparent!important}
    .modebar-btn path{fill:#334155!important}
    </style>
    """, unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — CONSEILLER IA
# ═══════════════════════════════════════════════════════════════════════════

def _sidebar_conseiller() -> None:
    """
    Sidebar intelligente : analyse le portfolio de l'utilisateur
    et génère des conseils personnalisés via Claude API.
    """
    with st.sidebar:
        st.markdown("""
        <div style="padding:.5rem 0 1rem">
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                        color:#F1F5F9;margin-bottom:.2rem">🤖 Conseiller IA</div>
            <div style="font-size:.72rem;color:#475569">Analyse personnalisée de votre portfolio</div>
        </div>
        """, unsafe_allow_html=True)

        pf_raw = st.session_state.get("portfolio", [])

        if not pf_raw:
            st.markdown("""
            <div style="background:#0B1628;border:1px solid #0F2040;border-radius:12px;
                        padding:1rem;text-align:center;color:#475569;font-size:.82rem">
                Ajoutez des investissements pour recevoir des conseils personnalisés.
            </div>
            """, unsafe_allow_html=True)
            return

        # Enrichir les positions
        pf = [enrichir(p) for p in pf_raw]
        total_ach = sum(p["val_ach"] for p in pf)
        total_now = sum(p["val_now"] for p in pf)
        gain_tot  = total_now - total_ach
        perf_pct  = gain_tot / total_ach * 100 if total_ach > 0 else 0

        # Résumé rapide affiché en permanence
        c1, c2 = st.columns(2)
        sg = "+" if gain_tot >= 0 else ""
        gc = "#34D399" if gain_tot >= 0 else "#F87171"
        c1.markdown(f"""
        <div style="background:#0B1628;border:1px solid #0F2040;border-radius:10px;padding:.7rem .8rem">
            <div style="font-size:.65rem;color:#475569;text-transform:uppercase;letter-spacing:.07em">Capital</div>
            <div style="font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;color:#6EE7F7">{total_ach:,.0f} €</div>
        </div>
        """, unsafe_allow_html=True)
        c2.markdown(f"""
        <div style="background:#0B1628;border:1px solid #0F2040;border-radius:10px;padding:.7rem .8rem">
            <div style="font-size:.65rem;color:#475569;text-transform:uppercase;letter-spacing:.07em">Perf.</div>
            <div style="font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;color:{gc}">{sg}{perf_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        st.divider()

        # Bouton analyse
        if st.button("✨  Analyser mon portfolio", type="primary", use_container_width=True):
            _generer_conseils(pf, total_ach, total_now, perf_pct)
        elif "derniers_conseils" in st.session_state:
            # Réafficher les derniers conseils
            _afficher_conseils(st.session_state["derniers_conseils"])

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.divider()

        # Questions rapides
        st.markdown("""
        <div style="font-size:.75rem;color:#475569;text-transform:uppercase;
                    letter-spacing:.07em;margin-bottom:.5rem">Questions rapides</div>
        """, unsafe_allow_html=True)

        questions = [
            "📊 Suis-je trop concentré ?",
            "⚖️  Comment diversifier ?",
            "🛡️  Comment réduire le risque ?",
            "📈 Quelles opportunités saisir ?",
        ]
        for q in questions:
            if st.button(q, key=f"q_{q[:10]}", use_container_width=True):
                _repondre_question(q, pf, total_ach)


def _get_api_key() -> str:
    """Lit la clé API depuis st.secrets ou le fichier secrets.toml directement."""
    # 1. Essayer st.secrets (fonctionne si Streamlit l'a chargé)
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if key:
            return key
    except Exception:
        pass

    # 2. Lire directement le fichier .streamlit/secrets.toml
    try:
        from pathlib import Path
        import re
        secrets_path = Path(__file__).parents[1] / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            text = secrets_path.read_text()
            m = re.search(r'ANTHROPIC_API_KEY\s*=\s*["\'](sk-ant-[^"\']+)["\'"]', text)
            if m:
                return m.group(1)
    except Exception:
        pass

    return ""


def _generer_conseils(pf: list, total_ach: float, total_now: float, perf_pct: float) -> None:
    """Appelle Claude API pour générer des conseils personnalisés."""
    
    # Construire le résumé du portfolio
    # Regrouper avant l'analyse pour éviter les doublons
    groupes_analyse = _grouper_positions(pf)
    positions_txt = []
    for p in groupes_analyse:
        cat = CAT_META.get(p.get("categorie","autre"), CAT_META["autre"])
        pct_port = p["val_now"] / total_now * 100 if total_now > 0 else 0
        positions_txt.append(
            f"- {p.get('nom','?')} ({cat['label']}): {p.get('total_investi', p.get('val_ach', 0)):,.0f}€ investis, "
            f"valeur actuelle {p['val_now']:,.2f}€ ({p['gain_pct']:+.1f}%), "
            f"soit {pct_port:.0f}% du portfolio"
        )

    # Calculer la diversification (depuis les groupes)
    cats_count: dict = {}
    for p in groupes_analyse:
        c = p.get("categorie","autre")
        cats_count[c] = cats_count.get(c, 0) + p.get("val_now", 0)
    
    prompt = f"""Tu es un conseiller financier expert, sympathique et direct. 
Analyse ce portfolio d'investissement et donne 3-4 conseils CONCRETS et PERSONNALISÉS.

PORTFOLIO (capital total: {total_ach:,.0f}€, performance: {perf_pct:+.1f}%):
{chr(10).join(positions_txt)}

Répartition par catégorie:
{chr(10).join([f"- {k}: {v/total_now*100:.0f}%" for k, v in cats_count.items()])}

Donne des conseils en français, directs et actionnables. Format:
- Commence chaque conseil par un emoji et une phrase courte en gras
- Explique pourquoi en 1-2 phrases max
- Sois concret : cite des pourcentages, des actifs spécifiques
- Varie entre encouragements, alertes et suggestions
- Ton: professionnel mais accessible, pas robotique

Réponds UNIQUEMENT avec les conseils, sans intro ni conclusion."""

    with st.spinner("Analyse en cours…"):
        try:
            import requests as req
            response = req.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                    "x-api-key": _get_api_key(),
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30,
            )
            if response.status_code == 200:
                conseils = response.json()["content"][0]["text"]
                st.session_state["derniers_conseils"] = conseils
                _afficher_conseils(conseils)
            else:
                if response.status_code == 401:
                    # Clé invalide → fallback statique
                    conseils = _conseils_statiques(pf, total_ach, total_now, perf_pct)
                    st.session_state["derniers_conseils"] = conseils
                    _afficher_conseils(conseils)
                    st.caption("⚠️ Clé API invalide — conseils automatiques affichés.")
                else:
                    conseils = _conseils_statiques(pf, total_ach, total_now, perf_pct)
                    st.session_state["derniers_conseils"] = conseils
                    _afficher_conseils(conseils)
        except Exception:
            conseils = _conseils_statiques(pf, total_ach, total_now, perf_pct)
            st.session_state["derniers_conseils"] = conseils
            _afficher_conseils(conseils)


def _repondre_question(question: str, pf: list, total_ach: float) -> None:
    """Répond à une question rapide — Claude si dispo, sinon analyse spécifique."""
    total_now = sum(p["val_now"] for p in pf)

    # Construire la liste des positions pour le prompt
    lignes_pf = []
    for p in pf:
        cat_label = CAT_META.get(p.get("categorie","autre"), CAT_META["autre"])["label"]
        lignes_pf.append(f"- {p.get('nom','?')}: {p['val_now']:,.0f}€ ({cat_label})")
    liste_pf = "\n".join(lignes_pf)

    prompt = (
        f"Tu es un conseiller financier expert, direct et bienveillant.\n"
        f"Question : \"{question}\"\n\n"
        f"Portfolio ({total_ach:,.0f}€ investis, {len(pf)} positions) :\n"
        f"{liste_pf}\n\n"
        f"Réponds en 4-6 phrases max, directement à la question, avec des chiffres concrets. "
        f"Cite les actifs de l'utilisateur. En français. Pas d'intro ni conclusion."
    )

    with st.spinner("Analyse en cours…"):
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                    "x-api-key": _get_api_key(),
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=20,
            )
            if response.status_code == 200:
                reponse = response.json()["content"][0]["text"]
                st.session_state["derniers_conseils"] = reponse
                _afficher_conseils(reponse)
                return
        except Exception:
            pass

    # Fallback : réponse spécifique à chaque question
    conseils = _reponse_question_statique(question, pf, total_ach, total_now)
    st.session_state["derniers_conseils"] = conseils
    _afficher_conseils(conseils)


def _reponse_question_statique(question: str, pf: list, total_ach: float, total_now: float) -> str:
    """Génère une réponse pertinente à la question sans API."""
    cats: dict = {}
    for p in pf:
        c = p.get("categorie", "autre")
        cats[c] = cats.get(c, 0) + p["val_now"]
    total = total_now if total_now > 0 else 1

    if "concentré" in question.lower():
        max_cat  = max(cats, key=cats.get) if cats else "autre"
        max_pct  = cats.get(max_cat, 0) / total * 100
        label    = CAT_META.get(max_cat, CAT_META["autre"])["label"]
        if max_pct > 60:
            return (f"⚠️ <b>Oui, votre portfolio est concentré à {max_pct:.0f}% sur {label}.</b><br>"
                    f"C'est au-dessus du seuil recommandé de 50-60% pour une catégorie. "
                    f"Une règle simple : aucune catégorie ne devrait dépasser 50% de votre capital total. "
                    f"Envisagez d'allouer 20-30% vers des ETF monde (SPY, IWDA) ou des obligations "
                    f"pour lisser la volatilité.")
        else:
            return (f"✅ <b>Non, votre concentration est correcte — {label} représente {max_pct:.0f}%.</b><br>"
                    f"Vous êtes dans les clous. Continuez à surveiller que no catégorie "
                    f"ne dépasse 50-60% au fil de vos prochains investissements.")

    elif "diversifier" in question.lower():
        nb_cats = len(cats)
        actifs_manquants = []
        if "obligations" not in cats: actifs_manquants.append("obligations (TLT, AGG) pour la stabilité")
        if "etf" not in cats:         actifs_manquants.append("ETF monde (SPY, IWDA) pour la croissance passive")
        if "immobilier" not in cats:  actifs_manquants.append("REIT ou immobilier direct pour les revenus")
        parts = ", ".join(actifs_manquants[:2]) if actifs_manquants else "vous avez une base solide"
        return (f"📊 <b>Vous avez {nb_cats} catégorie{'s' if nb_cats > 1 else ''} — "
                f"{'bonne base' if nb_cats >= 3 else 'à étoffer'}.</b><br>"
                f"Pour une diversification optimale, visez 4-5 catégories. "
                f"Ce qui manque à votre portfolio : {parts}. "
                f"Idéalement : 40% actions/crypto, 30% ETF, 20% obligations, 10% immobilier.")

    elif "risque" in question.lower():
        crypto_pct = cats.get("crypto", 0) / total * 100
        nb = len(pf)
        conseils_risque = []
        if crypto_pct > 30:
            conseils_risque.append(f"réduire la crypto de {crypto_pct:.0f}% à 15-20% max")
        if nb < 4:
            conseils_risque.append("augmenter le nombre de positions (viser 5-8 actifs)")
        conseils_risque.append("ajouter des obligations ou ETF obligataires (TLT, AGG)")
        return (f"🛡️ <b>Pour réduire le risque de votre portfolio :</b><br>"
                f"1. {conseils_risque[0].capitalize()}<br>"
                f"2. {conseils_risque[1].capitalize()}<br>"
                f"3. {conseils_risque[2].capitalize()}<br>"
                f"Le principe de base : ne jamais mettre plus de 5-10% de son capital sur un seul actif.")

    elif "opportunités" in question.lower():
        crypto_pct  = cats.get("crypto", 0) / total * 100
        etf_pct     = cats.get("etf", 0) / total * 100
        opps = []
        if etf_pct < 20:
            opps.append("📈 <b>ETF S&P 500 (SPY)</b> — exposez-vous à 500 grandes entreprises US en une seule position. Historiquement +10%/an sur 20 ans.")
        if crypto_pct < 10:
            opps.append("🪙 <b>Bitcoin (BTC)</b> — si vous n'êtes pas encore exposé, une petite allocation de 5-10% peut booster la performance globale.")
        opps.append("🌍 <b>MSCI World ETF (IWDA)</b> — diversification mondiale automatique, frais très bas (~0.2%/an), idéal sur le long terme.")
        return "<br><br>".join(opps[:3])

    # Fallback générique
    perf_pct = (total_now - total_ach) / total_ach * 100 if total_ach > 0 else 0
    return (f"💡 <b>Analyse de votre portfolio ({total_ach:,.0f}€, {perf_pct:+.1f}%)</b><br>"
            f"Votre portfolio compte {len(pf)} positions réparties sur {len(cats)} catégorie(s). "
            f"La règle d'or : diversifiez, investissez régulièrement (DCA), et ne paniquez pas à la baisse.")


def _afficher_conseils(texte: str) -> None:
    """Affiche les conseils avec un style card."""
    st.markdown(f"""
    <div style="background:#0B1628;border:1px solid #1E3A5F;border-radius:14px;
                padding:1.1rem;margin-top:.5rem;font-size:.82rem;
                line-height:1.7;color:#CBD5E1">
        {texte.replace(chr(10), "<br>")}
    </div>
    """, unsafe_allow_html=True)


def _conseils_statiques(pf: list, total_ach: float, total_now: float, perf_pct: float) -> str:
    """Génère des conseils basés sur des règles simples (fallback sans API)."""
    conseils = []
    total = total_now if total_now > 0 else 1
    
    # Analyser la concentration
    cats: dict = {}
    for p in pf:
        c = p.get("categorie","autre")
        cats[c] = cats.get(c, 0) + p["val_now"]
    
    max_cat = max(cats, key=cats.get) if cats else None
    max_pct = cats.get(max_cat, 0) / total * 100 if max_cat else 0
    cat_meta = CAT_META.get(max_cat, {})
    
    if max_pct > 70:
        conseils.append(
            f"⚠️ **Concentration élevée sur {cat_meta.get('label', max_cat)}** ({max_pct:.0f}%)<br>"
            f"Envisagez de diversifier : une règle classique est de ne pas dépasser 50% sur une seule catégorie. "
            f"Vous pourriez allouer 20-30% à des ETF monde pour équilibrer."
        )
    else:
        conseils.append(
            f"✅ **Bonne diversification entre vos catégories**<br>"
            f"Votre répartition est équilibrée. Continuez à surveiller que chaque poche "
            f"reste dans la zone 20-40% selon votre appétit au risque."
        )

    # Performance
    if perf_pct > 15:
        conseils.append(
            f"🎯 **Belle performance à +{perf_pct:.1f}%** — pensez à sécuriser une partie<br>"
            f"Quand un actif dépasse 30% de gain, il est raisonnable d'en vendre 20-25% "
            f"pour cristalliser les gains et réinvestir sur des positions moins avancées."
        )
    elif perf_pct < -10:
        conseils.append(
            f"📉 **Performance négative à {perf_pct:.1f}%** — restez calme<br>"
            f"Ne vendez pas sous le coup de l'émotion. Analysez si les fondamentaux ont changé : "
            f"si non, c'est une opportunité de renforcer à prix réduit (DCA)."
        )
    else:
        conseils.append(
            f"📊 **Performance de {perf_pct:+.1f}%** dans la norme<br>"
            f"Votre portfolio suit son cours. Restez investi sur le long terme : "
            f"l'intérêt composé fait la vraie différence sur 5-10 ans."
        )

    # Nb actifs
    nb = len(pf)
    if nb == 1:
        conseils.append(
            f"🔀 **Un seul actif — risque maximal**<br>"
            f"Avec un seul investissement, tout repose sur lui. "
            f"Visez au moins 3-5 actifs différents sur 2+ catégories pour lisser le risque."
        )
    elif nb >= 8:
        conseils.append(
            f"🗂️ **Portfolio bien garni ({nb} positions)**<br>"
            f"Au-delà de 10 actifs, la gestion devient complexe sans gain de diversification. "
            f"Concentrez-vous sur vos meilleures convictions plutôt que de tout diluer."
        )

    # Crypto check
    crypto_pct = cats.get("crypto", 0) / total * 100
    if crypto_pct > 40:
        conseils.append(
            f"🪙 **Crypto à {crypto_pct:.0f}% du portfolio — risque élevé**<br>"
            f"La crypto est volatile (+/- 50% en quelques mois). "
            f"La plupart des conseillers recommandent de ne pas dépasser 5-15% pour un profil modéré."
        )

    return "<br><br>".join(conseils)


build_app()
