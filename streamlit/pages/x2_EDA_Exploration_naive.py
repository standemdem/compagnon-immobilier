import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="EDA 1 — Exploration naïve", page_icon="📊", layout="wide")

# -----------------------------
# Paramètres
# -----------------------------
DATA_PATH = Path("data/parquet/optimized_2020.parquet")

DEFAULT_COLS = [
    "nature_mutation",
    "valeur_fonciere",
    "surface_reelle_bati",
    "surface_terrain",
    "type_local",
    "nom_commune",
    "code_commune",
    "code_departement",
    "latitude",
    "longitude",
]

# -----------------------------
# Data loader
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)

    # On garde seulement les colonnes utiles (si elles existent)
    cols = [c for c in DEFAULT_COLS if c in df.columns]
    df = df[cols].copy()

    return df


def add_zone_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Reprise de l'idée notebook : lat < 41 => DROM/COM (naïf mais utile en EDA 1)."""
    if "latitude" not in df.columns:
        df["zone_geo"] = "Inconnu"
        return df

    df = df.copy()
    df["zone_geo"] = "Métropole"
    df.loc[df["latitude"] < 41, "zone_geo"] = "DROM/COM"
    return df


def add_prix_m2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "valeur_fonciere" in df.columns and "surface_reelle_bati" in df.columns:
        df = df[df["surface_reelle_bati"].fillna(0) > 0].copy()
        df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    return df


# -----------------------------
# UI
# -----------------------------
st.title("📊 EDA 1 — Exploration naïve (dataset brut)")
st.caption("Objectif : visualiser rapidement le dataset DVF 2020 et comprendre pourquoi un cadrage + nettoyage sont indispensables.")

df = load_data(DATA_PATH)

if df.empty:
    st.error(
        "Impossible de charger le dataset. Vérifie le chemin : "
        f"`{DATA_PATH.as_posix()}` (attendu dans ton repo)."
    )
    st.stop()

# Sidebar controls
st.sidebar.header("⚙️ Réglages")
only_sales = st.sidebar.checkbox("Filtrer nature_mutation = Vente", value=True)
only_metropole = st.sidebar.checkbox("Garder uniquement la Métropole", value=True)
max_points = st.sidebar.slider("Nb points max sur la carte", 2_000, 50_000, 15_000, step=1_000)
outlier_q = st.sidebar.slider("Seuil outliers (valeur_fonciere quantile)", 0.90, 0.999, 0.99, step=0.001)

# -----------------------------
# Préparation "naïve" façon notebook
# -----------------------------
df = add_zone_geo(df)

if only_sales and "nature_mutation" in df.columns:
    df = df[df["nature_mutation"] == "Vente"].copy()

# Drop NaN valeur_fonciere (comme le notebook)
if "valeur_fonciere" in df.columns:
    df = df.dropna(subset=["valeur_fonciere"]).copy()

if only_metropole and "zone_geo" in df.columns:
    df = df[df["zone_geo"] == "Métropole"].copy()

# Ajout prix/m² (naïf)
df = add_prix_m2(df)

# -----------------------------
# KPIs
# -----------------------------
k1, k2, k3, k4 = st.columns(4)

k1.metric("Transactions (après filtres)", f"{len(df):,}".replace(",", " "))
if "valeur_fonciere" in df.columns:
    k2.metric("Valeur foncière médiane", f"{df['valeur_fonciere'].median():,.0f} €".replace(",", " "))
else:
    k2.metric("Valeur foncière médiane", "—")

if "prix_m2" in df.columns:
    k3.metric("Prix/m² médian", f"{df['prix_m2'].median():,.0f} €".replace(",", " "))
    k4.metric("Prix/m² max", f"{df['prix_m2'].max():,.0f} €".replace(",", " "))
else:
    k3.metric("Prix/m² médian", "—")
    k4.metric("Prix/m² max", "—")

st.divider()

# -----------------------------
# Section 1 — Ce qu’on voit tout de suite (brut)
# -----------------------------
st.subheader("1) Distributions brutes")

# Log(valeur fonciere + 1) comme le notebook
if "valeur_fonciere" in df.columns:
    df["valeur_fonciere_log"] = np.log1p(df["valeur_fonciere"])

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="valeur_fonciere_log", nbins=60, title="Log(valeur foncière + 1)")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # distribution brute (souvent illisible, mais justement: EDA naïf)
        fig = px.histogram(df, x="valeur_fonciere", nbins=80, title="Valeur foncière (brut)")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

if "prix_m2" in df.columns:
    fig = px.histogram(df, x="prix_m2", nbins=80, title="Prix au m² (brut — très bruité)")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.info(
    "On observe immédiatement des distributions très étalées et des valeurs extrêmes. "
    "C’est précisément ce qui motive le nettoyage et le choix d’un périmètre cohérent (pages suivantes)."
)

# -----------------------------
# Section 2 — Catégorielles (comme ton notebook : modalités <= 30)
# -----------------------------
st.subheader("2) Variables catégorielles (aperçu rapide)")
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
cat_cols = [c for c in cat_cols if c not in ["id_mutation", "id_parcelle"]]

col_pick = st.selectbox(
    "Choisis une variable catégorielle (affichage si peu de modalités)",
    options=["(aucune)"] + cat_cols,
    index=0,
)

if col_pick != "(aucune)":
    nunique = df[col_pick].nunique(dropna=False)
    st.caption(f"Modalités (avec NaN) : {nunique}")
    if nunique <= 30:
        vc = df[col_pick].value_counts(dropna=False).reset_index()
        vc.columns = [col_pick, "count"]
        fig = px.bar(vc, x=col_pick, y="count", title=f"Répartition des modalités — {col_pick}")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Trop de modalités pour un graphe lisible (c’est un point important pour le preprocessing).")
        st.dataframe(df[col_pick].value_counts(dropna=False).head(20))

# -----------------------------
# Section 3 — Carte brute (échantillon)
# -----------------------------
st.subheader("3) Carte brute (échantillon)")
if {"latitude", "longitude"}.issubset(df.columns):
    dmap = df.dropna(subset=["latitude", "longitude"]).copy()
    if len(dmap) > max_points:
        dmap = dmap.sample(max_points, random_state=42)

    color_col = "prix_m2" if "prix_m2" in dmap.columns else ("valeur_fonciere" if "valeur_fonciere" in dmap.columns else None)

    if color_col:
        fig = px.scatter_mapbox(
            dmap,
            lat="latitude",
            lon="longitude",
            color=color_col,
            zoom=4,
            height=550,
            title=f"Transactions (couleur = {color_col})",
        )
        fig.update_layout(mapbox_style="carto-positron", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Pas de colonne utilisable pour la couleur (prix_m2 / valeur_fonciere).")
else:
    st.warning("Pas de colonnes latitude/longitude disponibles pour la carte.")

# -----------------------------
# Section 4 — Outliers (reprend ta logique 1% plus chers)
# -----------------------------
st.subheader("4) Focus outliers — Top quantile en valeur foncière")
if "valeur_fonciere" in df.columns:
    seuil = df["valeur_fonciere"].quantile(outlier_q)
    df_out = df[df["valeur_fonciere"] > seuil].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Seuil outlier", f"{seuil:,.0f} €".replace(",", " "))
    c2.metric("Nb outliers", f"{len(df_out):,}".replace(",", " "))
    if "prix_m2" in df_out.columns and len(df_out) > 0:
        c3.metric("Prix/m² moyen (outliers)", f"{df_out['prix_m2'].mean():,.0f} €".replace(",", " "))
    else:
        c3.metric("Prix/m² moyen (outliers)", "—")

    if "type_local" in df_out.columns and len(df_out) > 0:
        top_types = df_out["type_local"].value_counts().head(10).reset_index()
        top_types.columns = ["type_local", "count"]
        fig = px.bar(top_types, x="count", y="type_local", orientation="h", title="Top 10 types de biens avec le plus d’outliers")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    if "prix_m2" in df_out.columns and len(df_out) > 0:
        fig = px.histogram(df_out, x="prix_m2", nbins=60, title="Distribution du prix/m² des outliers")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Colonne valeur_fonciere manquante : impossible de calculer les outliers.")

# -----------------------------
# Conclusion narrative (EDA 1)
# -----------------------------
st.divider()
st.subheader("✅ Conclusion (EDA 1)")
st.markdown(
    """
Cette exploration volontairement “brute” met en évidence :

- une **forte hétérogénéité** (types de biens, surfaces, géographies)
- des **outliers** massifs (valeur foncière et prix/m²)
- une donnée DVF qui nécessite des **règles métier** et un **périmètre comparable** avant toute modélisation

👉 Les pages suivantes montrent comment on passe d’un dataset bruité à un dataset **model-ready**.
"""
)
