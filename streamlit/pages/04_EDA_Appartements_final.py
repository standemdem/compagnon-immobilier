import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="EDA 3 — Jeux finaux (Modèle vs Streamlit)",
    page_icon="🏢",
    layout="wide",
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
PATH_MODEL = Path("/home/standm/dev/compagnon-immobilier/data/prod/df_model_appart_2020.parquet.gz")
PATH_STREAMLIT = Path("/home/standm/dev/compagnon-immobilier/data/prod/df_streamlit_appart_2020.parquet.gz")


# --------------------------------------------------
# Loaders
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def ensure_prix_m2(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule prix_m2 si absent, sans modifier la logique métier."""
    df = df.copy()
    if "prix_m2" not in df.columns:
        if {"valeur_fonciere", "surface_reelle_bati"}.issubset(df.columns):
            df = df[df["surface_reelle_bati"].fillna(0) > 0]
            df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    return df


def describe_dataset(df: pd.DataFrame) -> dict:
    out = {"rows": len(df), "cols": df.shape[1]}
    if "id_mutation" in df.columns:
        out["mutations"] = int(df["id_mutation"].nunique())
    else:
        out["mutations"] = np.nan

    if "prix_m2" in df.columns and len(df) > 0:
        out["prix_m2_med"] = float(df["prix_m2"].median())
        out["prix_m2_q25"] = float(df["prix_m2"].quantile(0.25))
        out["prix_m2_q75"] = float(df["prix_m2"].quantile(0.75))
    else:
        out["prix_m2_med"] = np.nan
        out["prix_m2_q25"] = np.nan
        out["prix_m2_q75"] = np.nan

    if "surface_reelle_bati" in df.columns and len(df) > 0:
        out["surf_med"] = float(df["surface_reelle_bati"].median())
    else:
        out["surf_med"] = np.nan

    return out


def col_diff(a: pd.DataFrame, b: pd.DataFrame):
    a_cols = set(a.columns)
    b_cols = set(b.columns)
    return sorted(list(a_cols - b_cols)), sorted(list(b_cols - a_cols))


# --------------------------------------------------
# Page
# --------------------------------------------------
st.title("🏢 EDA 3 — Jeux finaux : modélisation vs application")
st.caption(
    "Cette section documente les datasets finaux construits après nettoyage : "
    "un dataset destiné au ML et un dataset optimisé pour l’application Streamlit."
)

df_model = load_parquet(PATH_MODEL)
df_stream = load_parquet(PATH_STREAMLIT)

if df_model.empty:
    st.error(f"Dataset modèle introuvable : {PATH_MODEL}")
    st.stop()

if df_stream.empty:
    st.error(f"Dataset Streamlit introuvable : {PATH_STREAMLIT}")
    st.stop()

df_model = ensure_prix_m2(df_model)
df_stream = ensure_prix_m2(df_stream)

# --------------------------------------------------
# Positionnement méthodologique (académique)
# --------------------------------------------------
st.markdown(
    """
## 1️⃣ Deux jeux finaux : logique “training” vs “serving”

Deux jeux de données sont conservés afin de répondre à deux objectifs complémentaires :

- **Jeu “Modèle”** : destiné à l’entraînement et l’évaluation (variables explicatives, transformations, cible).
- **Jeu “Streamlit”** : destiné à l’exploration et à l’interface (colonnes utiles à la visualisation, cartes, filtres).

Cette séparation permet :
- d’optimiser la performance de l’application (fichiers plus légers et colonnes orientées UX),
- de garantir la reproductibilité du ML (schéma stable et cohérent pour l’entraînement).
"""
)

# --------------------------------------------------
# KPIs globaux
# --------------------------------------------------
st.subheader("📊 Indicateurs descriptifs globaux")

km = describe_dataset(df_model)
ks = describe_dataset(df_stream)

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Dataset Modèle")
    a1, a2, a3 = st.columns(3)
    a1.metric("Lignes", f"{km['rows']:,}".replace(",", " "))
    a2.metric("Colonnes", f"{km['cols']:,}".replace(",", " "))
    a3.metric("Mutations", f"{km['mutations']:,}".replace(",", " ") if not np.isnan(km["mutations"]) else "—")

    b1, b2, b3 = st.columns(3)
    b1.metric("Surface médiane", f"{km['surf_med']:.0f} m²" if not np.isnan(km["surf_med"]) else "—")
    b2.metric("Prix/m² médian", f"{km['prix_m2_med']:.0f} €" if not np.isnan(km["prix_m2_med"]) else "—")
    b3.metric("IQR prix/m²", f"{km['prix_m2_q25']:.0f}–{km['prix_m2_q75']:.0f} €" if not np.isnan(km["prix_m2_q25"]) else "—")

with c2:
    st.markdown("### Dataset Streamlit")
    a1, a2, a3 = st.columns(3)
    a1.metric("Lignes", f"{ks['rows']:,}".replace(",", " "))
    a2.metric("Colonnes", f"{ks['cols']:,}".replace(",", " "))
    a3.metric("Mutations", f"{ks['mutations']:,}".replace(",", " ") if not np.isnan(ks["mutations"]) else "—")

    b1, b2, b3 = st.columns(3)
    b1.metric("Surface médiane", f"{ks['surf_med']:.0f} m²" if not np.isnan(ks["surf_med"]) else "—")
    b2.metric("Prix/m² médian", f"{ks['prix_m2_med']:.0f} €" if not np.isnan(ks["prix_m2_med"]) else "—")
    b3.metric("IQR prix/m²", f"{ks['prix_m2_q25']:.0f}–{ks['prix_m2_q75']:.0f} €" if not np.isnan(ks["prix_m2_q25"]) else "—")

# --------------------------------------------------
# Diff colonnes (très utile à l’oral)
# --------------------------------------------------
st.subheader("🧾 Différences de schéma (colonnes)")

only_in_model, only_in_stream = col_diff(df_model, df_stream)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Colonnes uniquement dans le dataset Modèle**")
    if only_in_model:
        st.code(", ".join(only_in_model))
    else:
        st.caption("Aucune différence notable.")

with col2:
    st.markdown("**Colonnes uniquement dans le dataset Streamlit**")
    if only_in_stream:
        st.code(", ".join(only_in_stream))
    else:
        st.caption("Aucune différence notable.")

st.info(
    "Cette comparaison explicite la séparation des responsabilités : "
    "le dataset modèle est centré sur l’apprentissage et l’évaluation, "
    "tandis que le dataset Streamlit est centré sur l’exploration et la visualisation."
)

# --------------------------------------------------
# Cible (prix_m2) — distribution
# --------------------------------------------------
st.markdown(
    """
## 2️⃣ Variable cible : prix au m²

La variable cible est le **prix au m²**, calculée comme :
`prix_m2 = valeur_fonciere / surface_reelle_bati` (avec surface > 0).

Les distributions ci-dessous sont tronquées au 99ᵉ percentile afin de préserver la lisibilité.
"""
)

tab = st.tabs(["Distribution (Modèle)", "Distribution (Streamlit)"])

with tab[0]:
    if "prix_m2" in df_model.columns:
        q99 = df_model["prix_m2"].quantile(0.99)
        fig = px.histogram(df_model[df_model["prix_m2"] <= q99], x="prix_m2", nbins=90,
                           title="Prix/m² — dataset Modèle (99ᵉ percentile)")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Colonne prix_m2 absente du dataset modèle.")

with tab[1]:
    if "prix_m2" in df_stream.columns:
        q99 = df_stream["prix_m2"].quantile(0.99)
        fig = px.histogram(df_stream[df_stream["prix_m2"] <= q99], x="prix_m2", nbins=90,
                           title="Prix/m² — dataset Streamlit (99ᵉ percentile)")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Colonne prix_m2 absente du dataset Streamlit.")

# --------------------------------------------------
# Variables explicatives — exemples
# --------------------------------------------------
st.markdown(
    """
## 3️⃣ Variables explicatives : aperçu des relations principales

Sans préjuger du modèle, certaines relations structurantes peuvent être observées,
notamment avec la surface et (si disponible) le nombre de pièces.
"""
)

# Scatter surface vs prix_m2 (sample pour perfs)
sample_n = min(15_000, len(df_model))
if {"surface_reelle_bati", "prix_m2"}.issubset(df_model.columns):
    d = df_model.sample(sample_n, random_state=42) if len(df_model) > sample_n else df_model
    fig = px.scatter(
        d,
        x="surface_reelle_bati",
        y="prix_m2",
        opacity=0.25,
        title="Surface vs prix/m² — aperçu (dataset Modèle)"
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

if "nombre_pieces_principales" in df_model.columns and "prix_m2" in df_model.columns:
    # boxplot peut être lourd si énormément de modalités; on force un cast int si nécessaire
    tmp = df_model.copy()
    tmp = tmp[tmp["nombre_pieces_principales"].notna()]
    if len(tmp) > 0:
        # limiter valeurs aberrantes en nb de pièces si nécessaire
        tmp = tmp[tmp["nombre_pieces_principales"] <= tmp["nombre_pieces_principales"].quantile(0.99)]
        fig = px.box(tmp, x="nombre_pieces_principales", y="prix_m2",
                     title="Prix/m² selon le nombre de pièces (dataset Modèle)")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Dimension géographique (plutôt Streamlit)
# --------------------------------------------------
st.markdown(
    """
## 4️⃣ Dimension géographique (jeu Streamlit)

La localisation est un facteur déterminant du prix immobilier.
Le jeu Streamlit conserve typiquement les coordonnées afin de supporter :
- l’exploration cartographique,
- les filtres géographiques,
- l’interprétation des résultats.
"""
)

if {"latitude", "longitude", "prix_m2"}.issubset(df_stream.columns):
    dmap = df_stream.dropna(subset=["latitude", "longitude"]).copy()
    dmap = dmap.sample(min(20_000, len(dmap)), random_state=42) if len(dmap) > 20_000 else dmap

    fig = px.scatter_mapbox(
        dmap,
        lat="latitude",
        lon="longitude",
        color="prix_m2",
        color_continuous_scale="Viridis",
        zoom=4,
        height=520,
        title="Carte des transactions — couleur = prix/m² (dataset Streamlit)"
    )
    fig.update_layout(mapbox_style="carto-positron", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Coordonnées latitude/longitude absentes ou incomplètes dans le dataset Streamlit.")

# --------------------------------------------------
# Limites (académique)
# --------------------------------------------------
st.markdown(
    """
## 5️⃣ Discussion : limites et implications

Malgré un périmètre stabilisé, plusieurs limites restent structurantes :
- bruit résiduel inhérent aux déclarations DVF,
- hétérogénéité spatiale fine (micro-quartiers) difficile à capturer sans données exogènes,
- dépendance forte à la localisation et aux variables proxy disponibles.

Ces éléments motivent l’étape suivante : *feature engineering* et stratégie de modélisation.
"""
)

# --------------------------------------------------
# Conclusion
# --------------------------------------------------
st.divider()
st.markdown(
    """
## ✅ Conclusion

Les jeux finaux (« Modèle » et « Streamlit ») constituent la base stable du projet :
- le jeu Modèle supporte l’apprentissage supervisé du prix au m²,
- le jeu Streamlit supporte la visualisation et la démonstration.

👉 La section suivante présente le **feature engineering** et la construction du pipeline de modélisation.
"""
)
