import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Analyse descriptive finale", page_icon="🏢", layout="wide")

# -------------------------------------------------------------------
# Sources (déjà générées en fin de notebook 06)
# -------------------------------------------------------------------
PATH_STREAMLIT = Path("data/prod/df_streamlit_appart_2020.parquet.gz")
PATH_MODEL = Path("data/prod/df_model_appart_2020.parquet.gz")

# -------------------------------------------------------------------
# Loaders (cache)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

def ensure_prix_m2(df: pd.DataFrame)-> pd.DataFrame:
    df = df.copy()
    if "prix_m2" not in df.columns and {"valeur_fonciere", "surface_reelle_bati"}.issubset(df.columns):
        df = df[df["surface_reelle_bati"].fillna(0) > 0]
        df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    return df

def kpis_prix(df: pd.DataFrame) -> dict:
    out = {"rows": len(df), "cols": df.shape[1]}
    if "id_mutation" in df.columns:
        out["mutations"] = int(df["id_mutation"].nunique())
    else:
        out["mutations"] = np.nan
    if "prix_m2" in df.columns and len(df) > 0:
        out["median"] = float(df["prix_m2"].median())
        out["q25"] = float(df["prix_m2"].quantile(0.25))
        out["q75"] = float(df["prix_m2"].quantile(0.75))
        out["q01"] = float(df["prix_m2"].quantile(0.01))
        out["q99"] = float(df["prix_m2"].quantile(0.99))
    return out

# -------------------------------------------------------------------
# Page
# -------------------------------------------------------------------
st.title("🏢 Analyse descriptive finale")

df_stream = load_parquet(PATH_STREAMLIT)
df_model = load_parquet(PATH_MODEL)

if df_stream.empty:
    st.error(f"Dataset Streamlit introuvable : {PATH_STREAMLIT}")
    st.stop()

if df_model.empty:
    st.error(f"Dataset Modèle introuvable : {PATH_MODEL}")
    st.stop()

df_stream = ensure_prix_m2(df_stream)
df_model = ensure_prix_m2(df_model)  # au cas où

# Sidebar (contrôles légers)
st.sidebar.header("⚙️ Paramètres d'affichage")
sample_n = st.sidebar.slider("Taille d'échantillon pour scatter/carte", 2000, 200000, 20000, step=1000)
q_low = st.sidebar.slider("Quantile bas (coupe)", 0.0, 0.10, 0.01, step=0.005)
q_high = st.sidebar.slider("Quantile haut (coupe)", 0.90, 1.0, 0.99, step=0.005)

tabs = st.tabs([
    "1) Dataset & Sanity checks",
    "2) Distribution prix/m²",
    "3) Localisation géographique",
    "4) Corrélations",
    "5) Jeux finaux (Modèle vs Streamlit)",
])

# -------------------------------------------------------------------
# 1) Dataset & sanity checks
# -------------------------------------------------------------------
with tabs[0]:
    st.markdown(
        """
### 1) Chargement et vérifications de cohérence

Le dataset final “appartements en vente” est utilisé comme base pour :
- la visualisation (jeu Streamlit),
- l’entraînement (jeu Modèle).

Les contrôles suivants visent à vérifier :
- la validité de la cible (prix/m² > 0),
- l’ordre de grandeur des variables principales,
- la complétude des colonnes géographiques.
"""
    )

    ks = kpis_prix(df_stream)
    km = kpis_prix(df_model)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Jeu Streamlit")
        a1, a2, a3 = st.columns(3)
        a1.metric("Lignes", f"{ks['rows']:,}".replace(",", " "))
        a2.metric("Colonnes", f"{ks['cols']:,}".replace(",", " "))
        a3.metric("Prix/m² médian", f"{ks['median']:.0f} €" if "median" in ks else "—")

    with c2:
        st.markdown("#### Jeu Modèle")
        a1, a2, a3 = st.columns(3)
        a1.metric("Lignes", f"{km['rows']:,}".replace(",", " "))
        a2.metric("Colonnes", f"{km['cols']:,}".replace(",", " "))
        a3.metric("Prix/m² médian", f"{km['median']:.0f} €" if "median" in km else "—")

    # Null rates sur géoloc (reprend l’esprit du notebook)
    geo_cols = [c for c in ["latitude", "longitude"] if c in df_stream.columns]
    if geo_cols:
        null_rates = df_stream[geo_cols].isna().mean().mul(100).round(2).to_frame("Taux de NA (%)")
        st.markdown("#### Complétude des coordonnées")
        st.dataframe(null_rates)

    st.markdown("#### Aperçu (échantillon)")
    st.dataframe(df_stream.head(30))

# -------------------------------------------------------------------
# 2) Distribution prix/m²
# -------------------------------------------------------------------
with tabs[1]:
    st.markdown(
        """
### 2) Distribution de la variable cible (prix au m²)

Comme observé précédemment, la distribution du prix au m² est **asymétrique** avec des extrêmes.
Une approche standard consiste à **tronquer** la distribution (ex. 1%–99%) pour :
- améliorer la lisibilité,
- éviter que quelques valeurs très élevées écrasent l’analyse.
"""
    )

    if "prix_m2" not in df_stream.columns:
        st.warning("Colonne `prix_m2` absente.")
    else:
        ql = df_stream["prix_m2"].quantile(q_low)
        qh = df_stream["prix_m2"].quantile(q_high)

        c1, c2, c3 = st.columns(3)
        c1.metric("Quantile bas", f"{ql:,.0f} €".replace(",", " "))
        c2.metric("Médiane", f"{df_stream['prix_m2'].median():,.0f} €".replace(",", " "))
        c3.metric("Quantile haut", f"{qh:,.0f} €".replace(",", " "))

        fig = px.histogram(
            df_stream[(df_stream["prix_m2"] >= ql) & (df_stream["prix_m2"] <= qh)],
            x="prix_m2",
            nbins=90,
            title=f"Distribution du prix/m² (tronquée {int(q_low*100)}%–{int(q_high*100)}%)"
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")

        # Scatter rapide (surface vs prix_m2) comme dans le notebook (mais échantillonné)
        if "surface_reelle_bati" in df_stream.columns:
            d = df_stream.dropna(subset=["surface_reelle_bati", "prix_m2"])
            d = d.sample(min(sample_n, len(d)), random_state=42) if len(d) > sample_n else d

            fig = px.scatter(
                d[(d["prix_m2"] >= ql) & (d["prix_m2"] <= qh)],
                x="surface_reelle_bati",
                y="prix_m2",
                opacity=0.25,
                title="Surface vs prix/m² (échantillon, tronqué sur la cible)"
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, width="stretch")

# -------------------------------------------------------------------
# 3) Localisation géographique
# -------------------------------------------------------------------
with tabs[2]:
    st.markdown(
        """
### 3) Étude de la localisation géographique

Le notebook met en évidence des écarts marqués de médiane entre départements,
et une forte structuration spatiale des prix.

Deux niveaux sont proposés :
- agrégation par département (médiane du prix/m²),
- visualisation cartographique (points géolocalisés).
"""
    )

    needed = {"code_departement", "prix_m2"}
    if not needed.issubset(df_stream.columns):
        st.warning("Colonnes requises manquantes (code_departement, prix_m2).")
    else:
        dep = (
            df_stream
            .dropna(subset=["code_departement", "prix_m2"])
            .groupby("code_departement", as_index=False)["prix_m2"]
            .median()
            .sort_values("prix_m2", ascending=False)
        )

        top_k = st.slider("Nombre de départements à afficher", 5, 30, 15)

        dep_top = dep.head(top_k).copy()
        dep_top["code_departement"] = dep_top["code_departement"].astype(str)  # ✅ conversion sur le subset

        fig = px.bar(
            dep_top,
            x="code_departement",
            y="prix_m2",
            title=f"Top {top_k} départements par prix/m² médian",
            category_orders={"code_departement": dep_top["code_departement"].tolist()},  # ✅ ordre stable
        )
        fig.update_xaxes(type="category")
        fig.update_layout(
            height=380,
            xaxis_title="Département",
            yaxis_title="Prix médian (€/m²)",
        )
        st.plotly_chart(fig, width="stretch")

    # Carte (avec palette lisible + coupe quantiles)
    if {"latitude", "longitude", "prix_m2"}.issubset(df_stream.columns):
        dmap = df_stream.dropna(subset=["latitude", "longitude", "prix_m2"]).copy()

        # keep_bbox = st.checkbox("Filtrer à une bounding box France métro (approx.)", value=True)
        # if keep_bbox:
        #     dmap = dmap[
        #         (dmap["latitude"].between(41, 51.5)) &
        #         (dmap["longitude"].between(-5.5, 9.8))
        #     ]

        dmap = dmap.sample(min(sample_n, len(dmap)), random_state=42) if len(dmap) > sample_n else dmap

        ql = dmap["prix_m2"].quantile(q_low)
        qh = dmap["prix_m2"].quantile(q_high)

        fig = px.scatter_mapbox(
            dmap,
            lat="latitude",
            lon="longitude",
            color="prix_m2",
            color_continuous_scale="Viridis",
            range_color=[ql, qh],
            zoom=4,
            height=560,
            title="Répartition spatiale des ventes (couleur = prix/m², échantillon)"
        )
        fig.update_layout(mapbox_style="carto-positron", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Coordonnées non disponibles dans le dataset Streamlit.")

# -------------------------------------------------------------------
# 4) Corrélations
# -------------------------------------------------------------------
with tabs[3]:
    st.markdown(
        """
### 4) Corrélations entre variables numériques

Le notebook conclut à des corrélations **faibles à modérées**.
L’objectif n’est pas de démontrer une relation linéaire forte, mais de :
- identifier des variables informatives pour la modélisation,
- détecter d’éventuelles redondances,
- appuyer les choix de features retenus pour le modèle.
"""
    )

    # Corrélation sur le dataset modèle (plus proche du ML)
    num_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()

    # Sous-ensemble fixé, cohérent avec le notebook
    selected_cols = [
        c for c in [
            "prix_m2",
            "surface_reelle_bati",
            "nombre_pieces_principales",
            "latitude",
            "longitude",
            "has_dependance",
            "nb_ventes_commune",
        ]
        if c in num_cols
    ]

    if len(selected_cols) < 2:
        st.warning("Pas assez de variables numériques pour calculer une corrélation.")
    else:
        corr = df_model[selected_cols].corr(numeric_only=True)

        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Matrice de corrélation — variables numériques sélectionnées (jeu Modèle)",
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, width="stretch")

        st.markdown(
            """
**Lecture**  
Les coefficients observés confirment l’absence de corrélations linéaires fortes.
Cela justifie l’usage d’un modèle non linéaire (Random Forest), capable de capter
des interactions complexes entre variables.
"""
        )

# -------------------------------------------------------------------
# 5) Jeux finaux
# -------------------------------------------------------------------
with tabs[4]:
    st.markdown(
        """
### 5) Jeux finaux (Modèle vs Streamlit)

Comme dans le notebook, deux jeux sont produits :

- **Jeu Modèle** : features + target (apprentissage, évaluation)
- **Jeu Streamlit** : colonnes nécessaires à l’exploration et à l’interface

Cette séparation améliore :
- la reproductibilité ML (schéma stable du jeu modèle),
- la performance UX (jeu streamlit adapté aux visualisations).
"""
    )

    only_model = sorted(list(set(df_model.columns) - set(df_stream.columns)))
    only_stream = sorted(list(set(df_stream.columns) - set(df_model.columns)))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Colonnes uniquement dans le jeu Modèle")
        st.code(", ".join(only_model) if only_model else "Aucune")
    with c2:
        st.markdown("#### Colonnes uniquement dans le jeu Streamlit")
        st.code(", ".join(only_stream) if only_stream else "Aucune")

    st.markdown("#### Aperçu des schémas")
    s1, s2 = st.columns(2)
    with s1:
        st.write("Jeu Modèle")
        st.dataframe(pd.DataFrame({"col": df_model.columns, "dtype": df_model.dtypes.astype(str)}))
    with s2:
        st.write("Jeu Streamlit")
        st.dataframe(pd.DataFrame({"col": df_stream.columns, "dtype": df_stream.dtypes.astype(str)}))

    st.divider()
    st.markdown(
        """
### Conclusion

Le dataset final “appartements en vente” fournit une base cohérente pour :
- l’exploration (cartographie, agrégations),
- le feature engineering et la modélisation du prix au m²,
- l’intégration dans une application de démonstration.
"""
    )
