import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Exploration naïve — DVF", layout="wide")

# =========================
# Helpers
# =========================
@st.cache_data
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def format_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")

def memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / 1e6)

def top_missing(df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    s = df.isna().mean().sort_values(ascending=False) * 100
    out = s.head(k).round(2).reset_index()
    out.columns = ["colonne", "taux_manquant_%"]
    return out

def df_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "colonne": df.columns,
        "type": df.dtypes.astype(str),
        "taux_manquant_%": (df.isna().mean() * 100).round(2),
        "nb_uniques": df.nunique(dropna=True).values
    })
    return out

def safe_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)

# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ Paramètres")

default_path = "data/parquet/optimized_2020.parquet"
data_path = st.sidebar.text_input("Chemin du dataset DVF brut (.parquet)", value=default_path)

sample_n = st.sidebar.slider(
    "Échantillon pour les graphes (pour éviter de tout charger en mémoire)",
    min_value=5_000, max_value=200_000, value=50_000, step=5_000
)

top_k_modalities = st.sidebar.slider(
    "Top modalités (catégorielles)",
    min_value=5, max_value=50, value=15, step=1
)

st.sidebar.markdown("---")
show_preview = st.sidebar.checkbox("Afficher un aperçu (head)", value=True)
preview_rows = st.sidebar.slider("Nombre de lignes dans l'aperçu", 5, 50, 15)

# =========================
# Load data
# =========================
try:
    df = load_parquet(data_path)
except Exception as e:
    st.error(f"Impossible de charger le fichier : {data_path}\n\nErreur : {e}")
    st.stop()

# =========================
# Title / Intro
# =========================
st.title("🔍 Exploration naïve des données DVF (brutes)")
st.markdown(
    """
    Cette page présente une **première lecture descriptive** des données DVF, avant tout filtrage,
    agrégation ou choix de variables.  
    L’objectif est de **comprendre la structure** du dataset : volume, types de variables, valeurs manquantes,
    variables quantitatives et catégorielles.
    """
)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Vue d'ensemble", "Structure", "Variables quantitatives", "Variables catégorielles", "Constats"])
# =========================
# Section 1 — Overview
# =========================
st.divider()
with tab1:
    st.header("1) Vue d’ensemble")

    # Colonnes numériques / catégorielles
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    # Mutations
    nb_mutations = df["id_mutation"].nunique() if "id_mutation" in df.columns else None

    # Valeur foncière min/max
    vf_min, vf_max = None, None
    if "valeur_fonciere" in df.columns:
        vf_min = df["valeur_fonciere"].min(skipna=True)
        vf_max = df["valeur_fonciere"].max(skipna=True)

    # Max lignes par mutation
    max_lines_per_mut = None
    if "id_mutation" in df.columns:
        max_lines_per_mut = df.groupby("id_mutation").size().max()

    # Affichage KPI
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Lignes", format_int(df.shape[0]))
    with k2:
        st.metric("Colonnes", format_int(df.shape[1]))
    with k3:
        st.metric("Colonnes numériques", format_int(len(num_cols)))
    with k4:
        st.metric("Colonnes catégorielles", format_int(len(cat_cols)))

    k5, k6, k7, k8 = st.columns(4)
    with k5:
        st.metric("Mutations uniques", format_int(nb_mutations) if nb_mutations is not None else "—")
    with k6:
        st.metric(
            "Valeur foncière min",
            f"{vf_min:,.0f} €".replace(",", " ") if vf_min is not None else "—"
        )
    with k7:
        st.metric(
            "Valeur foncière max",
            f"{vf_max:,.0f} €".replace(",", " ") if vf_max is not None else "—"
        )
    with k8:
        st.metric(
            "Max lignes / mutation",
            format_int(int(max_lines_per_mut)) if max_lines_per_mut is not None else "—"
        )
    st.image(
        "streamlit/assets/images/map_ventes_full_2020.png",
        use_container_width=False
    )
    st.warning(
        """
        - Dataset lourd (bcp de lignes et de colonnes).
        - Les valeurs foncières extrêmes (min / max) peuvent indiquer des mutations particulières 
        (dons, mutations mixtes et/ou composite).
        - Présence de mutations dans les DROM/COM (Guadeloupe, Réunion, …) et en Corse.
        - Absence des mutations en Alsace-Moselle (Régime spécial).  
        **Point important :** dans DVF, une ligne ne correspond pas toujours à un bien unique.
        DVF décrit des **mutations** (transactions) pouvant comporter plusieurs lignes (lots, dépendances, parcelles…).
        """
    )
    if max_lines_per_mut is not None:
        with st.expander("Voir un exemple de mutation avec beaucoup de lignes"):
            # On récupère un id_mutation qui atteint le max (ou proche)
            sizes = df.groupby("id_mutation").size()
            example_id = sizes.sort_values(ascending=False).index[0]
            st.write(f"Exemple id_mutation : **{example_id}** (nb lignes = {int(sizes.loc[example_id])})")
            st.dataframe(df[df["id_mutation"] == example_id].head(30), width="stretch")


# =========================
# Section 2 — Schema / Missingness
# =========================
with tab2:
    st.header("2) Structure des colonnes et valeurs manquantes")

    schema = df_schema(df)

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("**Schéma des colonnes (type, manquants, cardinalité)**")
        st.dataframe(schema.sort_values("taux_manquant_%", ascending=False), width="stretch", height=420)

    with right:
        st.markdown("**Top colonnes les plus manquantes**")
        st.dataframe(top_missing(df, k=20), width="stretch", height=420)

    st.warning(
        "À ce stade, l’objectif est uniquement descriptif : la présence de valeurs manquantes et la diversité des types "
        "sont des signaux importants pour la suite (nettoyage, filtrage, agrégation)."
    )

# =========================
# Section 3 — Quantitative variables
# =========================
with tab3:
    st.header("3) Variables quantitatives (numériques)")


    # --- 3.1 Nombre de pièces principales ---
    st.subheader("3.1) Distribution du nombre de pièces principales")

    if "nombre_pieces_principales" not in df.columns:
        st.warning("La colonne `nombre_pieces_principales` n'est pas présente dans le dataset.")
    else:
        s = df["nombre_pieces_principales"].dropna()

        # Optionnel : filtrage léger pour lisibilité (souvent 0–10)
        s_plot = s[(s >= 0) & (s <= 10)]

        fig = px.histogram(
            s_plot,
            nbins=11,
            title="Nombre de pièces principales (0 à 10)",
            labels={"value": "nombre_pieces_principales", "count": "Fréquence"}
        )
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, width="stretch")


        st.warning(
            "Le nombre de pièces principales donne une lecture rapide de la typologie des mutations."
        )

    # --- 3.2 Top 10 codes postaux ---
    st.subheader("3.2) Top 10 des départements les plus représentés")

    if "code_departement" not in df.columns:
        st.warning("La colonne `code_departement` n'est pas présente dans le dataset.")
    else:
        dep = df["code_departement"].dropna().astype(str)

        # IMPORTANT : ne pas zfill sur 3 chiffres (ex: 2A/2B n'existent pas en métropole DVF standard)
        # On garde tel quel, et on nettoie juste les ".0" si la colonne est float
        dep = dep.str.replace(r"\.0$", "", regex=True)

        top10_dep = (
            dep.value_counts()
            .head(10)
            .reset_index()
        )
        top10_dep.columns = ["code_departement", "count"]

        # forcer l'ordre (du plus fréquent au moins fréquent)
        order = top10_dep.sort_values("count", ascending=False)["code_departement"].tolist()

        fig = px.bar(
            top10_dep,
            x="code_departement",
            y="count",
            category_orders={"code_departement": order},
            title="Top 10 des départements par fréquence",
            labels={"code_departement": "Code département", "count": "Nombre de lignes"},
            text="count"
        )
        fig.update_traces(textposition="inside")
        fig.update_layout(xaxis_type="category")  # 🔑 force catégoriel
        st.plotly_chart(fig, width="stretch")

        with st.expander("Afficher le détail (table)"):
            top10_dep["proportion_%"] = (top10_dep["count"] / top10_dep["count"].sum() * 100).round(2)
            st.dataframe(top10_dep, width="stretch")

        st.warning(
            "Les distributions numériques sont souvent asymétriques dans DVF et peuvent contenir des valeurs extrêmes. "
            "Cette observation motive des analyses plus poussées avant modélisation."
        )

# =========================
# Section 4 — Catégorielles (MODIFIÉE)
# =========================
with tab4:
    st.header("4) Variables catégorielles — répartition (%)")

    st.markdown(
        """
        On analyse ici deux variables catégorielles structurantes :
        - **nature_mutation** : type d’événement enregistré (vente, VEFA, échange, adjudication, …)
        - **type_local** : type de bien (appartement, maison, dépendance, …)
        """
    )

    def plot_cat_percent(df: pd.DataFrame, col: str, top_k: int = 15, title: str = ""):
        if col not in df.columns:
            st.warning(f"La colonne `{col}` n'est pas présente dans le dataset.")
            return

        s = df[col].copy()

        # Normaliser l'affichage des NaN
        s = s.astype("object")
        s = s.where(~s.isna(), other="NaN")

        # Répartition en %
        vc = (s.value_counts(normalize=True) * 100).round(3)

        # Top K modalités
        vc_top = vc.head(top_k).reset_index()
        vc_top.columns = [col, "pourcentage"]

        # Ordre décroissant (plus fréquent -> moins fréquent)
        order = vc_top.sort_values("pourcentage", ascending=False)[col].tolist()

        fig = px.bar(
            vc_top,
            x=col,
            y="pourcentage",
            category_orders={col: order},
            title=title if title else f"Répartition de {col} (Top {top_k})",
            labels={col: col, "pourcentage": "Pourcentage (%)"},
            text="pourcentage"
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="inside")
        fig.update_layout(
            xaxis_type="category",
            yaxis_ticksuffix="%",
            uniformtext_minsize=8,
            uniformtext_mode="hide",
            margin=dict(t=60, b=90)
        )
        fig.update_xaxes(tickangle=-30)

        st.plotly_chart(fig, width="stretch")

        with st.expander(f"Afficher le détail — {col}"):
            detail = s.value_counts(dropna=False).rename("count").reset_index().rename(columns={"index": col})
            detail["pourcentage_%"] = (detail["count"] / detail["count"].sum() * 100).round(3)
            st.dataframe(detail, width="stretch")

    # --- nature_mutation ---
    st.subheader("4.1) Répartition de la nature de mutation")
    plot_cat_percent(
        df,
        col="nature_mutation",
        top_k=10
    )

    # --- type_local ---
    st.subheader("4.2) Répartition du type de local")
    plot_cat_percent(
        df,
        col="type_local",
        top_k=10
    )
    st.warning(
        """
        On observe une majorité de mutations de type 'Vente' (90,74%).  
        La répartition des types de locaux est assez déséquilibrée avec un majorité de NaN et une répartition 
        relativement équilibrée enter maison, appartement et dépendance.
        """
    )
# =========================
# Section 5 — First takeaways
# =========================
with tab5:
    st.header("5) Premiers constats")

    st.warning(
        """
        Cette exploration du dataset brut met en évidence plusieurs éléments structurants :

        - **Hétérogénéité des variables** : montants, surfaces, catégories administratives.
        - **Valeurs manquantes** parfois très importantes voir majoritaire sur certaines colonnes.
        - **Granularité DVF** : la transaction (“mutation”) peut être décrite sur plusieurs lignes.
        - **Distributions asymétriques** et valeurs extrêmes possibles sur les variables numériques.

        👉 Ces constats justifient les étapes suivantes : compréhension des mutations, filtrage du périmètre
        (ventes, appartements), puis préparation d’un dataset exploitable pour l’analyse et la modélisation.
        """
    )

