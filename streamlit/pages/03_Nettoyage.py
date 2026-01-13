import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="EDA 2 — Démarche de nettoyage (Avant / Après)",
    page_icon="🧼",
    layout="wide"
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
PATH_BEFORE = Path("data/parquet/optimized_2020.parquet")
PATH_AFTER = Path("data/processed/dvf_appartements_vente_2020.parquet.gz")

# --------------------------------------------------
# Loaders
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_before(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    cols = [
        "id_mutation",
        "nature_mutation",
        "type_local",
        "valeur_fonciere",
        "surface_reelle_bati",
        "latitude",
        "longitude",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()

    # Filtrage cohérent avec l’EDA initiale
    df = df[df["nature_mutation"] == "Vente"]
    df = df[df["surface_reelle_bati"].fillna(0) > 0]

    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    return df


@st.cache_data(show_spinner=False)
def load_after(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["surface_reelle_bati"].fillna(0) > 0]
    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    return df


def kpis(df: pd.DataFrame) -> dict:
    return {
        "lignes": len(df),
        "mutations": df["id_mutation"].nunique(),
        "prix_m2_med": df["prix_m2"].median(),
        "prix_m2_q95": df["prix_m2"].quantile(0.95),
    }


# --------------------------------------------------
# Page
# --------------------------------------------------
st.title("🧼 Démarche de nettoyage orientée Machine Learning")
st.caption(
    "Cette section présente l’impact du nettoyage des données DVF à travers une comparaison "
    "entre le dataset initial et le dataset final utilisé pour la modélisation."
)

df_before = load_before(PATH_BEFORE)
df_after = load_after(PATH_AFTER)

kb = kpis(df_before)
ka = kpis(df_after)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Constat initial", "Méthodologie", 
                                  "Régles métier", "Avant / Après nettoyage", "Conclusion"])

# --------------------------------------------------
# Constat initial
# --------------------------------------------------
with tab1:
    st.markdown(
        """
    ## 1️⃣ Constat initial : limites du dataset DVF brut

    L’exploration du dataset DVF dans sa forme brute met en évidence plusieurs limites structurelles :

    - une **mutation** peut regrouper plusieurs lignes de transaction,
    - ces lignes peuvent correspondre à **plusieurs types de biens**,
    - une mutation peut inclure **plusieurs appartements**,
    - la valeur foncière est exprimée **au niveau de la mutation**, et non du lot.

    👉 **Conséquence méthodologique** :  
    le calcul naïf d’un prix au m² conduit à une **variable cible ambiguë**, incompatible avec une modélisation fiable.
    """
    )

    st.info(
        "Dans cet état, l’apprentissage d’un modèle de prédiction reviendrait à apprendre sur "
        "des observations dont la cible n’est pas clairement définie."
    )

# --------------------------------------------------
# Objectif méthodologique
# --------------------------------------------------
with tab2:
    st.markdown(
        """
    ## 2️⃣ Objectif méthodologique du nettoyage

    Le nettoyage des données vise à **poser correctement le problème de machine learning**.

    Le dataset recherché doit satisfaire les propriétés suivantes :
    - chaque observation représente **un bien immobilier unique**,
    - les biens sont **comparables entre eux**,
    - la variable cible (prix au m²) est **définie sans ambiguïté**,
    - les règles de sélection sont **explicites, justifiées et reproductibles**.
    """
    )

# --------------------------------------------------
# Règles métier
# --------------------------------------------------
with tab3:
    st.markdown(
        """
    ## 3️⃣ Règles métier appliquées

    Les règles suivantes sont issues de l’analyse exploratoire :

    **1. Restriction aux mutations de type *Vente***  
    → L’objectif est de modéliser le fonctionnement du marché immobilier.

    **2. Restriction au périmètre des appartements**  
    → Le mélange de biens structurellement différents (maisons, locaux, dépendances) augmente artificiellement la variance.

    **3. Exclusion des mutations mixtes**  
    → Les mutations combinant appartements et autres types de biens ne sont pas comparables.

    **4. Règle centrale : *1 mutation = 1 appartement***  
    → Cette contrainte garantit une correspondance non ambiguë entre la mutation et le bien modélisé.

    **5. Conservation d’informations mutationnelles sous forme de variables explicatives**  
    → Exemple : présence d’une dépendance.
    """
    )

# --------------------------------------------------
# KPIs
# --------------------------------------------------
with tab4:
    st.markdown(
        """
    ## 4️⃣ Résultats quantitatifs du nettoyage    
    """
    )
    st.subheader("📊 Indicateurs globaux — Avant / Après nettoyage")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes (avant)", f"{kb['lignes']:,}".replace(",", " "))
    c2.metric("Mutations (avant)", f"{kb['mutations']:,}".replace(",", " "))
    c3.metric("Lignes (après)", f"{ka['lignes']:,}".replace(",", " "))
    c4.metric("Mutations (après)", f"{ka['mutations']:,}".replace(",", " "))

    c1, c2 = st.columns(2)
    c1.metric("Prix/m² médian (avant)", f"{kb['prix_m2_med']:.0f} €")
    c2.metric("Prix/m² médian (après)", f"{ka['prix_m2_med']:.0f} €")

    # --------------------------------------------------
    # Distributions
    # --------------------------------------------------
    st.markdown(
        """
    

    Après nettoyage, la distribution du prix au m² devient plus resserrée et plus cohérente,
    ce qui traduit une réduction significative de l’hétérogénéité.

    Afin de préserver la lisibilité, les distributions sont tronquées au 99ᵉ percentile.
    """
    )

    q_before = df_before["prix_m2"].quantile(0.99)
    q_after = df_after["prix_m2"].quantile(0.99)

    fig1 = px.histogram(
        df_before[df_before["prix_m2"] <= q_before],
        x="prix_m2",
        nbins=80,
        title="Avant nettoyage — distribution étalée et bruitée"
    )
    st.plotly_chart(fig1, width="stretch")

    fig2 = px.histogram(
        df_after[df_after["prix_m2"] <= q_after],
        x="prix_m2",
        nbins=80,
        title="Après nettoyage — distribution stabilisée"
    )
    st.plotly_chart(fig2, width="stretch")

    # --------------------------------------------------
    # Types de biens
    # --------------------------------------------------
    st.markdown(
        """
        La comparaison suivante illustre la transition d’un dataset hétérogène
        vers un périmètre strictement défini.
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        vc = df_before["type_local"].value_counts(normalize=True).mul(100).reset_index()
        vc.columns = ["type_local", "pct"]
        fig = px.bar(vc, x="type_local", y="pct", title="Avant nettoyage — mix des types de biens")
        st.plotly_chart(fig, width="stretch")

    with c2:
        vc = df_after["type_local"].value_counts(normalize=True).mul(100).reset_index()
        vc.columns = ["type_local", "pct"]
        fig = px.bar(vc, x="type_local", y="pct", title="Après nettoyage — périmètre maîtrisé")
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------
# Conclusion
# --------------------------------------------------
with tab5:
    st.warning(
        """
    Cette étape de nettoyage transforme le problème initial :

    - d’un ensemble de transactions hétérogènes et ambiguës,
    - vers un dataset cohérent, comparable et exploitable en machine learning.

    Le dataset obtenu constitue la base :
    - du *feature engineering*,
    - de l’entraînement du modèle de prédiction du prix au m²,
    - et de l’interface de démonstration présentée dans l’application.
    """
    )
