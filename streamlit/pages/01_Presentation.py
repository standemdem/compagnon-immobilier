import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Accueil — Compagnon Immobilier",
    page_icon="🏠",
    layout="wide",
)

# --- Helpers (optionnels) ---
def file_status(path: str) -> str:
    return "✅ trouvé" if Path(path).exists() else "⚠️ manquant"

# --- Header ---
st.title("🏠 Compagnon Immobilier")
st.caption("Projet ML — Prédiction du prix au m² immobilier (DVF)")

# --- Hero / pitch ---
st.markdown(
    """
### 🎯 Objectif
Cette application présente la **démarche de compréhension, de nettoyage et de préparation des données**
 utilisée dans le cadre d’un projet de prédiction du prix au mètre carré des appartements en France métropolitaine.
Elle inclut également la **modélisation** et la **visualisation interactive** des résultats 
ainsi que l'interprétabilité des modèles utilisés grâce à **SHAP**.

A terme, cet outil pourrait servir à des agents immobiliers, des acheteurs ou des vendeurs 
souhaitant obtenir une estimation **rapide et fiable** du prix au m² d’un bien immobilier.


"""
)
st.markdown(
    """
    ### 🧭 Démarche projet

**Objectif**  
Construire un **MVP réaliste**, exploitable au-delà d’un simple exercice académique.

**Décision clé**  
Données collectées **à la source** via **data.gouv.fr**  
→ *Demandes de Valeurs Foncières (DVF)*

**Problème rencontré**  
Volumes importants → **instabilité du kernel** sur des ressources matérielles limitées.

**Solution mise en place**  
Chaîne automatisée : **téléchargement → structuration → CSV → Parquet**

**Impact**  
✔️ Mémoire optimisée  
✔️ Environnement stable  
✔️ Analyse et modélisation possibles à grande échelle
    """
)
# --- Layout 2 colonnes ---
left, right = st.columns(2)

with left:
    st.subheader("📦 Données & artefacts")
    # Ajuste les chemins selon ton repo
    status_dvf = file_status("data/parquet/optimized_2020.parquet")
    status_streamlit = file_status("data/prod/df_streamlit_appart_2020.parquet.gz")
    statuts_training = file_status("data/prod/df_model_appart_2020.parquet.gz")
    status_model = file_status("data/models/prix_m2_pipeline_2020.joblib")

    st.markdown(
        f"""
        - Dataset DVF (source projet) : **{status_dvf}**
        - Dataset Streamlit (apparts final) : **{status_streamlit}**
        - Dataset Modélisation (apparts final) : **{statuts_training}**
        - Modèle entraîné : **{status_model}**
        """
    )
with right:
    st.subheader("🛠️ Stack")
    st.markdown(
        """
- **Python**, **pandas**, **numpy**, **matplotlib**, **seaborn** , **plotly**
- **scikit-learn** (pipeline, modèles, métriques) **SHAP** (interprétabilité)
- **Streamlit** (app multipage)
- **parquet** (performance / taille)
"""
    )



# --- CTA (call-to-action) ---
st.divider()
st.subheader("🚀 Démarrer la visite")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.page_link("pages/02_Exploration.py", label="📊 Exploration")
with c2:
    st.page_link("pages/03_Nettoyage.py", label="🧼 Nettoyage")
with c3:
    st.page_link("pages/04_Analyse_descriptive_finale.py", label="🏢 Analyse descriptive")
with c4:
    st.page_link("pages/05_Prediction.py", label="🔮 Prédiction")
with c5:
    st.page_link("pages/06_Conclusion.py", label="🏁 Conclusion")

