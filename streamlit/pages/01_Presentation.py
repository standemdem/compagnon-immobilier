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
Construire un modèle capable d’estimer le **prix au m²** d’un bien à partir de variables DVF
(surface, pièces, localisation, etc.), puis le rendre **explorable** via une **app Streamlit multipage**.

### Pourquoi le prix au m² ?
- plus comparable entre biens
- moins dépendant de la surface que le prix total
- plus robuste pour une approche territoriale (communes, départements, etc.)
"""
)

# --- Layout 2 colonnes ---
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("🧭 Plan de lecture")
    st.markdown(
        """
1. **EDA 1 — Exploration naïve** : comprendre le dataset brut et ses limites (bruit, outliers, hétérogénéité).
2. **EDA 2 — Approche pro** : nettoyage, règles métier, comparaisons avant/après, réduction du bruit.
3. **EDA 3 — Focus appartements** : périmètre final stable → dataset exploitable pour le ML.
4. **Feature engineering** : sélection/transformations, prévention du leakage.
5. **Modélisation** : baseline vs modèle final, métriques & interprétation.
6. **Démo** : formulaire d’estimation €/m² (et éventuellement prix total).
"""
    )

    st.subheader("✅ Périmètre final")
    st.markdown(
        """
- Travail final centré sur **les appartements** (réduction de variance / comparabilité).
- Une démarche progressive : *brut → nettoyé → périmètre final*.
- Orientation “produit” : résultat présentable + démo.
"""
    )

with right:
    st.subheader("📦 Données & artefacts")
    # Ajuste les chemins selon ton repo
    status_dvf = file_status("data/parquet/optimized_2020.parquet")
    status_streamlit = file_status("data/prod/df_streamlit_appart_2020.parquet.gz")
    status_model = file_status("data/models/model.joblib")

    st.markdown(
        f"""
- Dataset DVF (source projet) : **{status_dvf}**
- Dataset Streamlit (apparts final) : **{status_streamlit}**
- Modèle entraîné : **{status_model}**
"""
    )

    st.subheader("🛠️ Stack")
    st.markdown(
        """
- **Python**, **pandas**, **numpy**
- **scikit-learn** (pipeline, modèles, métriques)
- **Streamlit** (app multipage)
- **parquet** (performance / taille)
"""
    )

    st.subheader("▶️ Lancer l’app")
    st.code("streamlit run app.py", language="bash")

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

