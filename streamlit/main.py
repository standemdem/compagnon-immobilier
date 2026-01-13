import streamlit as st

st.set_page_config(
    page_title="Compagnon Immobilier",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Compagnon Immobilier")
st.markdown("**Prédiction du prix au m² à partir des données DVF (France, 2020).**")

st.write(
    "Application de data science dédiée à l’exploration, au nettoyage, à l’analyse spatiale "
    "et à la modélisation du prix de vente au m² des appartements."
)

st.markdown("**Contributeurs :** Ylan Fleurant & Stanislas de Montmarin — *Bouygues Telecom*")

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.link_button("🔗 Repository GitHub", "https://github.com/standemdem/compagnon-immobilier/")
with col2:
    st.link_button("🚀 App Streamlit", "https://compagnon-immobilier.streamlit.app/")
with col3:
    st.link_button("📊 source de données", "https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres")

st.caption("Compagnon Immobilier — Data Science & Machine Learning")