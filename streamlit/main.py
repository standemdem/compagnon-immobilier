import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Présentation du projet",
    page_icon="🏠",
    layout="centered"
)

# Titre principal
st.title("Présentation du projet : Compagnon Immobilier")
st.write("")
# Description générale
st.write(
    """
    Ce projet a pour objectif de prédire le prix d'un appartement ou d'un lot d'appartements,\n
    qui pourrait autant être utilisé par des agences immobiliere, que par des citoyens en recherche de biens. \n
    Il s’inscrit dans un cadre de notre projet d'étude au sein de Datascientest et vise\n
    à apporter une solution claire et efficace à la détermination du prix d'un bien dans une \n
    zone précise.
    """
)
st.markdown("---")


# Objectifs
st.header("🎯 Quels sont les objectifs ?")
st.markdown(
    """
    - Comprendre et analyser les différentes données
    - Nettoyer et Préprocesser les datasets
    - Utiliser ou dévelloper un modèle pour prédire la variable cible
    - Développer une solution interactive avec **Streamlit**
    - Fournir une visualisation claire et intuitive
    - Faciliter la prise de décision
    """
)

# Technologies utilisées
st.header("🛠️ Technologies utilisées")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        - Python   
        - Streamlit  
        - Pandas  
        """
    )

with col2:
    st.markdown(
        """
        - NumPy  
        - Matplotlib / Plotly  
        - Seaborn
        """
    )

with col3:
    st.markdown(
        """
        - Scikit-learn
        - Shap
        - Streamlit
        """
    )

# Données
st.header("📂 Données")
st.write(
    """
    Les données utilisées dans ce projet proviennent de celles fourni pour ce projet par Datascientest.\n
    Elles sont nettoyées et prétraitées afin de garantir la qualité des analyses ainsi que de la prédiction.
    """
)

# Auteur / infos
st.markdown("---")
st.subheader("👤 Projet Réalisé par ")
st.write(
    """
    **DE MONTMARIN Stanislas**  
    **FLEURANT Ylan** 
    """
)
st.set_page_config(page_title="Compagnon Immobilier", layout="wide")
