import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Présentation du projet",
    page_icon="🏠",
    layout="wide"
)

# Titre principal
st.title("Compagnon Immobilier")
st.subheader("🏠 Prédiction du prix/m² des appartements en France")

# Description générale
st.markdown(
    """
    Cette application présente la **démarche de compréhension, de nettoyage et de
    préparation des données** utilisée dans le cadre d’un projet de prédiction
    du prix au mètre carré des appartements en France métropolitaine.  
    Elle inclut également la **modélisation** et la **visualisation interactive** des résultats ainsi
    que l'interprétabilité des modèles utilisés grâce à **SHAP**.  

    Cet outil est destiné à être utilisé par des agences immobilières ainsi que par 
    des particuliers en quête de biens immobiliers dans une zone précise.
    """
)
st.divider()

# Objectifs
st.header("🎯 Quels sont les objectifs du projet?")
st.markdown(
    """
    - Récupérer les données immobilières via des sources publiques ()
    - Comprendre et analyser les différentes données
    - Nettoyer et Préprocesser les datasets
    - Utiliser ou dévelloper un modèle pour prédire la variable cible
    - Développer une solution interactive avec **Streamlit**
    - Fournir une visualisation claire et intuitive
    - Faciliter la prise de décision
    """
)
st.write("")

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
        """
    )
st.write("")

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

