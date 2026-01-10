import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA Streamlit", layout="wide")

# ============================
# Markdown & Code from notebook
# ============================

st.markdown("# EDA du dataset optimized_2020.parquet")

st.markdown("## Descriptif des colonnes du fichier csv")

# 📦 Imports (déjà faits plus haut)

# 📄 Chargement du fichier optimisé
# ⚠️ Le chemin est identique à celui du notebook
# Assure-toi que le fichier existe au même endroit

df = pd.read_parquet('./data/parquet/optimized_2020.parquet')

st.subheader("Aperçu du dataset")
st.dataframe(df.head())

st.subheader("Statistiques descriptives")
st.dataframe(df.describe(include='all').T)

# ============================
# Visualisations
# ============================

st.subheader("Valeurs manquantes")
fig, ax = plt.subplots()
df.isna().sum().plot(kind='bar', ax=ax)
st.pyplot(fig)

st.subheader("Distribution des variables numériques")
num_cols = df.select_dtypes(include='number').columns
for col in num_cols:
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(col)
    st.pyplot(fig)

st.subheader("Matrice de corrélation")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df[num_cols].corr(), cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.success("Notebook converti en application Streamlit (1 page)")
