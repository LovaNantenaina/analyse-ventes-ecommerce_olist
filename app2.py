import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(page_title="Predictor VIP Olist", page_icon="🛍️")

# 2. CHARGEMENT DU MODÈLE
@st.cache_resource  # Pour ne charger le modèle qu'une seule fois (gain de performance)
def load_model():
    return joblib.load('D:/DOSSIERS LOVA/PROJET DATA ANALYSIS/PROJET E-COMMERCE/archive/model_xgboost_vip.pkl')

model = load_model()

# 3. INTERFACE UTILISATEUR (Sidebar)
st.sidebar.header("📥 Caractéristiques du Client")
st.sidebar.markdown("Modifiez les curseurs pour simuler un client :")

recence = st.sidebar.slider("Récence (Jours depuis le dernier achat)", 0, 700, 100)
frequence = st.sidebar.slider("Fréquence (Nombre de commandes)", 1, 50, 2)

# 4. PRÉDICTION
st.title("📊 Analyse Prédictive Client - E-commerce")
st.markdown("Ce dashboard utilise un modèle **XGBoost** pour prédire si un client est **VIP** ou **Standard**.")

# Création du vecteur d'entrée pour le modèle
input_data = pd.DataFrame([[recence, frequence]], columns=['Recence', 'Frequence'])

if st.button('🚀 Lancer la Prédiction'):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probabilité d'être VIP

    st.divider()

    if prediction == 1:
        st.success(f"### Résultat : **Client VIP** 💎")
        st.balloons()  # Petit effet visuel sympa
    else:
        st.info(f"### Résultat : **Client Standard** 🛒")

    st.write(f"Confiance du modèle : **{probability * 100:.1f}%**")

# 5. VISUALISATION (Optionnel)
st.divider()
st.subheader("💡 Pourquoi ce résultat ?")
st.write("Le modèle se base sur votre historique d'achat. Un client avec une **faible récence** et une **forte fréquence** a statistiquement plus de chances d'être classé VIP.")