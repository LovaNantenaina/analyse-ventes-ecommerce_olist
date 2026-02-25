import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CONFIGURATION ET STYLE
st.set_page_config(page_title="E-commerce Analytics", layout="wide")
st.title("🛒 Dashboard de Segmentation Client (Olist)")

# 2. CHARGEMENT (Modèle + Données pour les stats)
@st.cache_resource
def load_assets():
    model = joblib.load('C:/Users/ASUS/PycharmProjects/AnalyseECommerce/model_xgboost_vip.pkl')
    # Optionnel : Charger un échantillon de vos stats RFM pour comparer
    stats_moyennes = {'recence': 242, 'frequence': 1.1, 'montant': 120}
    return model

model = load_assets()

# 3. INTERFACE : COLONNES DE KPI (Haut de page)
st.subheader("📈 Indicateurs Clés de Performance (Simulés)")
c1, c2, c3 = st.columns(3)
c1.metric("Panier Moyen", "120.50 R$", "+5%")
c2.metric("Taux de Réachat", "3.2%", "-0.5%")
c3.metric("Score de Fidélité Moyen", "72/100")

st.divider()

# 4. CŒUR DU DASHBOARD : SIMULATION ET GRAPHIQUES
col_sim, col_viz = st.columns([1, 2])  # Colonne gauche étroite, droite large

with col_sim:
    st.header("⚙️ Simulation")
    recence = st.slider("Récence (Jours)", 0, 700, 100)
    frequence = st.slider("Fréquence (Achats)", 1, 10, 1)

    if st.button('🚀 Prédire le Segment'):
        input_data = pd.DataFrame([[recence, frequence]], columns=['Recence', 'Frequence'])
        prediction = model.predict(input_data)

        if prediction == 1:
            st.success("### Résultat : **VIP** 💎")
        else:
            st.info("### Résultat : **Standard** 🛒")

with col_viz:
    st.header("📊 Analyse de l'Importance")

    # Graphique d'importance des variables (Matplotlib)
    fig, ax = plt.subplots(figsize=(8, 4))
    importances = pd.Series(model.feature_importances_, index=['Récence', 'Fréquence'])
    importances.sort_values().plot(kind='barh', color='#ff4b4b', ax=ax)
    ax.set_title("Poids des critères dans la décision de l'IA")
    st.pyplot(fig)

    # Petit comparatif interactif (Bar chart natif Streamlit)
    st.write("Comparaison Récence : Votre simulation vs Moyenne")
    chart_data = pd.DataFrame({
        'Profil': ['Simulation', 'Moyenne'],
        'Jours': [recence, 242]
    })
    st.bar_chart(chart_data, x='Profil', y='Jours', color="#0083B8")

st.divider()
st.info(
    "💡 **Conseil Analyste :** Les clients avec une récence < 200 jours sont les cibles prioritaires pour vos campagnes d'emailing.")
