import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import requests 

st.set_page_config(layout = 'wide')
st.title("NBA Game Theory Strategy DashBoard")

FASTAPI_URL = 'http://localhost:8000'

st.header("Payoff Matrix Heatmap")
payoff_matrix = pd.read_csv("Data/payoff_matrix.csv")

fig, axis = plt.subplots(figsize = (6,5))
sns.heatmap(payoff_matrix, annot = True, fmt = ".2f", cmap = 'coolwarm', ax = axis)
axis.set_title("Team1 Strat vs Team2 Strat")
st.pyplot(fig)

st.header("Prediction and Explanation")

col1, col2 = st.columns(2)
with col1: 
    strategy_winrate = st.slider("Strategy Winrate", 0.0, 1.0, 0.5)
    strategy_mismatch = st.slider("Strategy Mismatch", 0.0, 4.0, 2.0)
with col2: 
    strategy_power = st.slider("Strategy Average Winrate", 0.0, 1.0, 0.5)
    entropy = st.slider("Entropy", 0.0, 2.33, 1.0)

payload = {
    "strategy_winrate": strategy_winrate,
    "strategy_mismatch": strategy_mismatch,
    "strategy_power": strategy_power, 
    "entropy": entropy
}

if st.button("Run Prediction"):
    with st.spinner("Querying model..."):
        res = requests.post(f"{FASTAPI_URL}/predict/", json=payload)
        if res.status_code == 200:
            data = res.json()
            st.metric("Team 1 Win Prob", f"{data['team1_win_probability']:.2%}")
        else: 
            st.error("Pred Failed")
        
if st.button("Explain with SHAP"):
    with st.spinner("Getting SHAP values..."):
        res = requests.post(f"{FASTAPI_URL}/shap/", json = payload)
        if res.status_code == 200: 
            data = res.json()
            shap_dict = data["shap_feature_contributions"]

            fig2, ax2 = plt.subplots()
            ax2.barh(list(shap_dict.keys()), list(shap_dict.values()), color = "skyblue")
            ax2.set_title("SHAP-Feature-Contributions")
            ax2.axvline(0, color = "red")
            st.pyplot(fig2)
        else: 
            st.error("SHAP req failed")