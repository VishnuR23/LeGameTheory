from fastapi import FastAPI
from pydantic import BaseModel 
import joblib 
import pandas as pd
import nashpy as nash
import numpy as np
import shap

app = FastAPI()
model = joblib.load('xgboost_model.pkl')
payoff_matrix = pd.read_csv("Data/payoff_matrix.csv")
explainer = shap.Explainer(model)

class Matchup(BaseModel):
    strategy_winrate: float
    strategy_mismatch: float
    strategy_power: float
    entropy: float


@app.post("/predict/")
def predict(matchup: Matchup):
    input_df = pd.DataFrame([matchup.dict()])
    pred_proba = model.predict_proba(input_df)[0][1]
    pred_label = model.predict(input_df)[0]
    return {
        "predicted_label": int(pred_label),
        "team1_win_probability": float(pred_proba)
    }

@app.post("/simulate/")
def simulate_matchup(strategy_1: int, strategy_2: int):
    winrate = payoff_matrix.values[strategy_1][strategy_2]
    loss_rate = 1 - winrate
    return {
        "strategy_1": strategy_1,
        "strategy_2": strategy_2, 
        "team_1_win_prob": round(winrate, 4),
        "team_2_win_prob": round(lossrate, 4)
    }

@app.post("/shap/")
def shap_explainer(matchup: Matchup):
    df_input = pd.DataFrame([matchup.dict()])
    shap_values = explainer(df_input)
    feature_names = df_input.columns.tolist()
    contributions = dict(zip(feature_names, shap_values.values[0].tolist()))
    return {
        "prediction": float(model.predict_proba(df_input)[0][1]),
        "shap_feature_contributions": contributions
    }

@app.get('/optimal-strategy/')
def recommend_strategy():
    game = nash.Game(payoff_matrix)
    equilibria = list(game.support_enumeration())

    output = []
    for eq in equilbria: 
        row_mix = [round(x, 3) for x in eq[0]]
        col_mix = [round(x, 3) for x in eq[1]]
        output.append({
            "team_1_mixed_strats": row_mix, 
            "team_2_mixed_strats": col_mix
        })
    return {"equilibria": output}