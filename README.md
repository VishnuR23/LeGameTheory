🏀 LeGameTheory: A NBA Game Theory + ML Strategy Predictor

This project combines game theory, machine learning, and strategic analysis to predict NBA game outcomes based on team strategies. Built from scratch using Python, FastAPI, XGBoost, SHAP, 
and Streamlit. The system estimates win probabilities, explains feature impact, simulates strategic matchups, and even recommends optimal strategies using Nash equilibria.

PROBLEM AND SOLUTION:
Traditional NBA prediction models focus on player stats and historical performance. This project takes a different approach — it models teams as strategic agents, analyzes play style 
clusters, and predicts outcomes using payoff matrices, entropy, and other features rooted in game theory.

PRIMARY CONCEPTUAL IDEAS:
1. Game Theory: Strategy modeling, Nash equilibrium, and matchup payoff matrices.
2. Machine Learning: Gradient boosting via XGBoost for classification.
3. Explainability: SHAP values to explain prediction rationale.
4. Statistical Feature Engineering: Entropy, mismatch, average win rate, and strategic power.

DATA EXPLORATION: 
Handled the large NBA-Data MAtchup Dataset from NBA-API and cleaned the data. 
Additionally, Explored the Data and visualized it using various graphical strategies.

FEATURE ENGINEERING: 
Firstly, Teams were clustered into 5 main strategy types utilizing factors such as: Possessions, Pace, Offensive rating, Defensive rating, 3 Point Attempts/efficiency, assist ratios, 
and overall efficiency. Furthermore, these strategies were then utilized to formulate various features, including: Individual Strategy Winrate, Individual Strategy Power, Strategy 
Mismatches, Entropy, and cumulative Strategy winrates.

MACHINE LEARNING MODEL: 
In order to create the strategy features, K-Means Clustering was utilized, grouping similar based outcomes into various strategies.

The Primary Models utilized in this project to train/test the data were RandomForests and XGBOOST, with XGBOOST taken to deployment after fine tuning the hyperparameters of the model. The accuracy metrics 
utilized were AUC and ROC AUC. The primary input features were the strategy metrics and the output features were win probability and classification. 

SHAP EXPLAINABILITY:
Uses SHAP to expose:
1. What features contribute to predictions
2. Direction and magnitude of influence
3. Integrated into Streamlit bar chart visual

PAYOFF-MATRIX SIM:
Head-to-head matchups are used to form a strategy payoff matrix. Simulates how each strategy performs vs. every other. Used to power a simulate endpoint and visualize heatmaps.

NASH EQUILIBRIUM STRATEGY RECOMMENDER:
Computes optimal mixed strategies using nashpy. Uses support enumeration to solve for equilibrium. Returns probabilistic strategy blends for both teams.

API ENDPOINTS:
/predict/: Predict win probability from strategy features

/simulate: Simulate matchup outcome from strategy IDs

/shap/: Return SHAP-based feature contributions

/optimal-strategy/: Return Nash equilibrium strategies

Streamlit Dashboard - Calls FastAPI endpoints via REST API - Simple interface for both technical and non-technical users
1. Interactive sliders for prediction
2. SHAP explainability bar chart
3. Strategy heatmap


Finally, This project was deployed using Docker, and hosted on Google Cloud. 

Tech Stack: 
Python 3.8+
FastAPI for API
Streamlit for dashboard
XGBoost for ML
SHAP for explainability
nashpy for game theory
Jupyter Labs for Data Exploration and Clustering
Docker for deployment
Google Cloud Run for hosting



