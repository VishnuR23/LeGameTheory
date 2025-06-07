import pandas as pd 
import numpy as np 
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from  sklearn.metrics import accuracy_score, roc_auc_score
import optuna 
import shap
from featureEng import load_payoff_matrix, create_feature_set
import joblib

matchups = pd.read_csv('Data/matchupsActual.csv')
payoffs = load_payoff_matrix('Data/payoff_matrix.csv')
games = pd.read_csv('Data/Strategy_labeled_games.csv')

df = create_feature_set(matchups, payoffs, games)

x = df.drop(columns = 'LABEL')
y = df['LABEL']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)

def model(trial): 
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.28),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 3.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1), 
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
    }

    model = xgb.XGBClassifier(use_label_encoder = False, eval_metric = 'logloss', **params)
    score = cross_val_score(model, x_train, y_train, scoring = 'roc_auc', cv = 3).mean()
    return score 

optimized = optuna.create_study(direction = 'maximize')
optimized.optimize(model, n_trials = 30)

print(optimized.best_params)


best_params = optimized.best_params
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **best_params)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

joblib.dump(model, 'xgboost_model.pkl')

