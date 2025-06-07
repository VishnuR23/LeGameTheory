import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from featureEng import load_payoff_matrix, create_feature_set

matchups = pd.read_csv('Data/matchupsActual.csv')
payoffs = load_payoff_matrix('Data/payoff_matrix.csv')
games = pd.read_csv('Data/Strategy_labeled_games.csv')

df = create_feature_set(matchups, payoffs, games)
x = df.drop(columns = 'LABEL')
y = df['LABEL']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = RandomForestClassifier(random_state = 42)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print('Accuracy:', accuracy_score(Y_test, y_pred))
print('ROC-AUC:', roc_auc_score(Y_test, y_proba))