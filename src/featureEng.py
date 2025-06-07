import pandas as pd 
from collections import defaultdict
from scipy.stats import entropy
import numpy as np

def load_payoff_matrix(path):
    return pd.read_csv(path).values

def add_strategy_features(df, payoff_matrix): 
    df['STRATEGY_WINRATE'] = df.apply(
        lambda row: payoff_matrix[int(row['TEAM_1_STRATEGY'])][int(row['TEAM_2_STRATEGY'])],
        axis = 1
    )

    #Distance between Strategies(Based on Clusters)
    df['STRATEGY_MISMATCH'] = abs(df['TEAM_1_STRATEGY'] - df['TEAM_2_STRATEGY'])

    #Particular Strategy's Average Winrate
    avg_rate = payoff_matrix.mean(axis=1)
    df['TEAM_1_AWR'] = df['TEAM_1_STRATEGY'].apply(lambda s: avg_rate[int(s)])
    df['TEAM_2_AWR'] = df['TEAM_2_STRATEGY'].apply(lambda s: avg_rate[int(s)])
    df['STRATEGY_AWR'] = df['TEAM_1_AWR'] - df['TEAM_2_AWR']
    return df

def compute_team_entropy(strategy_df):
    history = defaultdict(list)
    for _, row in strategy_df.iterrows():
        team = row['TEAM_ABBREVIATION']
        strat = int(row['STRATEGY_LABEL'])
        history[team].append(strat)
    
    entropy_dict = {}
    for team, strategies in history.items():
        counts = np.bincount(strategies, minlength = 5) + 1e-5
        probs = counts/counts.sum()
        entropy_dict[team] = entropy(probs)
    
    return entropy_dict

def create_feature_set(df, payoff_matrix, strategy_df):
    df = add_strategy_features(df, payoff_matrix)

    team_entropy = compute_team_entropy(strategy_df)
    df['TEAM_1_ENTROPY'] = df['TEAM_1'].map(team_entropy)
    df['TEAM_2_ENTROPY'] = df['TEAM_2'].map(team_entropy)
    df['ENTROPY'] = df['TEAM_1_ENTROPY'] - df['TEAM_2_ENTROPY']

    df['LABEL'] = (df['WINNER'] == df['TEAM_1']).astype(int)
    return df[['STRATEGY_WINRATE', 'STRATEGY_AWR', 'ENTROPY', 'STRATEGY_MISMATCH', 'LABEL']]