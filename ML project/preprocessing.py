import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer    

def impute_seasonal_zeros(df, save_path=None):
    data = df.copy()
    
    seasons = set()
    for col in data.columns:
        if '/' in col:
            season_suffix = col.split('_')[-1]
            seasons.add(season_suffix)
    
    for season in seasons:
        matches_col = f'Matches_{season}'
        if matches_col in data.columns:
            seasonal_cols = [c for c in data.columns if c.endswith(season)]
            mask = (data[matches_col] == 0)
            data.loc[mask, seasonal_cols] = 0
   
    return data

def encode_positions(df, position_col='Position', save_path=None):
    data = df.copy()
    if position_col in data.columns:
        data[position_col] = pd.Categorical(data[position_col]).codes
    
    return data


def knn_impute(df):
    data = df.copy()
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    imputer = KNNImputer(n_neighbors=3, weights='uniform')
    
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
    
    return data
def encode_features(df):
    data = df.copy()
    data = encode_positions(data)
    
    return data

def handle_missing_value(df):
    data = df.copy()
    
    if 'Player_Name' in data.columns:
        data = data.dropna(subset=['Player_Name']).copy()
        
    data = impute_seasonal_zeros(data)
    data = knn_impute(data)
    
    return data