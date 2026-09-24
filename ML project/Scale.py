import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, Normalizer

def scale_features(df, scaling_type="standard", apply_poly=False, degree=2, normalize=False):
    if df is None or df.empty:
        raise ValueError("DataFrame is empty")
    num_cols = df.select_dtypes(include='number').columns
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found")
    X = df[num_cols]
    #  Scaling 
    if scaling_type == "standard":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaling_type == "minmax":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        raise ValueError("Invalid scaling type")
    # DataFrame 
    X_scaled = pd.DataFrame(X_scaled, columns=num_cols)
    # Polynomial
    if apply_poly:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X_scaled)
        poly_cols = poly.get_feature_names_out(num_cols)
        X_scaled = pd.DataFrame(X_poly, columns=poly_cols)
    # Normalization 
    if normalize:
        normalizer = Normalizer()
        X_norm = normalizer.fit_transform(X_scaled)
        X_scaled = pd.DataFrame(X_norm, columns=X_scaled.columns)

    return X_scaled