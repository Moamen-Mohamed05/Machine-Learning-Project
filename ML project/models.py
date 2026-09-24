import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# 
from sklearn.metrics import accuracy_score, confusion_matrix

def _preprocess(df: pd.DataFrame, target_col: str, scale: bool = False):
    df = df.dropna(subset=[target_col]).copy()

    # أعمدة يتم حذفها تلقائياً
    auto_drop = []
    for col in df.columns:
        if col == target_col:
            continue
        # شيل ID columns
        if col.lower() in ['id', 'index', 'unnamed: 0']:
            auto_drop.append(col)
            continue
        # شيل الأعمدة النصية اللي فيها قيم unique كتير (زي الأسماء)
        if df[col].dtype == object:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.5:
                auto_drop.append(col)
                continue
        # شيل الأعمدة اللي فيها نفس القيمة في كل الصفوف
        if df[col].nunique() <= 1:
            auto_drop.append(col)

    cols_to_drop = [target_col] + auto_drop
    X = df.drop(columns=cols_to_drop)
    y = df[target_col].copy()

    if y.dtype == object or str(y.dtype) == "category":
        y = LabelEncoder().fit_transform(y)

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    if scale:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    return X_train, X_test, y_train, y_test
def train_logistic_regression(df: pd.DataFrame, target_col: str) -> dict:
    X_train, X_test, y_train, y_test = _preprocess(df, target_col, scale=False)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "model": model,
        "accuracy": accuracy,
        "confusion_matrix": cm,
    }

def train_decision_tree(df: pd.DataFrame, target_col: str) -> dict:
    X_train, X_test, y_train, y_test = _preprocess(df, target_col, scale=False)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "model": model,
        "accuracy": accuracy,
        "confusion_matrix": cm,
    }

def train_knn(df: pd.DataFrame, target_col: str, n_neighbors: int = 5) -> dict:
    X_train, X_test, y_train, y_test = _preprocess(df, target_col, scale=True)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "model": model,
        "accuracy": accuracy,
        "confusion_matrix": cm,
    }
    