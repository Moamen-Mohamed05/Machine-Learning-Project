import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

sns.set_theme(style="darkgrid")

def plot_line(fig, ax, df):
    try:
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha='center')
            return

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            ax.text(0.5, 0.5, "Need at least 2 numeric columns", transform=ax.transAxes, ha='center')
            return

        x_col = numeric_cols[0]
        y_col = numeric_cols[1]

        imputer = SimpleImputer(strategy='mean')
        
        X_vals = df[x_col].values.reshape(-1, 1)
        y_vals = df[y_col].values.reshape(-1, 1)
        
        X_clean = imputer.fit_transform(X_vals).flatten()
        y_clean = imputer.fit_transform(y_vals).flatten()

        sns.lineplot(x=X_clean, y=y_clean, marker='o', linestyle='-', linewidth=2, markersize=6, color='b', ax=ax)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Line Plot: {y_col} vs {x_col}")
        ax.grid(True, linestyle='--', alpha=0.7)

    except Exception as e:
        ax.text(0.5, 0.5, f"Plot error: {str(e)}", transform=ax.transAxes, ha='center')

def plot_scatter(fig, ax, df):
    try:
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha='center')
            return

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            ax.text(0.5, 0.5, "Need at least 2 numeric columns", transform=ax.transAxes, ha='center')
            return

        x_col = numeric_cols[0]
        y_col = numeric_cols[1]

        imputer = SimpleImputer(strategy='median')
        
        X_vals = df[x_col].values.reshape(-1, 1)
        y_vals = df[y_col].values.reshape(-1, 1)
        
        X_clean = imputer.fit_transform(X_vals).flatten()
        y_clean = imputer.fit_transform(y_vals).flatten()

        sns.scatterplot(x=X_clean, y=y_clean, color='green', alpha=0.6, edgecolor='black', s=50, ax=ax)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Scatter Plot: {y_col} vs {x_col}")
        ax.grid(True, linestyle=':', alpha=0.6)

    except Exception as e:
        ax.text(0.5, 0.5, f"Plot error: {str(e)}", transform=ax.transAxes, ha='center')







import pandas as pd
import matplotlib.pyplot as plt


def plot_box(fig, ax, df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        raise ValueError("plot_box: the DataFrame contains no numeric columns to plot.")

    ax.clear()

    numeric_df.boxplot(
        ax=ax,
        patch_artist=True,
        notch=False,
        vert=True,
        grid=True,
        boxprops=dict(facecolor="#4C72B0", color="#2c3e50"),
        medianprops=dict(color="#e74c3c", linewidth=2),
        whiskerprops=dict(color="#2c3e50", linewidth=1.5),
        capprops=dict(color="#2c3e50", linewidth=1.5),
        flierprops=dict(
            marker="o",
            markerfacecolor="#e74c3c",
            markersize=4,
            linestyle="none",
            alpha=0.6,
        ),
    )

    ax.set_title("Box Plot — Numeric Feature Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Features", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)

    if len(numeric_df.columns) > 6:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    ax.tick_params(axis="both", labelsize=9)

    fig.tight_layout()
    fig.canvas.draw_idle()


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        raise ValueError("detect_outliers: the DataFrame contains no numeric columns to evaluate.")

    q1 = numeric_df.quantile(0.25)
    q3 = numeric_df.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (numeric_df < lower_bound) | (numeric_df > upper_bound)
    rows_with_outliers = outlier_mask.any(axis=1)

    return df[rows_with_outliers].copy()
