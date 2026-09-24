import pandas as pd
from PyQt5.QtWidgets import QMessageBox,QTableWidgetItem

def load_file(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            return "Unsupported file format"
        return df

    except Exception as e:
        return str(e)

def display_data_preview(table_widget, df):
    if df is not None and not isinstance(df, str):
        preview_df = df.head()
        table_widget.setRowCount(preview_df.shape[0])
        table_widget.setColumnCount(preview_df.shape[1])
        table_widget.setHorizontalHeaderLabels(preview_df.columns)
        for i in range(preview_df.shape[0]):
            for j in range(preview_df.shape[1]):
                table_widget.setItem(i, j, QTableWidgetItem(str(preview_df.iat[i, j])))

