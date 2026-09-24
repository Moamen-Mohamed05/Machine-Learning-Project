import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QTabWidget,
    QTableWidget, QTableWidgetItem,
    QGroupBox, QStatusBar, QSplitter,
    QScrollArea, QMessageBox, QInputDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from visualization import plot_line, plot_scatter, plot_box, detect_outliers
from preprocessing import handle_missing_value, encode_features
from models import train_logistic_regression, train_decision_tree, train_knn
from Scale import scale_features


class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def clear_canvas(self):
        self.axes.clear()
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_df = None
        self.processed_df = None
        self.file_path = ""

        self.setWindowTitle("ML Desktop App")
        self.setGeometry(100, 100, 1200, 750)
        self.setMinimumSize(900, 600)

        self._build_ui()
        self._apply_styles()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("Ready - Load a file to begin.")

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)

        main_layout.addWidget(self._build_upload_section())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_buttons_panel())
        splitter.addWidget(self._build_display_panel())
        splitter.setSizes([280, 920])
        main_layout.addWidget(splitter)

    def _build_upload_section(self):
        group = QGroupBox("Data Upload")
        layout = QHBoxLayout(group)

        self.upload_btn = QPushButton("Select CSV / Excel")
        self.upload_btn.setFixedHeight(40)
        self.upload_btn.clicked.connect(self.handle_upload)

        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: gray; font-style: italic;")

        self.preview_btn = QPushButton("Preview Data")
        self.preview_btn.setFixedHeight(40)
        self.preview_btn.clicked.connect(self.show_data_preview)
        self.preview_btn.setEnabled(False)

        layout.addWidget(self.upload_btn)
        layout.addWidget(self.file_label, stretch=1)
        layout.addWidget(self.preview_btn)

        return group

    def _build_buttons_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout(viz_group)
        self.btn_line = QPushButton("Line Plot")
        self.btn_scatter = QPushButton("Scatter Plot")
        self.btn_box = QPushButton("Box Plot")
        self.btn_outliers = QPushButton("Detect Outliers")

        for btn in [self.btn_line, self.btn_scatter, self.btn_box, self.btn_outliers]:
            btn.setFixedHeight(35)
            btn.setEnabled(False)
            viz_layout.addWidget(btn)

        self.btn_line.clicked.connect(self.handle_line_plot)
        self.btn_scatter.clicked.connect(self.handle_scatter_plot)
        self.btn_box.clicked.connect(self.handle_box_plot)
        self.btn_outliers.clicked.connect(self.handle_outliers)

        prep_group = QGroupBox("Preprocessing")
        prep_layout = QVBoxLayout(prep_group)
        self.btn_missing = QPushButton("Handle Missing")
        self.btn_encode = QPushButton("Encoding")
        self.btn_scale = QPushButton("Scaling")
        self.btn_compare = QPushButton("Compare Before / After")
        self.btn_save_processed = QPushButton("Save Processed Data")

        for btn in [self.btn_missing, self.btn_encode, self.btn_scale,
                    self.btn_compare, self.btn_save_processed]:
            btn.setFixedHeight(35)
            btn.setEnabled(False)
            prep_layout.addWidget(btn)

        self.btn_missing.clicked.connect(self.handle_missing)
        self.btn_encode.clicked.connect(self.handle_encoding)
        self.btn_scale.clicked.connect(self.handle_scaling)
        self.btn_compare.clicked.connect(self.handle_compare)
        self.btn_save_processed.clicked.connect(self.handle_save_processed)

        model_group = QGroupBox("ML Models")
        model_layout = QVBoxLayout(model_group)
        self.btn_logreg = QPushButton("Logistic Regression")
        self.btn_tree = QPushButton("Decision Tree")
        self.btn_knn = QPushButton("KNN")

        for btn in [self.btn_logreg, self.btn_tree, self.btn_knn]:
            btn.setFixedHeight(35)
            btn.setEnabled(False)
            model_layout.addWidget(btn)

        self.btn_logreg.clicked.connect(self.handle_logistic_regression)
        self.btn_tree.clicked.connect(self.handle_decision_tree)
        self.btn_knn.clicked.connect(self.handle_knn)

        layout.addWidget(viz_group)
        layout.addWidget(prep_group)
        layout.addWidget(model_group)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(280)

        return scroll

    def _build_display_panel(self):
        self.tabs = QTabWidget()

        self.table_widget = QTableWidget()
        self.tabs.addTab(self.table_widget, "Original Data")

        self.processed_table = QTableWidget()
        self.tabs.addTab(self.processed_table, "Processed Data")

        self.outliers_table = QTableWidget()
        self.tabs.addTab(self.outliers_table, "Outliers")

        self.canvas = MatplotlibCanvas()
        self.tabs.addTab(self.canvas, "Plots")

        self.results_label = QLabel("No results yet")
        self.results_label.setAlignment(Qt.AlignCenter)
        self.results_label.setFont(QFont("Arial", 12))
        self.tabs.addTab(self.results_label, "Results")

        return self.tabs

    def _enable_all_buttons(self):
        buttons = [
            self.preview_btn, self.btn_line, self.btn_scatter, self.btn_box,
            self.btn_outliers, self.btn_missing, self.btn_encode, self.btn_scale,
            self.btn_logreg, self.btn_tree, self.btn_knn
        ]
        for btn in buttons:
            btn.setEnabled(True)

    def _enable_compare_buttons(self):
        self.btn_compare.setEnabled(True)
        self.btn_save_processed.setEnabled(True)

    def update_status(self, message: str):
        self.status_bar.showMessage(message)

    def _fill_table(self, table_widget: QTableWidget, df: pd.DataFrame):
        table_widget.setRowCount(min(100, len(df)))
        table_widget.setColumnCount(len(df.columns))
        table_widget.setHorizontalHeaderLabels(df.columns.tolist())

        for i in range(min(100, len(df))):
            for j in range(len(df.columns)):
                table_widget.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))

        table_widget.resizeColumnsToContents()

    def handle_upload(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "", "Data Files (*.csv *.xlsx *.xls);;All Files (*)"
        )

        if not file_path:
            return

        self.file_path = file_path
        self.update_status("Loading file...")

        try:
            if file_path.endswith('.csv'):
                self.current_df = pd.read_csv(file_path)
            else:
                self.current_df = pd.read_excel(file_path)

            self.processed_df = None

            file_name = file_path.split("/")[-1]
            self.file_label.setText(
                f"{file_name} | {self.current_df.shape[0]} rows, {self.current_df.shape[1]} cols"
            )
            self.file_label.setStyleSheet("color: green; font-weight: bold;")

            self._enable_all_buttons()
            self.show_data_preview()
            self.update_status(f"Loaded {file_name}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
            self.update_status("Error loading file")

    def show_data_preview(self):
        if self.current_df is None:
            return
        self._fill_table(self.table_widget, self.current_df)
        self.tabs.setCurrentIndex(0)

    def _refresh_processed_tab(self):
        if self.processed_df is None:
            return
        self._fill_table(self.processed_table, self.processed_df)
        self._enable_compare_buttons()

    def handle_line_plot(self):
        if self.current_df is None:
            QMessageBox.warning(self, "Warning", "Please upload a file first!")
            return
        self.update_status("Rendering Line Plot...")
        self.canvas.axes.clear()
        try:
            plot_line(self.canvas.fig, self.canvas.axes, self.current_df)
            self.canvas.draw()
            self.tabs.setCurrentIndex(3)
            self.update_status("Line Plot rendered successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.update_status("Failed to render Line Plot")

    def handle_scatter_plot(self):
        if self.current_df is None:
            QMessageBox.warning(self, "Warning", "Please upload a file first!")
            return
        self.update_status("Rendering Scatter Plot...")
        self.canvas.axes.clear()
        try:
            plot_scatter(self.canvas.fig, self.canvas.axes, self.current_df)
            self.canvas.draw()
            self.tabs.setCurrentIndex(3)
            self.update_status("Scatter Plot rendered successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.update_status("Failed to render Scatter Plot")

    def handle_box_plot(self):
        if self.current_df is None:
            QMessageBox.warning(self, "Warning", "Please upload a file first!")
            return
        self.canvas.axes.clear()
        try:
            plot_box(self.canvas.fig, self.canvas.axes, self.current_df)
            self.canvas.draw()
            self.tabs.setCurrentIndex(3)
            self.update_status("Box Plot rendered successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def handle_outliers(self):
        if self.current_df is None:
            QMessageBox.warning(self, "Warning", "Please upload a file first!")
            return
        try:
            outliers_df = detect_outliers(self.current_df)

            if outliers_df.empty:
                QMessageBox.information(self, "Outliers", "No outliers detected in the data.")
                self.update_status("No outliers found")
                return

            self._fill_table(self.outliers_table, outliers_df)
            self.tabs.setCurrentIndex(2)
            self.update_status(f"Outliers detected: {len(outliers_df)} rows")
            QMessageBox.information(self, "Outliers",
                f"{len(outliers_df)} rows with outliers found.\nCheck the 'Outliers' tab.")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def handle_missing(self):
        if self.current_df is None:
            QMessageBox.warning(self, "Warning", "Please upload a file first!")
            return
        try:
            missing_before = int(self.current_df.isnull().sum().sum())
            self.processed_df = handle_missing_value(self.current_df)
            self._refresh_processed_tab()
            QMessageBox.information(self, "Missing Values",
                f"Done — {missing_before} values were imputed.\nCheck 'Processed Data' tab to see the result.")
            self.update_status("Missing values handled successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def handle_encoding(self):
        if self.current_df is None:
            QMessageBox.warning(self, "Warning", "Please upload a file first!")
            return
        try:
            source_df = self.processed_df if self.processed_df is not None else self.current_df
            self.processed_df = encode_features(source_df)
            self._refresh_processed_tab()
            QMessageBox.information(self, "Encoding",
                "Encoding completed.\nCheck 'Processed Data' tab to see the result.")
            self.update_status("Encoding done successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def handle_scaling(self):
        if self.current_df is None:
            QMessageBox.warning(self, "Warning", "Please upload a file first!")
            return
        try:
            source_df = self.processed_df if self.processed_df is not None else self.current_df
            self.processed_df = scale_features(source_df, scaling_type="standard")
            self._refresh_processed_tab()
            QMessageBox.information(self, "Scaling",
                "Standard scaling applied.\nCheck 'Processed Data' tab to see the result.")
            self.update_status("Scaling done successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def handle_compare(self):
        if self.processed_df is None:
            QMessageBox.warning(self, "Warning", "No processed data yet. Run a preprocessing step first.")
            return

        compare_window = QMainWindow(self)
        compare_window.setWindowTitle("Before vs After Preprocessing")
        compare_window.setGeometry(150, 150, 1400, 600)

        central = QWidget()
        compare_window.setCentralWidget(central)
        layout = QHBoxLayout(central)

        before_group = QGroupBox(f"Before  ({self.current_df.shape[0]} rows x {self.current_df.shape[1]} cols)")
        before_layout = QVBoxLayout(before_group)
        before_table = QTableWidget()
        self._fill_table(before_table, self.current_df)
        before_layout.addWidget(before_table)

        after_group = QGroupBox(f"After  ({self.processed_df.shape[0]} rows x {self.processed_df.shape[1]} cols)")
        after_layout = QVBoxLayout(after_group)
        after_table = QTableWidget()
        self._fill_table(after_table, self.processed_df)
        after_layout.addWidget(after_table)

        layout.addWidget(before_group)
        layout.addWidget(after_group)

        compare_window.show()
        self.update_status("Compare window opened")

    def handle_save_processed(self):
        if self.processed_df is None:
            QMessageBox.warning(self, "Warning", "No processed data to save.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Data", "processed_data.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )

        if not save_path:
            return

        try:
            if save_path.endswith('.xlsx'):
                self.processed_df.to_excel(save_path, index=False)
            else:
                self.processed_df.to_csv(save_path, index=False)

            QMessageBox.information(self, "Saved",
                f"Processed data saved to:\n{save_path}\n\nOriginal data was not affected.")
            self.update_status(f"Processed data saved to {save_path.split('/')[-1]}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")

    def _ask_target_column(self, df: pd.DataFrame):
        valid_cols = []
        for col in df.columns:
            if col.lower() in ['id', 'index', 'unnamed: 0']:
                continue
            if df[col].nunique() <= 1:
                continue
            if df[col].dtype in [np.int64, np.float64]:
                if df[col].dtype in [np.int64, np.float64]:
                    unique_ratio = df[col].nunique() / len(df)

                    if unique_ratio > 0.05:
                        continue
            if df[col].dtype == object:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.5:
                    continue
            valid_cols.append(col)

        if not valid_cols:
            QMessageBox.warning(self, "Warning", "No valid target columns found!")
            return None

        col, ok = QInputDialog.getItem(
            self, "Select Target Column",
            "Choose the target column:", valid_cols, 0, False
        )
        if ok and col:
            return col
        return None

    def handle_logistic_regression(self):
        if self.current_df is None:
            QMessageBox.warning(self, "Warning", "Please upload a file first!")
            return
        target_col = self._ask_target_column(self.current_df)
        if not target_col:
            return
        try:
            result = train_logistic_regression(self.current_df, target_col)
            self.show_model_results("Logistic Regression", result)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def handle_decision_tree(self):
        if self.current_df is None:
            QMessageBox.warning(self, "Warning", "Please upload a file first!")
            return
        target_col = self._ask_target_column(self.current_df)
        if not target_col:
            return
        try:
            result = train_decision_tree(self.current_df, target_col)
            self.show_model_results("Decision Tree", result)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def handle_knn(self):
        if self.current_df is None:
            QMessageBox.warning(self, "Warning", "Please upload a file first!")
            return
        target_col = self._ask_target_column(self.current_df)
        if not target_col:
            return
        try:
            result = train_knn(self.current_df, target_col)
            self.show_model_results("KNN", result)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def show_model_results(self, model_name: str, result: dict):
        cm = result.get("confusion_matrix")
        cm_str = np.array2string(cm) if cm is not None else "N/A"
        text = (
            f"Model: {model_name}\n\n"
            f"Accuracy: {result.get('accuracy', 'N/A'):.4f}\n\n"
            f"Confusion Matrix:\n{cm_str}"
        )
        self.results_label.setText(text)
        self.tabs.setCurrentIndex(4)
        self.update_status(f"{model_name} completed — Accuracy: {result.get('accuracy', 0):.4f}")

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #B0BEC5; color: #78909C; }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 2px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QTabWidget::pane { border: 1px solid #ddd; border-radius: 5px; }
            QStatusBar { background-color: #263238; color: white; font-size: 12px; padding: 3px; }
        """)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()