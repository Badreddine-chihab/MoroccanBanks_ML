import sys
import os
from PyQt5.QtChart import QChart, QChartView, QPieSeries, QBarSeries, QBarSet, QValueAxis, QBarCategoryAxis
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QTextEdit, QPushButton, QComboBox, QTabWidget, QFormLayout, QListWidget,
                             QMessageBox, QFileDialog, QStatusBar, QFrame, QSplitter, QDialog,
                             QTableWidget, QTableWidgetItem, QGroupBox, QStackedWidget, QToolBar,
                             QAction, QSizePolicy, QSpacerItem, QScrollArea, QGridLayout)
from PyQt5.QtCore import Qt, QSize, QTimer, QMargins, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor, QPalette, QKeySequence, QPainter, QBrush
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from datetime import datetime
import pandas as pd
import numpy as np
from predict_star import SentimentAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModernStyle:
    """Modern UI styling for the application"""

    @staticmethod
    def setup(app, dark_mode=True):
        app.setStyle("Fusion")
        palette = QPalette()
        if dark_mode:
            palette.setColor(QPalette.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(35, 35, 35))
            palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.ToolTipText, Qt.white)
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ButtonText, Qt.white)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(42, 130, 218))
            palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.HighlightedText, Qt.black)
        else:
            palette.setColor(QPalette.Window, QColor(240, 240, 240))
            palette.setColor(QPalette.WindowText, Qt.black)
            palette.setColor(QPalette.Base, Qt.white)
            palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.ToolTipText, Qt.black)
            palette.setColor(QPalette.Text, Qt.black)
            palette.setColor(QPalette.Button, QColor(240, 240, 240))
            palette.setColor(QPalette.ButtonText, Qt.black)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(0, 122, 204))
            palette.setColor(QPalette.Highlight, QColor(0, 122, 204))
            palette.setColor(QPalette.HighlightedText, Qt.white)
        app.setPalette(palette)


class AnalysisResultWidget(QWidget):
    """Widget to display single analysis results"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Rating card
        self.rating_card = self.create_card("⭐ Rating", "0.0", "Enter text to analyze")

        # Sentiment card
        self.sentiment_card = self.create_card("😊 Sentiment", "Neutral", "Waiting for input...")

        layout.addWidget(self.rating_card)
        layout.addWidget(self.sentiment_card)
        layout.addStretch()

    def update_results(self, result):
        """Update results display"""
        # Update rating card
        rating_label = self.rating_card.findChild(QLabel, "value_label")
        rating_label.setText(f"{result['rating']:.1f}")

        # Update sentiment card
        sentiment_label = self.sentiment_card.findChild(QLabel, "value_label")
        sentiment_label.setText(result['sentiment'].capitalize())

        # Set color based on sentiment
        color = "#6EE7B7" if result['sentiment'] == 'positive' else "#FCA5A5"
        sentiment_label.setStyleSheet(f"color: {color};")

    def create_card(self, title, value, placeholder=""):
        """Create a styled result card"""
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setStyleSheet("""
            QFrame {
                background: #353535;
                border-radius: 12px;
                padding: 15px;
            }
            QLabel {
                font-size: 14px;
                color: #AAAAAA;
            }
            .value {
                font-size: 28px;
                font-weight: bold;
                color: white;
                margin-top: 5px;
            }
        """)

        layout = QVBoxLayout(card)
        title_label = QLabel(title)

        value_label = QLabel(value)
        value_label.setObjectName("value_label")
        value_label.setProperty("class", "value")
        value_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(title_label, alignment=Qt.AlignCenter)
        layout.addWidget(value_label, alignment=Qt.AlignCenter)

        return card


class BatchAnalysisWidget(QWidget):
    """Widget for batch analysis results"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Stats cards
        cards_layout = QHBoxLayout()
        self.avg_rating_card = self.create_stat_card("Average Rating", "0.0")
        self.positive_card = self.create_stat_card("Positive", "0%")
        self.negative_card = self.create_stat_card("Negative", "0%")

        cards_layout.addWidget(self.avg_rating_card)
        cards_layout.addWidget(self.positive_card)
        cards_layout.addWidget(self.negative_card)

        layout.addLayout(cards_layout)

        # Visualization tabs
        self.viz_tabs = QTabWidget()
        self.setup_visualization_tabs()
        layout.addWidget(self.viz_tabs)

    def setup_visualization_tabs(self):
        """Setup visualization tabs"""
        # Pie chart tab
        pie_tab = QWidget()
        pie_layout = QVBoxLayout(pie_tab)
        self.pie_figure = plt.figure(facecolor='#353535')
        self.pie_canvas = FigureCanvas(self.pie_figure)
        self.pie_toolbar = NavigationToolbar(self.pie_canvas, self)
        pie_layout.addWidget(self.pie_toolbar)
        pie_layout.addWidget(self.pie_canvas)
        self.viz_tabs.addTab(pie_tab, "Distribution")

    def update_results(self, stats):
        """Update batch results display"""
        # Update cards
        self.avg_rating_card.findChild(QLabel, "value_label").setText(f"{stats['avg_rating']:.1f}")
        self.positive_card.findChild(QLabel, "value_label").setText(f"{stats['positive_pct']:.1f}%")
        self.negative_card.findChild(QLabel, "value_label").setText(f"{stats['negative_pct']:.1f}%")

        # Update pie chart
        self.update_pie_chart(stats)

    def update_pie_chart(self, stats):
        """Update the pie chart visualization"""
        self.pie_figure.clear()
        ax = self.pie_figure.add_subplot(111)

        labels = ['Positive', 'Negative']
        sizes = [stats['positive_pct'], stats['negative_pct']]
        colors = ['#6EE7B7', '#FCA5A5']

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'color': 'white'})
        ax.set_title('Sentiment Distribution', color='white')

        self.pie_canvas.draw()

    def create_stat_card(self, title, value):
        """Create a stat card"""
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setStyleSheet("""
            QFrame {
                background: #353535;
                border-radius: 8px;
                padding: 12px;
            }
            QLabel {
                font-size: 14px;
            }
            .value {
                font-size: 18px;
                font-weight: bold;
            }
        """)

        layout = QVBoxLayout(card)
        title_label = QLabel(title)

        value_label = QLabel(value)
        value_label.setObjectName("value_label")
        value_label.setProperty("class", "value")

        layout.addWidget(title_label)
        layout.addWidget(value_label)

        return card


class SentimentAnalysisApp(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.history = []
        self.dark_mode = True
        self.setup_ui()
        self.load_models()

    def setup_ui(self):
        """Initialize UI components"""
        self.setWindowTitle("Sentiment Analysis")
        self.setGeometry(100, 100, 1000, 700)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Splitter for left/right panels
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Input
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # Input tabs
        self.input_tabs = QTabWidget()
        self.setup_input_tabs()
        left_layout.addWidget(self.input_tabs)

        # Model status
        self.model_status = QLabel("Model Status: Loading...")
        left_layout.addWidget(self.model_status)

        # Right panel - Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Results stack
        self.results_stack = QStackedWidget()
        self.single_result_widget = AnalysisResultWidget()
        self.batch_result_widget = BatchAnalysisWidget()
        self.results_stack.addWidget(self.single_result_widget)
        self.results_stack.addWidget(self.batch_result_widget)
        right_layout.addWidget(self.results_stack)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Menu bar
        self.setup_menu_bar()

    def setup_input_tabs(self):
        """Setup input tabs"""
        # Single review tab
        single_tab = QWidget()
        single_layout = QVBoxLayout(single_tab)

        self.review_input = QTextEdit()
        self.review_input.setPlaceholderText("Enter review text...")
        single_layout.addWidget(self.review_input)

        analyze_btn = QPushButton("Analyze")
        analyze_btn.clicked.connect(self.analyze_single_review)
        single_layout.addWidget(analyze_btn)

        # Batch review tab
        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)

        self.batch_input = QTextEdit()
        self.batch_input.setPlaceholderText("Enter one review per line...")
        batch_layout.addWidget(self.batch_input)

        batch_btn = QPushButton("Analyze Batch")
        batch_btn.clicked.connect(self.analyze_batch)
        batch_layout.addWidget(batch_btn)

        # Add tabs
        self.input_tabs.addTab(single_tab, "Single Review")
        self.input_tabs.addTab(batch_tab, "Batch Analysis")

    def setup_menu_bar(self):
        """Setup menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        export_action = QAction("Export Results", self)
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def load_models(self):
        """Load the sentiment analysis models"""
        try:
            self.analyzer = SentimentAnalyzer(
                model_path="../models/model_star.pkl",
                vectorizer_path="../models/vectorizer.pkl"
            )
            self.model_status.setText("Model Status: Loaded")
            self.status_bar.showMessage("Models loaded successfully", 3000)
        except Exception as e:
            self.model_status.setText("Model Status: Failed to load")
            QMessageBox.critical(self, "Error", f"Failed to load models: {str(e)}")

    def analyze_single_review(self):
        """Analyze a single review"""
        text = self.review_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Warning", "Please enter review text!")
            return

        try:
            result = self.analyzer.analyze_review(text)

            # Store in history
            self.history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "text": text[:100] + "..." if len(text) > 100 else text,
                "rating": result["rating"],
                "sentiment": result["sentiment"],
                "confidence": result["confidence"]
            })

            # Update UI
            self.single_result_widget.update_results(result)
            self.results_stack.setCurrentIndex(0)
            self.status_bar.showMessage("Analysis completed", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")

    def analyze_batch(self):
        """Analyze a batch of reviews"""
        text = self.batch_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Warning", "Please enter reviews to analyze!")
            return

        reviews = [line.strip() for line in text.split('\n') if line.strip()]

        try:
            # Analyze each review
            ratings = []
            sentiments = []

            for review in reviews:
                result = self.analyzer.analyze_review(review)
                ratings.append(result["rating"])
                sentiments.append(result["sentiment"])

                # Store in history
                self.history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "text": review[:100] + "..." if len(review) > 100 else review,
                    "rating": result["rating"],
                    "sentiment": result["sentiment"],
                    "confidence": result["confidence"]
                })

            # Calculate stats
            positive_count = sentiments.count("positive")
            negative_count = sentiments.count("negative")
            total = len(reviews)

            stats = {
                "avg_rating": sum(ratings) / total,
                "positive_pct": (positive_count / total) * 100,
                "negative_pct": (negative_count / total) * 100,
                "total_reviews": total
            }

            # Update UI
            self.batch_result_widget.update_results(stats)
            self.results_stack.setCurrentIndex(1)
            self.status_bar.showMessage(f"Analyzed {total} reviews", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Batch analysis failed: {str(e)}")

    def export_results(self):
        """Export analysis results to CSV"""
        if not self.history:
            QMessageBox.warning(self, "Warning", "No results to export!")
            return

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "CSV Files (*.csv)", options=options)

        if file_name:
            try:
                df = pd.DataFrame(self.history)
                df.to_csv(file_name, index=False)
                self.status_bar.showMessage(f"Results exported to {file_name}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About",
                          "Sentiment Analysis Tool\n\n"
                          "Version 1.0\n"
                          "© 2023")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ModernStyle.setup(app, dark_mode=True)
    window = SentimentAnalysisApp()
    window.show()
    sys.exit(app.exec_())