import sys

from PyQt5.QtChart import QBarSeries, QChart, QBarSet, QChartView
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QTextEdit, QPushButton, QComboBox, QTabWidget,
                             QFormLayout, QListWidget, QMessageBox, QFileDialog,
                             QStatusBar, QFrame, QSplitter, QDialog, QTableWidget,
                             QTableWidgetItem, QGroupBox, QStackedWidget)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor, QPalette
from predict_star import SentimentAnalyzer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ModernStyle:
    @staticmethod
    def setup(app, dark_mode=True):
        app.setStyle("Fusion")

        if dark_mode:
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.WindowText, Qt.white)
            dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
            dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.ToolTipText, Qt.white)
            dark_palette.setColor(QPalette.Text, Qt.white)
            dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ButtonText, Qt.white)
            dark_palette.setColor(QPalette.BrightText, Qt.red)
            dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
            app.setPalette(dark_palette)
        else:
            app.setPalette(app.style().standardPalette())

    @staticmethod
    def get_stylesheet(dark_mode=True):
        if dark_mode:
            return """
                QMainWindow {
                    background-color: #2D2D2D;
                }
                QTextEdit, QListWidget, QTableWidget {
                    background-color: #353535;
                    color: #FFFFFF;
                    border: 1px solid #444;
                    border-radius: 4px;
                    padding: 5px;
                }
                QPushButton {
                    background-color: #3A3A3A;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #4A4A4A;
                }
                QPushButton:pressed {
                    background-color: #2A2A2A;
                }
                QTabWidget::pane {
                    border: 1px solid #444;
                    border-radius: 4px;
                }
                QTabBar::tab {
                    background: #3A3A3A;
                    color: white;
                    padding: 8px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }
                QTabBar::tab:selected {
                    background: #505050;
                }
                QGroupBox {
                    border: 1px solid #444;
                    border-radius: 4px;
                    margin-top: 10px;
                    padding-top: 15px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                }
            """
        else:
            return """
                QMainWindow {
                    background-color: #F5F5F5;
                }
                QTextEdit, QListWidget, QTableWidget {
                    background-color: white;
                    color: #333333;
                    border: 1px solid #DDD;
                    border-radius: 4px;
                    padding: 5px;
                }
                QPushButton {
                    background-color: #E0E0E0;
                    color: #333333;
                    border: none;
                    border-radius: 4px;
                    padding: 8px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #D0D0D0;
                }
                QPushButton:pressed {
                    background-color: #C0C0C0;
                }
                QTabWidget::pane {
                    border: 1px solid #DDD;
                    border-radius: 4px;
                }
                QTabBar::tab {
                    background: #E0E0E0;
                    color: #333333;
                    padding: 8px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }
                QTabBar::tab:selected {
                    background: white;
                }
                QGroupBox {
                    border: 1px solid #DDD;
                    border-radius: 4px;
                    margin-top: 10px;
                    padding-top: 15px;
                }
            """

# Modern color palette
COLORS = {
    'background': '#1E1E2E', #soft black
    'foreground': '#E8E8E8', #soft white
    'primary': '#7EB3FF', #blue
    'secondary': '#166088',
    'accent': '#7EB3FF', #soft blue
    'positive': '#6EE7B7',  # Mint green
    'neutral': '#FBD38D',  # Warm sand
    'negative': '#FCA5A5',  # Soft coral
    'card': '#313244',  # Card backgrounds
}



class SentimentAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.history_data = []
        self.dark_mode = True
        self.initUI()
        self.load_models()

    def initUI(self):
        self.setWindowTitle("Sentiment Analysis Pro")
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon('../assets/oujda.png'))

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Create splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Input controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # App header
        header = QLabel("Sentiment Analysis Pro")
        header.setFont(QFont('Arial', 16, QFont.Bold))
        left_layout.addWidget(header)

        # Theme toggle
        theme_btn = QPushButton("Toggle Dark/Light")
        theme_btn.clicked.connect(self.toggle_theme)
        left_layout.addWidget(theme_btn)

        # Input tabs
        self.input_tabs = QTabWidget()
        self.input_tabs.setDocumentMode(True)

        # Single review tab
        single_tab = QWidget()
        single_layout = QVBoxLayout(single_tab)

        self.review_input = QTextEdit()
        self.review_input.setPlaceholderText("Enter your review text here...")
        single_layout.addWidget(self.review_input)

        analyze_btn = QPushButton("Analyze")
        analyze_btn.clicked.connect(self.analyze_single_review)
        single_layout.addWidget(analyze_btn)

        # Batch review tab
        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)

        self.batch_input = QTextEdit()
        self.batch_input.setPlaceholderText("Enter multiple reviews, one per line...")
        batch_layout.addWidget(self.batch_input)

        batch_btn = QPushButton("Analyze Batch")
        batch_btn.clicked.connect(self.analyze_batch)
        batch_layout.addWidget(batch_btn)

        self.input_tabs.addTab(single_tab, "Single")
        self.input_tabs.addTab(batch_tab, "Batch")
        left_layout.addWidget(self.input_tabs)

        # File import group
        file_group = QGroupBox("Import Reviews")
        file_layout = QVBoxLayout()

        import_btn = QPushButton("Choose File")
        import_btn.clicked.connect(self.import_reviews)
        file_layout.addWidget(import_btn)

        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # Model status
        self.model_status = QLabel("Model Status: Loading...")
        left_layout.addWidget(self.model_status)

        left_layout.addStretch()

        # Right panel - Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)

        # Results stacked widget
        self.results_stack = QStackedWidget()

        # Single result view
        single_result = QWidget()
        single_result_layout = QVBoxLayout(single_result)

        result_group = QGroupBox("Analysis Results")
        result_layout = QFormLayout()

        self.rating_label = QLabel()
        self.sentiment_label = QLabel()
        self.confidence_label = QLabel()

        result_layout.addRow("Rating:", self.rating_label)
        result_layout.addRow("Sentiment:", self.sentiment_label)
        result_layout.addRow("Confidence:", self.confidence_label)

        result_group.setLayout(result_layout)
        single_result_layout.addWidget(result_group)
        self.results_stack.addWidget(single_result)

        # Batch result view
        batch_result = QWidget()
        batch_result_layout = QVBoxLayout(batch_result)

        stats_group = QGroupBox("Batch Statistics")
        stats_layout = QFormLayout()

        self.avg_rating_label = QLabel()
        self.positive_pct_label = QLabel()
        self.neutral_pct_label = QLabel()
        self.negative_pct_label = QLabel()
        self.total_reviews_label = QLabel()

        stats_layout.addRow("Avg Rating:", self.avg_rating_label)
        stats_layout.addRow("Positive:", self.positive_pct_label)
        stats_layout.addRow("Neutral:", self.neutral_pct_label)
        stats_layout.addRow("Negative:", self.negative_pct_label)
        stats_layout.addRow("Total:", self.total_reviews_label)

        stats_group.setLayout(stats_layout)
        batch_result_layout.addWidget(stats_group)

        # Visualization
        self.figure = Figure(facecolor='#353535' if self.dark_mode else 'white')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        batch_result_layout.addWidget(self.toolbar)
        batch_result_layout.addWidget(self.canvas)

        self.results_stack.addWidget(batch_result)

        right_layout.addWidget(self.results_stack)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])

        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Menu bar
        self.create_menu()

        # Apply initial style
        self.update_style()

    def create_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        export_action = file_menu.addAction("Export Results")
        export_action.triggered.connect(self.export_results)

        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        # View menu
        view_menu = menubar.addMenu("View")

        history_action = view_menu.addAction("History")
        history_action.triggered.connect(self.show_history)

        dashboard_action = view_menu.addAction("Dashboard")
        dashboard_action.triggered.connect(self.show_dashboard)

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.update_style()

    def update_style(self):
        ModernStyle.setup(QApplication.instance(), self.dark_mode)
        self.setStyleSheet(ModernStyle.get_stylesheet(self.dark_mode))

        # Update matplotlib figure background
        self.figure.set_facecolor('#353535' if self.dark_mode else 'white')
        if hasattr(self, 'canvas'):
            self.canvas.draw()
    def load_models(self):
        try:
            self.analyzer = SentimentAnalyzer(
                model_path="../models/sentiment_model_fr.pkl",
                vectorizer_path="../models/tfidf_vectorizer_fr.pkl"
            )
            self.model_status.setText("Model Status: Loaded")
            QTimer.singleShot(3000, lambda: self.status_bar.showMessage("Models loaded successfully"))
        except Exception as e:
            self.model_status.setText("Model Status: Failed")
            QMessageBox.critical(self, "Error", f"Failed to load models: {str(e)}")

    def show_dashboard(self):
        """Create a comprehensive analytics dashboard"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Analytics Dashboard")
        dialog.resize(1000, 700)

        layout = QVBoxLayout()
        tab_widget = QTabWidget()

        # Tab 1: Key Metrics
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout()

        # Create metrics cards
        metrics_grid = QHBoxLayout()

        # Card 1: Sentiment Distribution
        card1 = QFrame()
        card1.setStyleSheet(f"background: {COLORS['card']}; border-radius: 10px;")
        card1_layout = QVBoxLayout(card1)

        # Add a modern chart
        chart = QChart()
        chart.setTitle("Sentiment Distribution")
        chart.setAnimationOptions(QChart.SeriesAnimations)

        # Sample data - replace with your actual data
        series = QBarSeries()
        set_positive = QBarSet("Positive")
        set_neutral = QBarSet("Neutral")
        set_negative = QBarSet("Negative")

        # Get real data from analyzer
        if self.history_data:
            pos = sum(1 for d in self.history_data if d['sentiment'] == 'positive')
            neu = sum(1 for d in self.history_data if d['sentiment'] == 'neutral')
            neg = sum(1 for d in self.history_data if d['sentiment'] == 'negative')
            total = max(1, len(self.history_data))

            set_positive.append(pos / total * 100)
            set_neutral.append(neu / total * 100)
            set_negative.append(neg / total * 100)

        series.append(set_positive)
        series.append(set_neutral)
        series.append(set_negative)

        chart.addSeries(series)
        chart.createDefaultAxes()
        chart.legend().setVisible(True)

        chart_view = QChartView(chart)
        card1_layout.addWidget(chart_view)
        metrics_grid.addWidget(card1)

        # Add more cards similarly...
        metrics_layout.addLayout(metrics_grid)
        metrics_tab.setLayout(metrics_layout)
        tab_widget.addTab(metrics_tab, "Key Metrics")

        # Tab 2: History Log
        history_tab = QWidget()
        history_layout = QVBoxLayout()
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["Date", "Rating", "Sentiment", "Preview"])
        history_layout.addWidget(self.history_table)
        history_tab.setLayout(history_layout)
        tab_widget.addTab(history_tab, "History Log")

        layout.addWidget(tab_widget)
        dialog.setLayout(layout)
        dialog.exec_()

        # NEW FEATURE 2: Export Capability


    def export_results(self):
        """Export analysis results to CSV/Excel"""
        if not self.history_data:
            QMessageBox.warning(self, "No Data", "No analysis history to export")
            return

        options = QFileDialog.Options()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "",
            "CSV Files (*.csv);;Excel Files (*.xlsx)",
            options=options)

        if path:
            try:
                df = pd.DataFrame(self.history_data)
                if path.endswith('.xlsx'):
                    df.to_excel(path, index=False)
                else:
                    df.to_csv(path, index=False)
                QMessageBox.information(self, "Success", f"Exported {len(df)} records")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def _update_result_display(self, result):
        """Update the results display with analysis results"""
        try:
            # Update rating display
            self.rating_label.setText(f"{result['rating']:.1f} ★")

            # Update sentiment with color coding
            sentiment = result['sentiment'].capitalize()
            self.sentiment_label.setText(sentiment)

            # Set sentiment color
            sentiment_color = {
                'positive': COLORS['positive'],
                'neutral': COLORS['neutral'],
                'negative': COLORS['negative']
            }.get(result['sentiment'], COLORS['foreground'])

            self.sentiment_label.setStyleSheet(f"color: {sentiment_color}; font-weight: bold;")

            # Update confidence
            self.confidence_label.setText(f"{result['confidence'] * 100:.1f}%")

            # Create star rating visualization
            stars = ""
            full_stars = int(result['rating'])
            half_star = (result['rating'] - full_stars) >= 0.5

            stars = "★" * full_stars
            if half_star:
                stars += "½"
            stars += "☆" * (5 - full_stars - (1 if half_star else 0))

            self.rating_label.setText(f"{stars} ({result['rating']:.1f})")

        except Exception as e:
            logger.error(f"Error updating display: {str(e)}")
            QMessageBox.warning(self, "Display Error", f"Could not update results: {str(e)}")

    def analyze_single_review(self):
        try:
            text = self.review_input.toPlainText().strip()
            if not text:
                QMessageBox.warning(self, "Warning", "Please enter some text to analyze!")
                return

            result = self.analyzer.analyze_sentiment(text)

            # Store in history
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rating": result['rating'],
                "sentiment": result['sentiment'],
                "text": text[:100] + "..." if len(text) > 100 else text,
                "confidence": result['confidence']
            }
            self.history_data.append(record)

            # Update UI
            self.rating_label.setText(f"{result['rating']:.1f} ★")
            self.sentiment_label.setText(result['sentiment'].capitalize())
            self.confidence_label.setText(f"{result['confidence']:.1%}")

            # Set sentiment color
            color = {
                'positive': '#6EE7B7',
                'neutral': '#FBD38D',
                'negative': '#FCA5A5'
            }.get(result['sentiment'], 'white')
            self.sentiment_label.setStyleSheet(f"color: {color}; font-weight: bold;")

            self.results_stack.setCurrentIndex(0)
            self.status_bar.showMessage("Analysis completed", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")

    def show_history(self):
        """Display analysis history in a table"""
        if not self.history_data:
            QMessageBox.information(self, "History", "No analysis history yet")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Analysis History")
        dialog.resize(800, 600)

        layout = QVBoxLayout()
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Timestamp", "Rating", "Sentiment", "Confidence", "Text Preview"])

        table.setRowCount(len(self.history_data))
        for i, record in enumerate(self.history_data):
            table.setItem(i, 0, QTableWidgetItem(record['timestamp']))
            table.setItem(i, 1, QTableWidgetItem(str(record['rating'])))
            table.setItem(i, 2, QTableWidgetItem(record['sentiment'].capitalize()))
            table.setItem(i, 3, QTableWidgetItem(f"{record['confidence']:.1%}"))
            table.setItem(i, 4, QTableWidgetItem(record['text']))

        layout.addWidget(table)
        dialog.setLayout(layout)
        dialog.exec_()

    def analyze_batch(self):
        """Analyze multiple reviews at once"""
        if not self.analyzer:
            QMessageBox.warning(self, "Warning", "Models not loaded yet!")
            return

        text = self.batch_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Warning", "Please enter some text to analyze!")
            return

        reviews = [line.strip() for line in text.split('\n') if line.strip()]

        try:
            stats = self.analyzer.get_sentiment_stats(reviews)

            # Update stats
            self.avg_rating_label.setText(f"{stats['avg_rating']:.2f} ★")
            self.positive_pct_label.setText(f"{stats['positive_pct']:.1f}%")
            self.neutral_pct_label.setText(f"{stats['neutral_pct']:.1f}%")
            self.negative_pct_label.setText(f"{stats['negative_pct']:.1f}%")
            self.total_reviews_label.setText(str(stats['total_reviews']))

            # Update visualization
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # Pie chart
            labels = ['Positive', 'Neutral', 'Negative']
            sizes = [stats['positive_pct'], stats['neutral_pct'], stats['negative_pct']]
            colors = [COLORS['positive'], COLORS['neutral'], COLORS['negative']]

            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, textprops={'color': 'white'})
            ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
            ax.set_title('Sentiment Distribution', color='white')

            self.figure.patch.set_facecolor(COLORS['card'])
            self.canvas.draw()

            # Update review list
            self.review_list.clear()
            for i, review in enumerate(reviews[:50]):  # Limit to 50 for performance
                try:
                    result = self.analyzer.analyze_sentiment(review)
                    item_text = f"{result['rating']:.1f}★ - {result['sentiment'].capitalize()} - {review[:50]}..."
                    self.review_list.addItem(item_text)
                except:
                    continue

            self.results_tabs.setCurrentIndex(1)
            self.status_bar.showMessage(f"Analyzed {len(reviews)} reviews", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Batch analysis failed: {str(e)}")

    def import_reviews(self):
        """Import reviews from a text file"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Review File", "",
            "Text Files (*.txt);;All Files (*)",
            options=options)

        if file_name:
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.batch_input.setPlainText(content)
                    self.results_tabs.setCurrentIndex(2)  # Switch to batch tab
                    self.status_bar.showMessage(f"Loaded reviews from {file_name}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Apply modern style
    ModernStyle.setup(app, dark_mode=True)

    window = SentimentAnalysisApp()
    window.show()
    sys.exit(app.exec_())