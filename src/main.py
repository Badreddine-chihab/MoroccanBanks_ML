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
            # Dark palette
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
            # Light palette
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

    @staticmethod
    def get_stylesheet(dark_mode=True):
        if dark_mode:
            return """
                QMainWindow {
                    background-color: #2D2D2D;
                }
                QTextEdit, QListWidget, QTableWidget, QComboBox {
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
                    background: #353535;
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
                    border-bottom: 2px solid #7EB3FF;
                }
                QGroupBox {
                    border: 1px solid #444;
                    border-radius: 4px;
                    margin-top: 10px;
                    padding-top: 15px;
                    color: white;
                }
                QHeaderView::section {
                    background-color: #3A3A3A;
                    color: white;
                    padding: 4px;
                    border: 1px solid #444;
                }
                QScrollBar:vertical {
                    border: 1px solid #444;
                    background: #353535;
                    width: 10px;
                    margin: 0px 0px 0px 0px;
                }
                QScrollBar::handle:vertical {
                    background: #5A5A5A;
                    min-height: 20px;
                }
                QScrollBar::add-line:vertical {
                    border: 1px solid #444;
                    background: #353535;
                    height: 0px;
                    subcontrol-position: bottom;
                    subcontrol-origin: margin;
                }
                QScrollBar::sub-line:vertical {
                    border: 1px solid #444;
                    background: #353535;
                    height: 0px;
                    subcontrol-position: top;
                    subcontrol-origin: margin;
                }
            """
        else:
            return """
                QMainWindow {
                    background-color: #F5F5F5;
                }
                QTextEdit, QListWidget, QTableWidget, QComboBox {
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
                    background: white;
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
                    border-bottom: 2px solid #166088;
                }
                QGroupBox {
                    border: 1px solid #DDD;
                    border-radius: 4px;
                    margin-top: 10px;
                    padding-top: 15px;
                    color: #333333;
                }
                QHeaderView::section {
                    background-color: #E0E0E0;
                    color: #333333;
                    padding: 4px;
                    border: 1px solid #DDD;
                }
            """


class AnalysisResultWidget(QWidget):
    """Widget to display single analysis results"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Mood Ring (centered at top)
        self.mood_ring = MoodRingWidget()
        layout.addWidget(self.mood_ring, 0, Qt.AlignCenter)

        # Results grid (cards)
        cards_grid = QGridLayout()
        cards_grid.setSpacing(15)

        # Rating card
        self.rating_card = self.create_card(
            "⭐ Rating",
            "0.0",
            "Enter text to analyze"
        )
        cards_grid.addWidget(self.rating_card, 0, 0)

        # Sentiment card
        self.sentiment_card = self.create_card(
            "😊 Sentiment",
            "Neutral",
            "Waiting for input..."
        )
        cards_grid.addWidget(self.sentiment_card, 0, 1)

        layout.addLayout(cards_grid)
        layout.addStretch()  # Push content up


    def update_results(self, result):
        """Update all result elements"""
        # Update mood ring
        self.mood_ring.update_mood(result['sentiment'], result['confidence'])

        # Update rating card
        rating_label = self.rating_card.findChildren(QLabel)[1]
        stars = "★" * int(result['rating'])
        if result['rating'] % 1 >= 0.5:
            stars += "½"
        stars += "☆" * (5 - int(np.ceil(result['rating'])))
        rating_label.setText(f"{result['rating']:.1f}\n{stars}")
        rating_label.setStyleSheet("color: #FFD700;")  # Gold color for stars

        # Update sentiment card
        sentiment_label = self.sentiment_card.findChildren(QLabel)[1]
        sentiment_label.setText(result['sentiment'].capitalize())

        # Set sentiment color
        color = {
            'positive': '#6EE7B7',  # Mint green
            'neutral': '#FBD38D',  # Light orange
            'negative': '#FCA5A5'  # Light red
        }.get(result['sentiment'], 'white')
        sentiment_label.setStyleSheet(f"color: {color};")



        # Remove placeholder text
        for card in [self.rating_card, self.sentiment_card]:
            subtext = card.findChildren(QLabel)[2]
            subtext.setText("")

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
            .subtext {
                font-size: 12px;
                color: #777777;
                font-style: italic;
                margin-top: 5px;
            }
        """)

        layout = QVBoxLayout(card)
        title_label = QLabel(title)

        value_label = QLabel(value)
        value_label.setProperty("class", "value")
        value_label.setAlignment(Qt.AlignCenter)

        subtext_label = QLabel(placeholder)
        subtext_label.setProperty("class", "subtext")
        subtext_label.setAlignment(Qt.AlignCenter)
        subtext_label.setWordWrap(True)

        layout.addWidget(title_label, alignment=Qt.AlignCenter)
        layout.addWidget(value_label, alignment=Qt.AlignCenter)
        layout.addWidget(subtext_label, alignment=Qt.AlignCenter)

        return card


class MoodRingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(150, 150)
        self._color = QColor(200, 200, 200)  # Start neutral (gray)
        self._pulse_animation = None
        self.setup_animations()

    def setup_animations(self):
        """Setup pulse animation effect"""
        self._pulse_animation = QPropertyAnimation(self, b"size")
        self._pulse_animation.setDuration(1000)
        self._pulse_animation.setLoopCount(-1)  # Infinite
        self._pulse_animation.setEasingCurve(QEasingCurve.OutInQuad)
    def update_mood(self, sentiment, confidence=0.5):
        """Update color based on sentiment"""
        colors = {
            'positive': QColor(100, 220, 100),  # Vibrant green
            'neutral': QColor(220, 220, 100),  # Yellow
            'negative': QColor(220, 100, 100)  # Red
        }
        self._color = colors.get(sentiment, QColor(200, 200, 200))

        # Simple animation (no confidence-based scaling)
        if self._pulse_animation:
            self._pulse_animation.stop()
            self._pulse_animation.setStartValue(QSize(140, 140))
            self._pulse_animation.setEndValue(QSize(160, 160))
            self._pulse_animation.start()

        self.update()

    def paintEvent(self, event):
        """Draw the mood ring with subtle glow"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Outer glow (subtle)
        if self._pulse_animation:
            glow_color = self._color.lighter(150)
            glow_color.setAlpha(80)
            painter.setBrush(QBrush(glow_color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(5, 5, self.width() - 10, self.height() - 10)

        # Main ring
        painter.setBrush(QBrush(self._color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(15, 15, self.width() - 30, self.height() - 30)
class BatchAnalysisWidget(QWidget):
    """Widget for batch analysis results"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Stats cards - FOR BATCH ANALYSIS ONLY
        cards_container = QWidget()
        cards_layout = QHBoxLayout(cards_container)

        # These are the cards needed for batch analysis
        self.avg_rating_card = self.create_stat_card("Average Rating", "0.0 ★")
        self.positive_card = self.create_stat_card("Positive", "0%")
        self.neutral_card = self.create_stat_card("Neutral", "0%")
        self.negative_card = self.create_stat_card("Negative", "0%")

        cards_layout.addWidget(self.avg_rating_card)
        cards_layout.addWidget(self.positive_card)
        cards_layout.addWidget(self.neutral_card)
        cards_layout.addWidget(self.negative_card)

        layout.addWidget(cards_container)

        # Visualization tabs
        self.viz_tabs = QTabWidget()
        self.setup_visualization_tabs()
        layout.addWidget(self.viz_tabs)

    def setup_visualization_tabs(self):
        """Setup visualization tabs for batch results"""
        # Pie chart tab
        self.pie_chart_tab = QWidget()
        pie_layout = QVBoxLayout(self.pie_chart_tab)
        self.pie_figure = plt.figure(facecolor='#353535')
        self.pie_canvas = FigureCanvas(self.pie_figure)
        self.pie_toolbar = NavigationToolbar(self.pie_canvas, self)
        pie_layout.addWidget(self.pie_toolbar)
        pie_layout.addWidget(self.pie_canvas)
        self.viz_tabs.addTab(self.pie_chart_tab, "Distribution")

        # Bar chart tab
        self.bar_chart_tab = QWidget()
        bar_layout = QVBoxLayout(self.bar_chart_tab)
        self.bar_figure = plt.figure(facecolor='#353535')
        self.bar_canvas = FigureCanvas(self.bar_figure)
        self.bar_toolbar = NavigationToolbar(self.bar_canvas, self)
        bar_layout.addWidget(self.bar_toolbar)
        bar_layout.addWidget(self.bar_canvas)
        self.viz_tabs.addTab(self.bar_chart_tab, "Ratings")

    def update_results(self, stats):
        """Update the widget with batch analysis results"""
        # Get the value labels from each card
        avg_rating_label = self.avg_rating_card.findChildren(QLabel)[1]
        positive_label = self.positive_card.findChildren(QLabel)[1]
        neutral_label = self.neutral_card.findChildren(QLabel)[1]
        negative_label = self.negative_card.findChildren(QLabel)[1]

        # Update the labels
        avg_rating_label.setText(f"{stats['avg_rating']:.1f} ★")
        positive_label.setText(f"{stats['positive_pct']:.1f}%")
        neutral_label.setText(f"{stats['neutral_pct']:.1f}%")
        negative_label.setText(f"{stats['negative_pct']:.1f}%")

        # Update charts
        self.update_pie_chart(stats)
        self.update_bar_chart(stats)

    def create_stat_card(self, title, value):
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
        value_label.setProperty("class", "value")
        value_label.setObjectName("value_label")  # Add this line

        layout.addWidget(title_label)
        layout.addWidget(value_label)

        return card

    def setup_pie_chart_tab(self):
        layout = QVBoxLayout(self.pie_chart_tab)

        # Matplotlib figure
        self.pie_figure = plt.figure(facecolor='#353535')
        self.pie_canvas = FigureCanvas(self.pie_figure)
        self.pie_toolbar = NavigationToolbar(self.pie_canvas, self)

        layout.addWidget(self.pie_toolbar)
        layout.addWidget(self.pie_canvas)

    def setup_bar_chart_tab(self):
        layout = QVBoxLayout(self.bar_chart_tab)

        # Matplotlib figure
        self.bar_figure = plt.figure(facecolor='#353535')
        self.bar_canvas = FigureCanvas(self.bar_figure)
        self.bar_toolbar = NavigationToolbar(self.bar_canvas, self)

        layout.addWidget(self.bar_toolbar)
        layout.addWidget(self.bar_canvas)


    def update_pie_chart(self, stats):
        self.pie_figure.clear()
        ax = self.pie_figure.add_subplot(111)

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [stats['positive_pct'], stats['neutral_pct'], stats['negative_pct']]
        colors = ['#6EE7B7', '#FBD38D', '#FCA5A5']

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'color': 'white'})
        ax.set_title('Sentiment Distribution', color='white')

        self.pie_canvas.draw()

    def update_bar_chart(self, stats):
        self.bar_figure.clear()
        ax = self.bar_figure.add_subplot(111)

        labels = ['1 ★', '2 ★', '3 ★', '4 ★', '5 ★']
        # Placeholder data - replace with actual distribution from your analyzer if available
        values = [5, 10, 20, 30, 35]  # Example distribution

        bars = ax.bar(labels, values, color='#7EB3FF')
        ax.set_title('Rating Distribution', color='white')
        ax.set_ylabel('Percentage', color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Set background colors
        ax.set_facecolor('#353535')
        self.bar_figure.patch.set_facecolor('#353535')

        self.bar_canvas.draw()


class SentimentAnalysisApp(QMainWindow):
    """Main application window for Sentiment Analysis"""

    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.history = []
        self.dark_mode = True
        self.setup_ui()
        self.load_models()

    def setup_ui(self):
        """Initialize the main UI components"""
        self.setWindowTitle("Sentiment Analysis")
        self.setGeometry(100, 100, 1200, 800)

        # Set window icon
        if os.path.exists("../assets/oujda.png"):
            self.setWindowIcon(QIcon("../assets/oujda.png"))

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Create splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Input controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # App header
        header = QLabel("Sentiment Analysis")
        header.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px;")
        left_layout.addWidget(header)

        # Input tabs
        self.input_tabs = QTabWidget()
        self.setup_input_tabs()
        left_layout.addWidget(self.input_tabs)

        # Model status
        self.model_status = QLabel("Model Status: Loading...")
        self.model_status.setStyleSheet("padding: 5px; font-style: italic;")
        left_layout.addWidget(self.model_status)

        left_layout.addStretch()

        # Right panel - Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Results stacked widget
        self.results_stack = QStackedWidget()

        # Single result view
        self.single_result_widget = AnalysisResultWidget()
        self.results_stack.addWidget(self.single_result_widget)

        # Batch result view
        self.batch_result_widget = BatchAnalysisWidget()
        self.results_stack.addWidget(self.batch_result_widget)

        right_layout.addWidget(self.results_stack)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 850])

        main_layout.addWidget(splitter)

        # Status bar
        self.setup_status_bar()

        # Menu bar
        self.setup_menu_bar()

        # Apply initial style
        self.update_style()

    def setup_input_tabs(self):
        """Setup the input tabs for single and batch analysis"""
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

        # File import group
        file_group = QGroupBox("Import Options")
        file_layout = QVBoxLayout()

        import_btn = QPushButton("Import from File")
        import_btn.clicked.connect(self.import_reviews)
        file_layout.addWidget(import_btn)

        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        file_layout.addWidget(export_btn)

        file_group.setLayout(file_layout)
        batch_layout.addWidget(file_group)

        self.input_tabs.addTab(single_tab, "Single Review")
        self.input_tabs.addTab(batch_tab, "Batch Analysis")

    def setup_status_bar(self):
        """Setup the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def setup_menu_bar(self):
        """Setup the menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        import_action = QAction("Import Reviews", self)
        import_action.triggered.connect(self.import_reviews)
        file_menu.addAction(import_action)

        export_action = QAction("Export Results", self)
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        history_action = QAction("View History", self)
        history_action.setShortcut(QKeySequence("Ctrl+H"))
        history_action.triggered.connect(self.show_history)
        view_menu.addAction(history_action)

        dashboard_action = QAction("Dashboard", self)
        dashboard_action.triggered.connect(self.show_dashboard)
        view_menu.addAction(dashboard_action)

        view_menu.addSeparator()

        theme_action = QAction("Toggle Dark/Light", self)
        theme_action.setShortcut(QKeySequence("Ctrl+T"))
        theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(theme_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def load_models(self):
        """Load the sentiment analysis models"""
        try:
            self.analyzer = SentimentAnalyzer(
                model_path="../models/sentiment_model_fr.pkl",
                vectorizer_path="../models/tfidf_vectorizer_fr.pkl"
            )
            self.model_status.setText("Model Status: Loaded")
            self.status_bar.showMessage("Models loaded successfully", 3000)
        except Exception as e:
            self.model_status.setText("Model Status: Failed to load")
            QMessageBox.critical(self, "Error", f"Failed to load models: {str(e)}")

    def analyze_single_review(self):
        """Analyze a single review"""
        if not self.analyzer:
            QMessageBox.warning(self, "Warning", "Models not loaded yet!")
            return

        text = self.review_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Warning", "Please enter some text to analyze!")
            return

        try:
            result = self.analyzer.analyze_sentiment(text)

            # Store in history
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rating": result['rating'],
                "sentiment": result['sentiment'],
                "text": result['text'],
                "confidence": result['confidence']
            }
            self.history.append(record)

            # Update UI
            self.single_result_widget.update_results(result)
            self.results_stack.setCurrentIndex(0)
            self.status_bar.showMessage("Analysis completed", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")

    def analyze_batch(self):
        """Analyze a batch of reviews"""
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

            # Store in history
            batch_record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "batch",
                "stats": stats,
                "count": len(reviews)
            }
            self.history.append(batch_record)

            # Update UI
            self.batch_result_widget.update_results(stats)
            self.results_stack.setCurrentIndex(1)
            self.status_bar.showMessage(f"Analyzed {len(reviews)} reviews", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Batch analysis failed: {str(e)}")

    def import_reviews(self):
        """Import reviews from a file"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Review File", "",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)",
            options=options)

        if file_name:
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.batch_input.setPlainText(content)
                    self.input_tabs.setCurrentIndex(1)  # Switch to batch tab
                    self.status_bar.showMessage(f"Loaded reviews from {file_name}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def export_results(self):
        """Export analysis results to file"""
        if not self.history:
            QMessageBox.warning(self, "Warning", "No analysis results to export!")
            return

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)",
            options=options)

        if file_name:
            try:
                # Prepare data for export
                export_data = []
                for record in self.history:
                    if record.get('type') == 'batch':
                        export_data.append({
                            "Timestamp": record['timestamp'],
                            "Type": "Batch",
                            "Count": record['count'],
                            "Avg Rating": record['stats']['avg_rating'],
                            "Positive %": record['stats']['positive_pct'],
                            "Neutral %": record['stats']['neutral_pct'],
                            "Negative %": record['stats']['negative_pct']
                        })
                    else:
                        export_data.append({
                            "Timestamp": record['timestamp'],
                            "Type": "Single",
                            "Rating": record['rating'],
                            "Sentiment": record['sentiment'],
                            "Confidence": record['confidence'],
                            "Text Preview": record['text']
                        })

                df = pd.DataFrame(export_data)

                if file_name.endswith('.xlsx'):
                    df.to_excel(file_name, index=False)
                else:
                    df.to_csv(file_name, index=False)

                self.status_bar.showMessage(f"Results exported to {file_name}", 3000)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def show_history(self):
        """Show analysis history"""
        if not self.history:
            QMessageBox.information(self, "History", "No analysis history yet")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Analysis History")
        dialog.resize(800, 600)

        layout = QVBoxLayout()

        # Create table
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(["Timestamp", "Type", "Rating", "Sentiment", "Confidence", "Preview"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)

        # Populate table
        table.setRowCount(len(self.history))
        for row, record in enumerate(self.history):
            table.setItem(row, 0, QTableWidgetItem(record['timestamp']))

            if record.get('type') == 'batch':
                table.setItem(row, 1, QTableWidgetItem("Batch"))
                table.setItem(row, 2, QTableWidgetItem(f"{record['stats']['avg_rating']:.1f}"))
                table.setItem(row, 3, QTableWidgetItem("N/A"))
                table.setItem(row, 4, QTableWidgetItem("N/A"))
                table.setItem(row, 5, QTableWidgetItem(f"{record['count']} reviews"))
            else:
                table.setItem(row, 1, QTableWidgetItem("Single"))
                table.setItem(row, 2, QTableWidgetItem(f"{record['rating']:.1f}"))
                table.setItem(row, 3, QTableWidgetItem(record['sentiment'].capitalize()))
                table.setItem(row, 4, QTableWidgetItem(f"{record['confidence']:.1%}"))
                table.setItem(row, 5, QTableWidgetItem(record['text']))

        # Resize columns to contents
        table.resizeColumnsToContents()

        layout.addWidget(table)
        dialog.setLayout(layout)
        dialog.exec_()

    def show_dashboard(self):
        """Show analytics dashboard"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Analytics Dashboard")
        dialog.resize(1000, 700)

        layout = QVBoxLayout()

        # Create tabs
        tab_widget = QTabWidget()

        # Summary tab
        summary_tab = QWidget()
        self.setup_summary_tab(summary_tab)
        tab_widget.addTab(summary_tab, "Summary")

        # History tab
        history_tab = QWidget()
        self.setup_history_tab(history_tab)
        tab_widget.addTab(history_tab, "History")

        layout.addWidget(tab_widget)
        dialog.setLayout(layout)
        dialog.exec_()

    def setup_summary_tab(self, tab):
        """Setup the summary tab for the dashboard"""
        layout = QVBoxLayout(tab)

        # Stats cards
        stats_container = QWidget()
        stats_layout = QHBoxLayout(stats_container)

        # Calculate stats from history
        total_analyses = len(self.history)
        single_analyses = len([r for r in self.history if r.get('type') != 'batch'])
        batch_analyses = len([r for r in self.history if r.get('type') == 'batch'])

        # Create cards
        total_card = self.create_dashboard_card("Total Analyses", str(total_analyses))
        single_card = self.create_dashboard_card("Single Analyses", str(single_analyses))
        batch_card = self.create_dashboard_card("Batch Analyses", str(batch_analyses))

        stats_layout.addWidget(total_card)
        stats_layout.addWidget(single_card)
        stats_layout.addWidget(batch_card)

        layout.addWidget(stats_container)

        # Add visualization
        self.dashboard_figure = plt.figure(facecolor='#353535' if self.dark_mode else 'white')
        self.dashboard_canvas = FigureCanvas(self.dashboard_figure)
        self.dashboard_toolbar = NavigationToolbar(self.dashboard_canvas, self)

        layout.addWidget(self.dashboard_toolbar)
        layout.addWidget(self.dashboard_canvas)

        # Update visualization
        self.update_dashboard_chart()

    def create_dashboard_card(self, title, value):
        """Create a card for the dashboard"""
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
                font-size: 24px;
                font-weight: bold;
            }
        """)

        layout = QVBoxLayout(card)
        title_label = QLabel(title)
        value_label = QLabel(value)
        value_label.setProperty("class", "value")

        layout.addWidget(title_label)
        layout.addWidget(value_label)

        return card

    def update_dashboard_chart(self):
        """Update the dashboard chart with historical data"""
        self.dashboard_figure.clear()
        ax = self.dashboard_figure.add_subplot(111)

        # Prepare data for chart
        dates = []
        ratings = []

        for record in self.history:
            if record.get('type') != 'batch':  # Only single analyses
                dates.append(datetime.strptime(record['timestamp'], "%Y-%m-%d %H:%M:%S"))
                ratings.append(record['rating'])

        if not dates:
            ax.text(0.5, 0.5, "No analysis data available",
                    ha='center', va='center', color='white')
            ax.set_facecolor('#353535')
            self.dashboard_canvas.draw()
            return

        # Plot ratings over time
        ax.plot(dates, ratings, 'o-', color='#7EB3FF')
        ax.set_title('Rating Trend Over Time', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Rating', color='white')

        # Style the plot
        ax.set_facecolor('#353535')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Rotate date labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        self.dashboard_canvas.draw()

    def setup_history_tab(self, tab):
        """Setup the history tab for the dashboard"""
        layout = QVBoxLayout(tab)

        # Create table
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(["Timestamp", "Type", "Rating", "Sentiment", "Confidence", "Preview"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)

        # Filter to show only single analyses
        single_analyses = [r for r in self.history if r.get('type') != 'batch']
        table.setRowCount(len(single_analyses))

        for row, record in enumerate(single_analyses):
            table.setItem(row, 0, QTableWidgetItem(record['timestamp']))
            table.setItem(row, 1, QTableWidgetItem("Single"))
            table.setItem(row, 2, QTableWidgetItem(f"{record['rating']:.1f}"))
            table.setItem(row, 3, QTableWidgetItem(record['sentiment'].capitalize()))
            table.setItem(row, 4, QTableWidgetItem(f"{record['confidence']:.1%}"))
            table.setItem(row, 5, QTableWidgetItem(record['text']))

        # Resize columns to contents
        table.resizeColumnsToContents()

        layout.addWidget(table)

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About Sentiment Analysis ",
                          "Sentiment Analysis \n\n"
                          "A powerful tool for analyzing customer reviews and feedback.\n"
                          "made by Badreddine Chihab and Amine chakri as an academic project\n\n"
                          "©2025")

    def toggle_theme(self):
        """Toggle between dark and light theme"""
        self.dark_mode = not self.dark_mode
        self.update_style()

    def update_style(self):
        """Update the application style"""
        ModernStyle.setup(QApplication.instance(), self.dark_mode)
        self.setStyleSheet(ModernStyle.get_stylesheet(self.dark_mode))

        # Update matplotlib figure backgrounds
        if hasattr(self, 'pie_figure'):
            self.pie_figure.set_facecolor('#353535' if self.dark_mode else 'white')
            if hasattr(self, 'pie_canvas'):
                self.pie_canvas.draw()

        if hasattr(self, 'bar_figure'):
            self.bar_figure.set_facecolor('#353535' if self.dark_mode else 'white')
            if hasattr(self, 'bar_canvas'):
                self.bar_canvas.draw()

        if hasattr(self, 'dashboard_figure'):
            self.dashboard_figure.set_facecolor('#353535' if self.dark_mode else 'white')
            if hasattr(self, 'dashboard_canvas'):
                self.dashboard_canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Apply modern style
    ModernStyle.setup(app, dark_mode=True)

    # Create and show main window
    window = SentimentAnalysisApp()
    window.show()

    # Run the application
    sys.exit(app.exec_())