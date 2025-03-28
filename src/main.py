import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QTextEdit, QPushButton, QComboBox, QTabWidget,
                             QFormLayout, QProgressBar, QListWidget, QMessageBox,
                             QFileDialog, QStatusBar, QFrame, QSplitter)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor
from predict_star import SentimentAnalyzer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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
        self.initUI()
        self.load_models()

    def initUI(self):
        """Initialize the main UI components"""
        self.setWindowTitle("Advanced Sentiment Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon('../assets/oujda.png'))

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create a splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Input and controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # App header
        header = QLabel("Sentiment Analyser ")
        header.setFont(QFont('Arial', 18, QFont.Bold))
        header.setStyleSheet(f"color: {COLORS['accent']}; margin-bottom: 20px;")
        left_layout.addWidget(header)

        # Model status
        self.model_status = QLabel("Model Status: Loading...")
        self.model_status.setStyleSheet(f"color: {COLORS['neutral']}; font-style: italic;")
        left_layout.addWidget(self.model_status)

        # Input tabs
        input_tabs = QTabWidget()

        # Single review tab
        single_tab = QWidget()
        single_layout = QVBoxLayout(single_tab)

        self.review_input = QTextEdit()
        self.review_input.setPlaceholderText("Enter your review text here...")
        self.review_input.setStyleSheet(f"""
            QTextEdit {{
                background: {COLORS['card']};
                color: {COLORS['foreground']};
                border: 1px solid {COLORS['secondary']};
                border-radius: 5px;
                padding: 10px;
                min-height: 150px;
            }}
        """)
        single_layout.addWidget(self.review_input)

        analyze_btn = QPushButton("Analyze Sentiment")
        analyze_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {COLORS['secondary']};
            }}
        """)
        analyze_btn.clicked.connect(self.analyze_single_review)
        single_layout.addWidget(analyze_btn)

        input_tabs.addTab(single_tab, "Single Review")

        # Batch processing tab
        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)

        self.batch_input = QTextEdit()
        self.batch_input.setPlaceholderText("Enter multiple reviews, one per line...")
        self.batch_input.setStyleSheet(f"""
            QTextEdit {{
                background: {COLORS['card']};
                color: {COLORS['foreground']};
                border: 1px solid {COLORS['secondary']};
                border-radius: 5px;
                padding: 10px;
                min-height: 150px;
            }}
        """)
        batch_layout.addWidget(self.batch_input)

        batch_btn = QPushButton("Analyze Batch")
        batch_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {COLORS['secondary']};
            }}
        """)
        batch_btn.clicked.connect(self.analyze_batch)
        batch_layout.addWidget(batch_btn)

        input_tabs.addTab(batch_tab, "Batch Processing")

        left_layout.addWidget(input_tabs)

        # File import section
        file_frame = QFrame()
        file_frame.setFrameShape(QFrame.StyledPanel)
        file_frame.setStyleSheet(f"background: {COLORS['card']}; border-radius: 5px; padding: 10px;")
        file_layout = QVBoxLayout(file_frame)

        file_label = QLabel("Import Reviews from File:")
        file_label.setStyleSheet(f"color: {COLORS['accent']};")
        file_layout.addWidget(file_label)

        import_btn = QPushButton("Choose File")
        import_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['secondary']};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background: {COLORS['primary']};
            }}
        """)
        import_btn.clicked.connect(self.import_reviews)
        file_layout.addWidget(import_btn)

        left_layout.addWidget(file_frame)

        # Add some spacing
        left_layout.addStretch()

        # Right panel - Results and visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Results tabs
        self.results_tabs = QTabWidget()

        # Single result tab
        self.single_result_tab = QWidget()
        single_result_layout = QVBoxLayout(self.single_result_tab)

        self.result_card = QFrame()
        self.result_card.setFrameShape(QFrame.StyledPanel)
        self.result_card.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['card']};
                border-radius: 10px;
                padding: 20px;
            }}
        """)

        card_layout = QVBoxLayout(self.result_card)

        self.result_header = QLabel("Analysis Results")
        self.result_header.setStyleSheet(f"color: {COLORS['accent']}; font-size: 16px; font-weight: bold;")
        card_layout.addWidget(self.result_header)

        # Form layout for results
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignLeft)

        self.rating_label = QLabel()
        self.sentiment_label = QLabel()
        self.confidence_label = QLabel()

        form_layout.addRow("Predicted Rating:", self.rating_label)
        form_layout.addRow("Sentiment:", self.sentiment_label)
        form_layout.addRow("Confidence:", self.confidence_label)

        card_layout.addLayout(form_layout)

        # Sentiment visualization
        self.sentiment_visual = QLabel()
        self.sentiment_visual.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.sentiment_visual)

        single_result_layout.addWidget(self.result_card)

        self.results_tabs.addTab(self.single_result_tab, "Single Analysis")

        # Batch results tab
        self.batch_result_tab = QWidget()
        batch_result_layout = QVBoxLayout(self.batch_result_tab)

        # Stats card
        self.stats_card = QFrame()
        self.stats_card.setFrameShape(QFrame.StyledPanel)
        self.stats_card.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['card']};
                border-radius: 10px;
                padding: 20px;
            }}
        """)

        stats_layout = QVBoxLayout(self.stats_card)

        self.stats_header = QLabel("Batch Statistics")
        self.stats_header.setStyleSheet(f"color: {COLORS['accent']}; font-size: 16px; font-weight: bold;")
        stats_layout.addWidget(self.stats_header)

        # Stats form
        self.stats_form = QFormLayout()
        self.stats_form.setLabelAlignment(Qt.AlignLeft)

        self.avg_rating_label = QLabel()
        self.positive_pct_label = QLabel()
        self.neutral_pct_label = QLabel()
        self.negative_pct_label = QLabel()
        self.total_reviews_label = QLabel()

        self.stats_form.addRow("Average Rating:", self.avg_rating_label)
        self.stats_form.addRow("Positive Reviews:", self.positive_pct_label)
        self.stats_form.addRow("Neutral Reviews:", self.neutral_pct_label)
        self.stats_form.addRow("Negative Reviews:", self.negative_pct_label)
        self.stats_form.addRow("Total Reviews:", self.total_reviews_label)

        stats_layout.addLayout(self.stats_form)

        batch_result_layout.addWidget(self.stats_card)

        # Visualization
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        batch_result_layout.addWidget(self.canvas)

        self.results_tabs.addTab(self.batch_result_tab, "Batch Results")

        # Review list tab
        self.review_list_tab = QWidget()
        review_list_layout = QVBoxLayout(self.review_list_tab)

        self.review_list = QListWidget()
        self.review_list.setStyleSheet(f"""
            QListWidget {{
                background: {COLORS['card']};
                color: {COLORS['foreground']};
                border: 1px solid {COLORS['secondary']};
                border-radius: 5px;
            }}
        """)
        review_list_layout.addWidget(self.review_list)

        self.results_tabs.addTab(self.review_list_tab, "Review Details")

        right_layout.addWidget(self.results_tabs)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])

        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready", 3000)

        # Apply dark theme
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {COLORS['background']};
                color: {COLORS['foreground']};
            }}
            QTabWidget::pane {{
                border: 1px solid {COLORS['secondary']};
                border-radius: 5px;
                padding: 5px;
                background: {COLORS['background']};
            }}
            QTabBar::tab {{
                background: {COLORS['card']};
                color: {COLORS['foreground']};
                padding: 8px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background: {COLORS['primary']};
                color: white;
            }}
        """)

    def load_models(self):
        """Load the sentiment analysis models"""
        try:
            self.analyzer = SentimentAnalyzer(
                model_path="../models/sentiment_model_fr.pkl",
                vectorizer_path="../models/tfidf_vectorizer_fr.pkl"
            )
            self.model_status.setText("Model Status: Loaded Successfully")
            self.model_status.setStyleSheet(f"color: {COLORS['positive']};")
        except Exception as e:
            self.model_status.setText("Model Status: Failed to Load")
            self.model_status.setStyleSheet(f"color: {COLORS['negative']};")
            QMessageBox.critical(self, "Error", f"Failed to load models: {str(e)}")

    def analyze_single_review(self):
        """Analyze a single review text"""
        if not self.analyzer:
            QMessageBox.warning(self, "Warning", "Models not loaded yet!")
            return

        text = self.review_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Warning", "Please enter some text to analyze!")
            return

        try:
            result = self.analyzer.analyze_sentiment(text)

            # Update UI with results
            self.rating_label.setText(f"{result['rating']:.1f} ★")
            self.sentiment_label.setText(result['sentiment'].capitalize())
            self.confidence_label.setText(f"{result['confidence'] * 100:.1f}%")

            # Style based on sentiment
            sentiment_color = {
                'positive': COLORS['positive'],
                'neutral': COLORS['neutral'],
                'negative': COLORS['negative']
            }[result['sentiment']]

            self.sentiment_label.setStyleSheet(f"color: {sentiment_color}; font-weight: bold;")

            # Create a simple visualization
            pixmap = QPixmap(300, 50)
            pixmap.fill(QColor(COLORS['card']))

            # Show rating as stars
            stars = ""
            full_stars = int(result['rating'])
            half_star = result['rating'] - full_stars >= 0.5

            stars = "★" * full_stars
            if half_star:
                stars += "½"
            stars += "☆" * (5 - full_stars - (1 if half_star else 0))

            self.rating_label.setText(f"{stars} ({result['rating']:.1f})")

            self.results_tabs.setCurrentIndex(0)
            self.status_bar.showMessage("Analysis completed", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")

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

    # Set modern style
    app.setStyle('Fusion')

    # Create and show the main window
    window = SentimentAnalysisApp()
    window.show()

    sys.exit(app.exec_())