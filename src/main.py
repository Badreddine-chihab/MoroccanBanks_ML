import sys

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QTextEdit, QPushButton, QTabWidget,
                             QMessageBox, QFileDialog, QStatusBar, QFrame, QSplitter,
                             QTableWidget, QTableWidgetItem,
                             QAction, QDateEdit)
from PyQt5.QtCore import Qt,QDate
from PyQt5.QtGui import QColor, QPalette,QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from datetime import datetime, timedelta
import pandas as pd
from predict_star import SentimentAnalyzer



class ModernStyle:
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
        self.rating_card = self.create_card("⭐ Rating", "NAN", "Enter text to analyze")

        # Sentiment card
        self.sentiment_card = self.create_card("😊 Sentiment", "NAN", "Waiting for input...")

        layout.addWidget(self.rating_card)
        layout.addWidget(self.sentiment_card)
        layout.addStretch()

    def update_results(self, result):

        # Update rating card
        rating_label = self.rating_card.findChild(QLabel, "value_label")
        rating_label.setText(f"{result['rating']:.1f}")


        sentiment_label = self.sentiment_card.findChild(QLabel, "value_label")
        sentiment_label.setText(result['sentiment'].capitalize())

        #based on sentiment
        color = "#6EE7B7" if result['sentiment'] == 'positive' else "#FCA5A5"
        sentiment_label.setStyleSheet(f"color: {color};")

    def create_card(self, title, value, placeholder=""):
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


class HistoryWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history_data = []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Filter controls
        filter_layout = QHBoxLayout()

        self.date_from = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setDate(QDate.currentDate().addDays(-7))
        self.date_to = QDateEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.setDate(QDate.currentDate())

        filter_btn = QPushButton("Filter")
        filter_btn.clicked.connect(self.filter_history)

        filter_layout.addWidget(QLabel("From:"))
        filter_layout.addWidget(self.date_from)
        filter_layout.addWidget(QLabel("To:"))
        filter_layout.addWidget(self.date_to)
        filter_layout.addWidget(filter_btn)
        filter_layout.addStretch()

        layout.addLayout(filter_layout)

        # History table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["Date", "Review", "Rating", "Sentiment", "Confidence"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)

        layout.addWidget(self.history_table)

        # History visualization
        self.history_figure = plt.figure(facecolor='#353535')
        self.history_canvas = FigureCanvas(self.history_figure)
        self.history_toolbar = NavigationToolbar(self.history_canvas, self)

        layout.addWidget(self.history_toolbar)
        layout.addWidget(self.history_canvas)

    def update_history(self, history_data):

        self.history_data = history_data
        self.filter_history()

    def filter_history(self):

        from_date = self.date_from.date().toPyDate()
        to_date = self.date_to.date().toPyDate() + timedelta(days=1)  # Include the end date

        filtered_data = [
            item for item in self.history_data
            if from_date <= datetime.strptime(item['timestamp'], "%Y-%m-%d %H:%M:%S").date() < to_date
        ]


        self.history_table.setRowCount(len(filtered_data))
        for row, item in enumerate(filtered_data):
            self.history_table.setItem(row, 0, QTableWidgetItem(item['timestamp']))
            self.history_table.setItem(row, 1, QTableWidgetItem(item['text']))
            self.history_table.setItem(row, 2, QTableWidgetItem(f"{item['rating']:.1f}"))
            self.history_table.setItem(row, 3, QTableWidgetItem(item['sentiment'].capitalize()))
            self.history_table.setItem(row, 4, QTableWidgetItem(f"{item['confidence']:.2f}"))


        self.update_history_plot(filtered_data)

    def update_history_plot(self, filtered_data):

        self.history_figure.clear()

        if not filtered_data:
            ax = self.history_figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No data in selected date range',
                    ha='center', va='center', color='white')
            ax.set_facecolor('#353535')
            self.history_canvas.draw()
            return


        df = pd.DataFrame(filtered_data)
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_stats = df.groupby('date').agg({
            'rating': 'mean',
            'sentiment': lambda x: (x == 'positive').mean()
        }).reset_index()


        ax = self.history_figure.add_subplot(111)


        ax.plot(daily_stats['date'], daily_stats['rating'],
                marker='o', color='#6EE7B7', label='Average Rating')
        ax.set_ylabel('Average Rating', color='#6EE7B7')
        ax.tick_params(axis='y', labelcolor='#6EE7B7')
        ax.set_ylim(0, 5)


        ax2 = ax.twinx()
        ax2.plot(daily_stats['date'], daily_stats['sentiment'] * 100,
                 marker='s', color='#93C5FD', label='Positive %')
        ax2.set_ylabel('Positive Sentiment %', color='#93C5FD')
        ax2.tick_params(axis='y', labelcolor='#93C5FD')
        ax2.set_ylim(0, 100)


        ax.set_title('Review History Trend', color='white')
        ax.set_facecolor('#353535')
        ax.grid(True, color='#555555', linestyle='--')
        ax.xaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')


        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')

        self.history_canvas.draw()


class BatchAnalysisWidget(QWidget):


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)


        cards_layout = QHBoxLayout()
        self.avg_rating_card = self.create_stat_card("Average Rating", "0.0")
        self.positive_card = self.create_stat_card("Positive", "0%")
        self.negative_card = self.create_stat_card("Negative", "0%")

        cards_layout.addWidget(self.avg_rating_card)
        cards_layout.addWidget(self.positive_card)
        cards_layout.addWidget(self.negative_card)

        layout.addLayout(cards_layout)


        self.viz_tabs = QTabWidget()
        self.setup_visualization_tabs()
        layout.addWidget(self.viz_tabs)

    def setup_visualization_tabs(self):

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


        self.avg_rating_card.findChild(QLabel, "value_label").setText(f"{stats['avg_rating']:.1f}")
        self.positive_card.findChild(QLabel, "value_label").setText(f"{stats['positive_pct']:.1f}%")
        self.negative_card.findChild(QLabel, "value_label").setText(f"{stats['negative_pct']:.1f}%")


        self.update_pie_chart(stats)

    def update_pie_chart(self, stats):

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


    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.history = []
        self.dark_mode = True
        self.setup_ui()
        self.load_models()

    def setup_ui(self):

        self.setWindowTitle("Bank Review Sentiment Analysis")
        self.setWindowIcon(QIcon('../assets/oujda.png'))

        self.setGeometry(100, 100, 1200, 800)

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

        # Results tabs
        self.results_tabs = QTabWidget()
        self.single_result_widget = AnalysisResultWidget()
        self.batch_result_widget = BatchAnalysisWidget()
        self.history_widget = HistoryWidget()

        self.results_tabs.addTab(self.single_result_widget, "Single Analysis")
        self.results_tabs.addTab(self.batch_result_widget, "Batch Analysis")
        self.results_tabs.addTab(self.history_widget, "Review History")

        right_layout.addWidget(self.results_tabs)

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

        # Single review tab
        single_tab = QWidget()
        single_layout = QVBoxLayout(single_tab)

        self.review_input = QTextEdit()
        self.review_input.setPlaceholderText("Enter bank review text...")
        single_layout.addWidget(self.review_input)

        analyze_btn = QPushButton("Analyze")
        analyze_btn.clicked.connect(self.analyze_single_review)
        single_layout.addWidget(analyze_btn)

        # Batch review tab
        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)

        self.batch_input = QTextEdit()
        self.batch_input.setPlaceholderText("Enter one bank review per line...")
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

        clear_history_action = QAction("Clear History", self)
        clear_history_action.triggered.connect(self.clear_history)
        file_menu.addAction(clear_history_action)

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
            history_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "text": text[:100] + "..." if len(text) > 100 else text,
                "rating": result["rating"],
                "sentiment": result["sentiment"],
                "confidence": result["confidence"]
            }
            self.history.append(history_entry)

            # Update UI
            self.single_result_widget.update_results(result)
            self.results_tabs.setCurrentIndex(0)
            self.history_widget.update_history(self.history)
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
            self.results_tabs.setCurrentIndex(1)
            self.history_widget.update_history(self.history)
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

    def clear_history(self):
        """Clear the analysis history"""
        reply = QMessageBox.question(
            self, 'Clear History',
            'Are you sure you want to clear all review history?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.history = []
            self.history_widget.update_history(self.history)
            self.status_bar.showMessage("History cleared", 3000)

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About",
                          "Bank Review Sentiment Analysis\n\n"
                          "Badr Chihab && Amine Chakri\n"
                          "©Bank Customer Feedback Tool\n"
                          "ENSAO")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ModernStyle.setup(app, dark_mode=True)
    window = SentimentAnalysisApp()
    window.show()
    sys.exit(app.exec_())