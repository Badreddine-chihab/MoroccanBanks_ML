                         PROJECT PLAAN

       📊 Data Cleaning Plan :

🔹 Step 1: Remove Unnecessary Columns ✅ (Done)
Removed Unnamed: 0, Google Map ID, and Timestamp (not useful).

Removed Address, Website, and Phone Number (not necessary for analysis).

🔹 Step 2: Handle Missing Values
✅ What We Will Do:
1️⃣ Review Text

Remove rows where Review Text is "No review text found" (these are not real reviews).

2️⃣ Stars

Ensure all values are between 1 and 5.

If any invalid values exist, remove or correct them.

🔹 Step 3: Remove Unreal or Incorrect Cities
✅ What We Will Do:
1️⃣ Convert all city names to lowercase for consistency.
2️⃣ Compare city names against a list of real Moroccan cities.
3️⃣ Remove rows with non-existent or misspelled cities.

📌 Question: Do you have a reference list of valid Moroccan cities, or should I use a general one?

🔹 Step 4: Remove Duplicates
✅ What We Will Do:
1️⃣ Check for duplicate rows based on:

Business Name

Review Text

TrueTimestamp

2️⃣ Remove exact duplicates while keeping unique reviews.

🔹 Step 5: Convert & Format Data
✅ What We Will Do:
1️⃣ City & Business Name

Standardize formatting (e.g., trim spaces, fix capitalization).

2️⃣ TrueTimestamp

Ensure it is in proper datetime format for analysis.

Final Check & Save Cleaned Data
✅ Verify the dataset for consistency.
✅ Save the cleaned version for further analysis.

            📊 Detailed Plan for Visualization & Analysis
Now that our dataset is clean, we will explore insights through visualizations and analysis before moving to sentiment analysis and prediction.

🔹 Step 1: Exploratory Data Analysis (EDA)
✅ Overview of Data

Display summary statistics (mean, median, distribution of stars, etc.).

Count of reviews per city and per bank.

Check for trends in TrueTimestamp (seasonality, peaks in reviews).

🔹 Step 2: Data Visualizations
✅ Distribution of Star Ratings

Histogram or Pie Chart: Show the proportion of 1-star to 5-star reviews.

✅ Top Cities by Review Count

Bar Chart: Rank cities by the number of reviews.

✅ Most Reviewed Banks

Bar Chart: Show which banks have the most reviews.

✅ Review Trends Over Time

Time Series Line Chart: Plot number of reviews over time to see growth or seasonal patterns.

✅ Sentiment-Based Word Cloud (Optional)

Visualize frequently used words in positive and negative reviews.

🔹 Step 3: Deeper Analysis
✅ Star Ratings by City

Box Plot: Compare rating distributions per city.

✅ Star Ratings by Bank

Bar Chart: Average ratings for different banks.

✅ Relationship Between Review Length & Star Ratings

Scatter Plot or Box Plot: Do longer reviews tend to have lower/higher ratings?


       
         🤖 Machine Learning Plan (Part 3: Sentiment Analysis & Star Prediction)
Now that we have cleaned the data and performed visualization, we will move on to Machine Learning (ML) tasks:

🔹 Step 1: Sentiment Analysis
✅ Objective: Determine whether a review is positive, neutral, or negative based on text.

1️⃣ Preprocessing Review Text
Convert text to lowercase.

Remove stopwords, punctuation, and special characters.

Tokenization and lemmatization.

2️⃣ Labeling Sentiments
Define labels:

Positive → Stars >= 4

Negative → Stars <= 2

Neutral → Stars == 3

3️⃣ Train a Sentiment Classification Model
Model choices:

Naïve Bayes (Baseline, fast)

Logistic Regression

Deep Learning (LSTM/BERT, optional)

Evaluation Metrics:

Accuracy, Precision, Recall, F1-score.

🔹 Step 2: Star Rating Prediction
✅ Objective: Predict the Stars (1 to 5) based on review text.

1️⃣ Feature Engineering
Convert review text into numerical format using:

TF-IDF (Term Frequency-Inverse Document Frequency)

Word Embeddings (Word2Vec, FastText, or BERT embeddings)

2️⃣ Train a Regression or Classification Model
Model Choices:

Random Forest

XGBoost

Deep Learning (LSTM, Transformer)

Evaluation Metrics:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

Classification Accuracy (if treating as classification task).

🔹 Step 3: Model Deployment (Optional)
Save the trained model.

Build an API or simple web app for real-time sentiment prediction.


