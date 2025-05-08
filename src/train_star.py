import pandas as pd
import joblib
import json
import numpy as np
from sklearn.utils import resample, compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from lightgbm import LGBMClassifier

# Load dataset
df = pd.read_csv("../data/cleaned/updated_dataset.csv")
df["Stars"] = df["Stars"].astype(int)

# Upsample each class to 200 samples (if needed)
upsampled_classes = []
for star in df["Stars"].unique():
    class_df = df[df["Stars"] == star]
    if len(class_df) < 200:
        class_df = resample(class_df, replace=True, n_samples=200, random_state=42)
    upsampled_classes.append(class_df)

df_balanced = pd.concat(upsampled_classes).sample(frac=1, random_state=42)  # shuffle

# Extract features and target
X = df_balanced["Review Text"]
y = df_balanced["Stars"]

# Load pre-fitted TF-IDF vectorizer
vectorizer = joblib.load("../models/vectorizer.pkl")
X_tfidf = vectorizer.transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.3, random_state=42, stratify=y
)

# Compute class weights
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

# Initialize and train LightGBM classifier
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.1,
    random_state=101,
    class_weight=class_weight_dict
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f"✅ Accuracy: {accuracy:.4f}")
print("📋 Classification Report:")
print(json.dumps(report, indent=2))

# Save model
joblib.dump(model, "../models/model_star.pkl")

# Save evaluation results
results = {
    "Accuracy": accuracy,
    "Classification Report": report,
    "Confusion Matrix": conf_matrix.tolist()
}
with open("../reports/star_rating_evaluation_upsampled.json", "w") as f:
    json.dump(results, f)

print("✅ Balanced & upsampled classifier model saved and evaluation results stored.")
