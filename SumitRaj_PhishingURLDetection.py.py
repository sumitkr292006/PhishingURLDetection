
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


# ==========================================
# 1. OUTPUT FOLDER
# ==========================================
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================
print("Loading UCI Phishing Websites dataset...")

dataset = fetch_ucirepo(id=327)

X = dataset.data.features.copy()
y = dataset.data.targets.iloc[:, 0].copy()

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([float("inf"), float("-inf")], float("nan"))

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y.astype(str).str.strip())

class_names = [str(name) for name in label_encoder.classes_]

print("Dataset loaded successfully!")
print("Number of records:", X.shape[0])
print("Number of features:", X.shape[1])
print("Target classes:", class_names)


# ==========================================
# 3. SPLIT DATA
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ==========================================
# 4. TRAIN MACHINE LEARNING MODEL
# ==========================================
print("\nTraining Random Forest model...")

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    ),
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# ==========================================
# 5. EVALUATE MODEL
# ==========================================
accuracy = accuracy_score(y_test, predictions)
report = classification_report(
    y_test,
    predictions,
    target_names=class_names,
    zero_division=0,
)

print("\n========== PROJECT RESULTS ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)


# Save evaluation results
with open(
    OUTPUT_DIR / "evaluation_results.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write(report)


# Save predictions
results = pd.DataFrame({
    "Actual_Class": label_encoder.inverse_transform(y_test),
    "Predicted_Class": label_encoder.inverse_transform(predictions),
})

results.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
)


# Save target label mapping
with open(
    OUTPUT_DIR / "target_label_mapping.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write("Encoded class mapping:\n")
    for number, label in enumerate(class_names):
        file.write(f"{number} = {label}\n")


# ==========================================
# 6. CREATE BOTH GRAPHS
# ==========================================
print("\nCreating graphs...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))


# Graph 1: Confusion Matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=list(range(len(class_names))),
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

display.plot(
    ax=axes[0],
    values_format="d",
    colorbar=False,
)

axes[0].set_title(
    "Phishing Detection - Confusion Matrix",
    fontsize=13,
)
axes[0].set_xlabel("Predicted Class")
axes[0].set_ylabel("Actual Class")


# Graph 2: Feature Importance
classifier = model.named_steps["classifier"]

importance = pd.Series(
    classifier.feature_importances_,
    index=X.columns,
).sort_values(ascending=False).head(15).sort_values()

importance.plot(
    kind="barh",
    ax=axes[1],
)

axes[1].set_title(
    "Top 15 Important Features",
    fontsize=13,
)
axes[1].set_xlabel("Feature Importance")
axes[1].set_ylabel("Dataset Feature")


# Save graphs and show them on screen
plt.tight_layout()

graph_path = OUTPUT_DIR / "combined_graphs.png"

plt.savefig(
    graph_path,
    dpi=200,
    bbox_inches="tight",
)

print("Graphs saved at:", graph_path.resolve())
print("Opening graphs on screen...")

plt.show()


# Save feature importance separately as CSV
importance_df = pd.DataFrame({
    "Feature": classifier.feature_names_in_,
    "Importance": classifier.feature_importances_,
}).sort_values(
    by="Importance",
    ascending=False,
)

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False,
)

print("\nProject completed successfully!")
print("All output files are in:", OUTPUT_DIR.resolve())
