from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "drug200.csv"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA}. Put drug200.csv in the data folder."
    )

df = pd.read_csv(DATA)
required = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K", "Drug"]
missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing columns: {missing}")

df = df[required].dropna()
X = df[["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]]
y = df["Drug"]

categorical = ["Sex", "BP", "Cholesterol"]
numeric = ["Age", "Na_to_K"]

preprocessor = ColumnTransformer(
    [
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", StandardScaler(), numeric),
    ]
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

joblib.dump(pipeline, MODEL_DIR / "drug_model.joblib")

initial_types = [
    ("Age", FloatTensorType([None, 1])),
    ("Sex", StringTensorType([None, 1])),
    ("BP", StringTensorType([None, 1])),
    ("Cholesterol", StringTensorType([None, 1])),
    ("Na_to_K", FloatTensorType([None, 1])),
]
onnx_model = convert_sklearn(
    pipeline,
    initial_types=initial_types,
    target_opset=17,
)
with open(MODEL_DIR / "drug_model.onnx", "wb") as file:
    file.write(onnx_model.SerializeToString())

matrix = confusion_matrix(y_test, predictions, labels=pipeline.classes_)
display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=pipeline.classes_,
)
display.plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
plt.close()

report = (
    "Drug Classification Model Evaluation\n"
    f"Dataset rows used: {len(df)}\n"
    f"Training rows: {len(X_train)}\n"
    f"Test rows: {len(X_test)}\n"
    f"Accuracy: {accuracy:.4f}\n\n"
    + classification_report(y_test, predictions, zero_division=0)
)
(RESULTS_DIR / "metrics.txt").write_text(report, encoding="utf-8")

print(report)
print("Saved model/drug_model.joblib")
print("Saved model/drug_model.onnx")
print("Saved results/metrics.txt")
print("Saved results/confusion_matrix.png")
