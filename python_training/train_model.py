import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

os.makedirs("data/processed", exist_ok=True)
os.makedirs("docs", exist_ok=True)

print("Loading data...")
df = pd.read_csv("data/raw/gestures_all.csv")
print(f"Total samples: {len(df)}")
print(f"Gesture distribution:\n{df['label'].value_counts()}")

X = df.drop("label", axis=1).values
y = df["label"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nClasses: {le.classes_}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Training complete!")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

with open("data/processed/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("data/processed/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Model saved to data/processed/model.pkl")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix — Gesture Classifier")
plt.colorbar()
tick_marks = np.arange(len(le.classes_))
plt.xticks(tick_marks, le.classes_, rotation=45)
plt.yticks(tick_marks, le.classes_)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 ha="center", va="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig("docs/confusion_matrix.png", dpi=150)
print("Confusion matrix saved to docs/confusion_matrix.png")

importances = model.feature_importances_
plt.figure(figsize=(10, 4))
plt.plot(importances)
plt.title("Feature Importances — Random Forest")
plt.xlabel("Feature index")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("docs/feature_importances.png", dpi=150)
print("Feature importances saved to docs/feature_importances.png")

print(f"\nSummary:")
print(f"  Model:    Random Forest (100 trees)")
print(f"  Accuracy: {accuracy * 100:.2f}%")
print(f"  Classes:  {list(le.classes_)}")
print(f"  Features: {X.shape[1]}")
