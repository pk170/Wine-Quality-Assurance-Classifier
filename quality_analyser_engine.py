
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load Dataset
try:
    df = pd.read_csv('wine_data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'wine_data.csv' not found. Please ensure the file is in the same folder.")
    exit()

# Data Cleaning & Preprocessing
df.columns = df.columns.str.strip()

df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Normalize chemical values)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Random Forest Model trained successfully.")

# Evaluate Performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n Quality Assurance Report:")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualisation
importances = model.feature_importances_
feature_names = X.columns
indices = importances.argsort()

plt.figure(figsize=(10, 6))
plt.title('Feature Importance (Drivers of Wine Quality)')
plt.barh(range(len(indices)), importances[indices], color='maroon', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature Importance plot saved as 'feature_importance.png'")