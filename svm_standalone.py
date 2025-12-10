# =============================================================================
# KODE 1: SVM STANDALONE (BASELINE) - NOISE BAWAAN
# =============================================================================

# 1. SETUP & LOAD DATA
!pip install opendatasets --quiet
import opendatasets as od
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

dataset_url = 'https://www.kaggle.com/datasets/abdallaahmed77/healthcare-risk-factors-dataset'
if not os.path.exists('healthcare-risk-factors-dataset'):
    od.download(dataset_url)

dataset_dir = 'healthcare-risk-factors-dataset'
csv_file = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(dataset_dir, csv_file))

# 2. PREPROCESSING
feature_cols = ['Age', 'Gender', 'Glucose', 'Blood Pressure', 'BMI', 'Oxygen Saturation',
                'Cholesterol', 'Triglycerides', 'Smoking', 'Alcohol', 'Physical Activity',
                'Stress Level', 'Sleep Hours', 'LengthOfStay', 'HbA1c',
                'Diet Score', 'Family History', 'noise_col']

# Bersihkan Missing Values
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Encoding
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}).fillna(0).astype(int)
if 'Family History' in df.columns:
    df['Family History'] = df['Family History'].fillna(0).astype(int)

# Target Encoding
df = df.dropna(subset=['Medical Condition'])
le = LabelEncoder()
df['Medical Condition'] = le.fit_transform(df['Medical Condition'])
target_classes = le.classes_

# Pastikan hanya kolom yang ada yang dipakai
final_features = [c for c in feature_cols if c in df.columns]
X = df[final_features]
y = df['Medical Condition']

# Split & Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Data Siap. Fitur: {len(final_features)} (Termasuk Noise Bawaan)")

# 3. TRAINING SVM STANDALONE
print("\nMelatih SVM Standalone...")
svm_model = SVC(kernel='rbf', C=10, probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# 4. EVALUASI
y_pred = svm_model.predict(X_test_scaled)
y_prob = svm_model.predict_proba(X_test_scaled)

# A. Tabel Metrik
print("\n=== HASIL: SVM STANDALONE ===")
metrics_df = pd.DataFrame({
    'Metrik': ['Akurasi', 'Presisi (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
    'Nilai': [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, average='macro'),
        recall_score(y_test, y_pred, average='macro'),
        f1_score(y_test, y_pred, average='macro')
    ]
})
display(metrics_df)

# B. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_classes, yticklabels=target_classes)
plt.title('Confusion Matrix: SVM Standalone')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.show()

# C. ROC Curve (Multi-class)
y_test_bin = label_binarize(y_test, classes=range(len(target_classes)))
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Micro-average AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve: SVM Standalone')
plt.legend()
plt.show()
