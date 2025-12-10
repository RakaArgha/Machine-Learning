# =============================================================================
# KODE 2: HYBRID RF-SVM (METODE USULAN) - NOISE BAWAAN [UPDATED]
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from matplotlib.patches import Patch

# 1. LOAD DATA
dataset_dir = 'healthcare-risk-factors-dataset'
csv_file = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(dataset_dir, csv_file))

# 2. PREPROCESSING
feature_cols = ['Age', 'Gender', 'Glucose', 'Blood Pressure', 'BMI', 'Oxygen Saturation',
                'Cholesterol', 'Triglycerides', 'Smoking', 'Alcohol', 'Physical Activity',
                'Stress Level', 'Sleep Hours', 'LengthOfStay', 'HbA1c',
                'Diet Score', 'Family History', 'noise_col'] # Menggunakan noise bawaan

# Cleaning
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}).fillna(0).astype(int)
if 'Family History' in df.columns:
    df['Family History'] = df['Family History'].fillna(0).astype(int)

df = df.dropna(subset=['Medical Condition'])
le = LabelEncoder()
df['Medical Condition'] = le.fit_transform(df['Medical Condition'])
target_classes = le.classes_

final_features = [c for c in feature_cols if c in df.columns]
X = df[final_features]
y = df['Medical Condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. HYBRID STEP 1: FEATURE SELECTION (RF)
print("\nMelatih Random Forest untuk Seleksi Fitur...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# Visualisasi Feature Importance AWAL (Semua Fitur)
importances = rf.feature_importances_
fi_df = pd.DataFrame({'Feature': final_features, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
# Warna Merah untuk 'noise_col', Teal untuk lainnya
colors = ['red' if 'noise' in x else 'teal' for x in fi_df['Feature']]
plt.barh(fi_df['Feature'], fi_df['Importance'], color=colors)
plt.gca().invert_yaxis()
plt.title('Feature Importance Awal (Sebelum Seleksi)', fontsize=14)
legend_elements = [Patch(facecolor='teal', label='Fitur Medis'), Patch(facecolor='red', label='Noise Bawaan')]
plt.legend(handles=legend_elements, loc='lower right')
plt.show()

# 4. HYBRID STEP 2: FILTER & TRAIN SVM
# Ambang batas (threshold)
threshold = 0.02
selected_feats = fi_df[fi_df['Importance'] > threshold]['Feature'].tolist()

print(f"\nHasil Seleksi (Threshold > {threshold}):")
print(f"   -> Fitur Terpilih ({len(selected_feats)}): {selected_feats}")
print(f"   -> Fitur Dibuang: {fi_df[fi_df['Importance'] <= threshold]['Feature'].tolist()}")

# --- VISUALISASI FITUR TERPILIH ---
fi_selected = fi_df[fi_df['Importance'] > threshold]

plt.figure(figsize=(10, 5))
plt.barh(fi_selected['Feature'], fi_selected['Importance'], color='teal')
plt.gca().invert_yaxis() # Urutkan dari yang paling penting di atas
plt.title(f'Feature Importance Terpilih (Threshold > {threshold})', fontsize=14)
plt.xlabel('Importance Score')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
# ----------------------------------------------------

# Filter Dataset
selected_indices = [final_features.index(f) for f in selected_feats]
X_train_hybrid = X_train_scaled[:, selected_indices]
X_test_hybrid = X_test_scaled[:, selected_indices]

print("\nMelatih Hybrid SVM pada Fitur Terpilih...")
svm_hybrid = SVC(kernel='rbf', C=10, probability=True, random_state=42)
svm_hybrid.fit(X_train_hybrid, y_train)

# 5. EVALUASI
y_pred_h = svm_hybrid.predict(X_test_hybrid)
y_prob_h = svm_hybrid.predict_proba(X_test_hybrid)

# A. Tabel Metrik
print("\n=== HASIL: HYBRID RF-SVM ===")
metrics_hybrid = pd.DataFrame({
    'Metrik': ['Akurasi', 'Presisi (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
    'Nilai': [
        accuracy_score(y_test, y_pred_h),
        precision_score(y_test, y_pred_h, average='macro'),
        recall_score(y_test, y_pred_h, average='macro'),
        f1_score(y_test, y_pred_h, average='macro')
    ]
})
display(metrics_hybrid)

# B. Confusion Matrix
plt.figure(figsize=(8, 6))
cm_h = confusion_matrix(y_test, y_pred_h)
sns.heatmap(cm_h, annot=True, fmt='d', cmap='Greens',
            xticklabels=target_classes, yticklabels=target_classes)
plt.title('Confusion Matrix: Hybrid RF-SVM')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.show()

# C. ROC Curve
y_test_bin = label_binarize(y_test, classes=range(len(target_classes)))
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_prob_h.ravel())
roc_auc_h = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'Micro-average AUC = {roc_auc_h:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve: Hybrid RF-SVM')
plt.legend()
plt.show()
