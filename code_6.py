from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import StandardScaler

# Memuat dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# Membagi dataset menjadi data latih dan uji (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Penskalaan fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definisi model
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

# Evaluasi model dengan cross-validation (cv=5)
cv_results = {}
for model_name, model in models.items():
    if model_name == 'SVM':  # SVM memerlukan data yang diskalakan
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    else:  # Model lain menggunakan data asli
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
    cv_results[model_name] = np.mean(scores)
    print(f"{model_name}: {np.mean(scores)}")

# Menentukan model terbaik berdasarkan hasil cross-validation
best_model_name = max(cv_results, key=cv_results.get)
best_model = models[best_model_name]

# Melatih model terbaik di seluruh data latih
if best_model_name == 'SVM':
    best_model.fit(X_train_scaled, y_train)
    test_score = best_model.score(X_test_scaled, y_test)
else:
    best_model.fit(X_train, y_train)
    test_score = best_model.score(X_test, y_test)

# Menampilkan hasil model terbaik
print(f"Best Model: {best_model_name}")
print(f"Test Set Accuracy: {test_score}")
