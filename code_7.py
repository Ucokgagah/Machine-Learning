# Install library (jalankan di terminal/command prompt, bukan di dalam script Python)
# !pip install pandas seaborn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Map target integers to species names
data['species'] = data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display first few rows
print(data.head())

# Summary statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Pairplot
sns.pairplot(data, hue='species')
plt.show()
# Heatmap of correlation
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()
# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x='species', y='sepal length (cm)', data=data)
plt.show()
# Violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='species', y='sepal width (cm)', data=data)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
# Split data into training and testing sets
X = data.drop('species', axis=1)
y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
random_state=42)
# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
# Predict on test set
y_pred = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_test, y_pred) 
print("ClassificationReport:\n", class_report)
