from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
# Memuat dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
# Mengatur model
model = LogisticRegression(max_iter=200)

#Mengatur K-Fold Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# Melakukan cross-validation dan menghitung skor
scores = cross_val_score(model, X, y, cv=kfold)
print(f'K-Fold Cross-validation Scores: {scores}')
print(f'Mean Score: {scores.mean()}')

from sklearn.model_selection import StratifiedKFold
# Mengatur Stratified K-Fold Cross-validation
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Melakukan cross-validation dan menghitung skor
stratified_scores = cross_val_score(model, X, y, cv=stratified_kfold)
print(f'Stratified K-Fold Cross-validation Scores: {stratified_scores}')
print(f'Mean Score: {stratified_scores.mean()}')

from sklearn.model_selection import LeaveOneOut
# Mengatur LOOCV
loocv = LeaveOneOut()
# Melakukan cross-validation dan menghitung skor
loocv_scores = cross_val_score(model, X, y, cv=loocv)
print(f'Leave-One-Out Cross-validation Mean Score: {loocv_scores.mean()}')

from sklearn.model_selection import RepeatedKFold
# Mengatur Repeated K-Fold Cross-validation
repeated_kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
# Melakukan cross-validation dan menghitung skor
repeated_scores = cross_val_score(model, X, y, cv=repeated_kfold)
print(f'Repeated K-Fold Cross-validation Mean Score: {repeated_scores.mean()}')

