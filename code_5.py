from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier()

param_distributions = {
    'n_estimators': np.arange(10, 200, 10),
    'max_depth': np.arange(1, 20),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(model, param_distributions, n_iter=100, cv=5,
                                   scoring='accuracy', random_state=42)

random_search.fit(X, y)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)