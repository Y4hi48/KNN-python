import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("Employee.csv")
data = data.drop("City", axis=1)

categorical_columns = data.select_dtypes(include=["object"]).columns
numerical_columns = data.select_dtypes(exclude=["object"]).columns

categorical_imputer = SimpleImputer(strategy="most_frequent")
numerical_imputer = SimpleImputer(strategy="mean")

data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])
data[numerical_columns] = numerical_imputer.fit_transform(data[numerical_columns])

label_encoders = [LabelEncoder() for _ in categorical_columns]
for i, col in enumerate(categorical_columns):
    data[col] = label_encoders[i].fit_transform(data[col])

X = data.iloc[:, :-1].values.astype(float)
y = data.iloc[:, -1].values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class KNN:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common, _ = np.unique(k_nearest_labels, return_counts=True)
        return most_common.argmax()

knn = KNN(n_neighbors=3)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy:", accuracy)
