from collections import Counter
import numpy as np


distance = lambda x, y: np.linalg.norm(y - x, axis=1)

class KNN:
    def __init__(self, n_neighbors=2):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        predicted_labels = [-1] *len(X)
        for i in range(len(X)):
            point = X[i]
            distances = distance(point, self.X_train)
            point_neighbors = np.argsort(distances)[:k]
            types = self.y_train[point_neighbors]

            cant = Counter(types)
            cant_sorted = sorted(cant.items(), key=lambda x: x[1], reverse=True)
            predicted_labels[i] = cant_sorted[0][0]
    
        return predicted_labels

x_train = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [5, 5],
    [6, 7],
    [10,5]
])

y_train = np.array(['label1', 'label2', 'label2', 'label1', 'label2','label3'])

X = np.array([
    [5, 5],
    [2, 3],
    [3, 4],
    [2, 2],
    [6, 7]
])
k = 3

model = KNN(2)
model.fit(x_train, y_train)
model.predict(X)