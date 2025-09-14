import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class RBFNetwork:
    def __init__(self, n_neurons=3, sigma=1.0):
        self.n_neurons = n_neurons
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _rbf(self, X, center):
        return np.exp(-self.sigma * np.linalg.norm(X - center, axis=1) ** 2)

    def fit(self, X, y):
        # определение центров
        kmeans = KMeans(n_clusters=self.n_neurons)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        # расчёт активаций скрытого слоя
        activations = np.zeros((len(X), self.n_neurons))
        for i, center in enumerate(self.centers):
            activations[:, i] = self._rbf(X, center)

        # обучение весов
        self.weights = np.linalg.pinv(activations.T @ activations) @ activations.T @ y

    def predict(self, X):
        activations = np.zeros((len(X), self.n_neurons))
        for i, center in enumerate(self.centers):
            activations[:, i] = self._rbf(X, center)
        return activations @ self.weights


# Загрузка данных
data = pd.read_csv('Mall_Customers.csv')

# Преобразование пола в числовой формат
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# Выбор признаков
features_reg = ['Annual Income (k$)', 'Age']
features_clf = ['Annual Income (k$)', 'Age', 'Gender']

# Нормализация данных
scaler_reg = StandardScaler()
X_reg = scaler_reg.fit_transform(data[features_reg])
y_reg = data['Spending Score (1-100)'].values

scaler_clf = StandardScaler()
X_clf = scaler_clf.fit_transform(data[features_clf])
y_clf = data['Gender'].values

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)



