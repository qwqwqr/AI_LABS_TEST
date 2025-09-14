import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class RBFNetwork:
    """Модель Radial Basis Function Network для регрессии и классификации"""
    def __init__(self, n_neurons=3, sigma=1.0):
        # Инициализация RBF сети
        self.n_neurons = n_neurons
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _rbf(self, X, center):
        # Активация RBF функции (Функция Гаусса)
        return np.exp(-self.sigma * np.linalg.norm(X - center, axis=1) ** 2)

    def fit(self, X, y):
        # Обучение RBF сети

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
        # Предсказание значений для новых данных
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

# предсказание Spending Score по Annual Income и Age
print("Регрессия: предсказание Spending Score")
rbf_reg = RBFNetwork(n_neurons=5, sigma=1.0)
rbf_reg.fit(X_reg_train, y_reg_train)

# Оценка на тестовых данных
y_reg_pred = rbf_reg.predict(X_reg_test)
mse = np.mean((y_reg_pred - y_reg_test) ** 2)
print(f"Mean Squared Error: {mse:.2f}")

# Визуализация 1
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_reg_test[:, 0], y_reg_test, label="Реальные значения")
plt.scatter(X_reg_test[:, 0], y_reg_pred, color='red', label="Предсказания")
plt.xlabel("Annual Income (нормализованный)")
plt.ylabel("Spending Score")
plt.title("Предсказание Spending Score по Annual Income")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_reg_test[:, 1], y_reg_test, label="Реальные значения")
plt.scatter(X_reg_test[:, 1], y_reg_pred, color='red', label="Предсказания")
plt.xlabel("Age (нормализованный)")
plt.ylabel("Spending Score")
plt.title("Предсказание Spending Score по Age")
plt.legend()

plt.tight_layout()
plt.show()

# Классификация предсказание пола по Annual Income, Age и Spending Score
print("\nКлассификация: предсказание пола")
rbf_clf = RBFNetwork(n_neurons=5, sigma=1.0)
rbf_clf.fit(X_clf_train, y_clf_train)

# Оценка на тестовых данных
y_clf_pred = rbf_clf.predict(X_clf_test)
y_clf_pred_class = (y_clf_pred > 0.5).astype(int)  # Преобразование в бинарные классы

accuracy = np.mean(y_clf_pred_class == y_clf_test)
print(f"Accuracy: {accuracy:.2f}")

# Визуализация 2
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_clf_test[y_clf_test == 0, 0], X_clf_test[y_clf_test == 0, 1],
            label="Female (реальные)", alpha=0.5)
plt.scatter(X_clf_test[y_clf_test == 1, 0], X_clf_test[y_clf_test == 1, 1],
            label="Male (реальные)", alpha=0.5)
plt.xlabel("Annual Income (нормализованный)")
plt.ylabel("Age (нормализованный)")
plt.title("Реальные значения пола")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_clf_test[y_clf_pred_class == 0, 0], X_clf_test[y_clf_pred_class == 0, 1],
            label="Female (предсказанные)", alpha=0.5)
plt.scatter(X_clf_test[y_clf_pred_class == 1, 0], X_clf_test[y_clf_pred_class == 1, 1],
            label="Male (предсказанные)", alpha=0.5)
plt.xlabel("Annual Income (нормализованный)")
plt.ylabel("Age (нормализованный)")
plt.title("Предсказанные значения пола")
plt.legend()

plt.tight_layout()
plt.show()
