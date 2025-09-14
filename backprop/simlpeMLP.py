import numpy as np


class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # Ошибка выходного слоя
        dZ2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Ошибка скрытого слоя
        dZ1 = np.dot(dZ2, self.W2.T) * (1 - np.power(self.a1, 2))  # Производная tanh
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Обновление весов
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs=10000, learning_rate=0.1):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)

            if epoch % 1000 == 0:
                loss = np.mean(np.square(output - y))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")


if __name__ == "__main__":
    # Данные XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Создаем и обучаем сеть
    mlp = SimpleMLP(input_size=2, hidden_size=4, output_size=1)
    mlp.train(X, y, epochs=10000, learning_rate=0.1)

    # Тестирование
    print("\nFinal Predictions:")
    for x in X:
        pred = mlp.forward(x.reshape(1, -1))
        print(f"Input: {x}, Output: {pred[0][0]:.4f}")
