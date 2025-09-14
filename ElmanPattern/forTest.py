import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader


# 1. Генерация данных
def generate_data(num_samples=500, seq_len=10):
    """Генерация последовательностей и меток (1 если есть паттерн [1,0,1])"""
    X = np.random.randint(0, 2, size=(num_samples, seq_len))
    y = np.zeros(num_samples)
    for i in range(num_samples):
        for j in range(seq_len - 2):
            if X[i, j] == 1 and X[i, j + 1] == 0 and X[i, j + 2] == 1:
                y[i] = 1
                break
    return torch.FloatTensor(X), torch.FloatTensor(y)


# 2. Создание моделей
class ElmanRNN(nn.Module):
    """Сеть Элмана"""

    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        _, hidden = self.rnn(x)
        return torch.sigmoid(self.fc(hidden.squeeze(0)))


class MLP(nn.Module):
    """Многослойный перцептрон"""

    def __init__(self, input_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.layers(x))


# 3. Функция обучения и оценки
def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_type, epochs=100):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    # Обучение
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Оценка
    model.eval()
    with torch.no_grad():
        # На тренировочных данных
        train_preds = (model(X_train) > 0.5).float()
        train_acc = accuracy_score(y_train, train_preds)

        # На тестовых данных
        test_preds = (model(X_test) > 0.5).float()
        test_acc = accuracy_score(y_test, test_preds)

    print(f"\n{model_type} Результаты:")
    print(f"  Точность на обучающей выборке: {train_acc:.4f}")
    print(f"  Точность на тестовой выборке: {test_acc:.4f}")

    return test_acc


# 4. Основной эксперимент
def main():
    # Генерация данных
    X, y = generate_data()
    X_train, X_test = X[:420], X[420:]
    y_train, y_test = y[:420], y[420:]

    # Инициализация моделей
    rnn_model = ElmanRNN()
    mlp_model = MLP()

    # Обучение и оценка
    print("=== Сравнение моделей ===")
    print("Задача: Обнаружение паттерна [1, 0, 1] в последовательности")
    print(f"Всего примеров: {len(X)} (Обучающих: {len(X_train)}, Тестовых: {len(X_test)})")

    rnn_acc = train_and_evaluate(rnn_model, X_train, X_test, y_train, y_test, "Сеть Элмана")
    mlp_acc = train_and_evaluate(mlp_model, X_train, X_test, y_train, y_test, "Многослойный перцептрон")

    # Сравнение производительности
    print("\n=== Итоговое сравнение ===")
    print(f"Сеть Элмана показала точность: {rnn_acc:.4f}")
    print(f"MLP показал точность: {mlp_acc:.4f}")
    print(f"Разница: {(rnn_acc - mlp_acc):.4f} в пользу {'сети Элмана' if rnn_acc > mlp_acc else 'MLP'}")

    # Примеры предсказаний
    test_sequences = [
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Должен вернуть 1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Должен вернуть 0
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Должен вернуть 0
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0]  # Должен вернуть 1
    ]

    print("\nПримеры предсказаний сети Элмана:")
    for seq in test_sequences:
        seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
        with torch.no_grad():
            prob = rnn_model(seq_tensor).item()
        print(
            f"Последовательность: {seq} -> {'Есть паттерн' if prob > 0.5 else 'Нет паттерна'} (вероятность: {prob:.4f})")


main()