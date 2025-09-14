import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score


# 1. Оптимизированная генерация данных
def generate_data(num_samples=1000, seq_len=5):
    """Генерация последовательностей и меток (1 если есть паттерн [1,0,1])"""
    # Генерируем все последовательности сразу
    X = np.random.randint(0, 2, size=(num_samples, seq_len))

    # Проверяем наличие паттерна [1,0,1] в каждой последовательности
    y = np.zeros(num_samples)
    for i in range(num_samples):
        for j in range(seq_len - 2):
            if X[i, j] == 1 and X[i, j + 1] == 0 and X[i, j + 2] == 1:
                y[i] = 1
                break

    # Конвертируем в тензоры одним преобразованием
    X_tensor = torch.from_numpy(X).float().unsqueeze(-1)  # форма (N, seq_len, 1)
    y_tensor = torch.from_numpy(y).float()
    return X_tensor, y_tensor


X, y = generate_data()
print(f"Пример данных:\n{X[:3]}\nСоответствующие метки: {y[:3]}")

# 2. Разделение данных (остается без изменений)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# 3. Модель сети Элмана
class ElmanPatternDetector(nn.Module):
    """Модель сети Элмана"""

    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, hidden = self.rnn(x)
        return self.sigmoid(self.fc(hidden.squeeze(0)))


model = ElmanPatternDetector()

# 4. Обучение
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train(model, X, y, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X).squeeze()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            preds = (outputs > 0.5).float()
            acc = accuracy_score(y.numpy(), preds.detach().numpy())
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')


train(model, X_train, y_train, epochs=100)


# 5. Оценка
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X).squeeze()
        preds = (outputs > 0.5).float()
        acc = accuracy_score(y.numpy(), preds.numpy())
        print(f'Test Accuracy: {acc:.4f}')


evaluate(model, X_test, y_test)

# 6. Проверка на конкретных примерах
test_sequences = [
    [1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 1, 1, 0]
]


def predict(model, sequence):
    model.eval()
    with torch.no_grad():
        seq_array = np.array(sequence, dtype=np.float32)
        seq_tensor = torch.from_numpy(seq_array).unsqueeze(-1).unsqueeze(0)
        output = model(seq_tensor)
        return "Есть паттерн [1,0,1]" if output.item() > 0.5 else "Нет паттерна"


for seq in test_sequences:
    print(f"Последовательность: {seq} -> {predict(model, seq)}")
