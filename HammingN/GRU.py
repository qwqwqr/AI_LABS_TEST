import torch
import torch.nn as nn
import numpy as np



def generate_sequence(length=100):
    x = np.linspace(0, 10, length)
    y = np.sin(x) + np.random.normal(0, 0.1, length)
    return y.astype(np.float32)


data = generate_sequence()


def create_dataset(data, window_size=5):
    X = np.array([data[i:i + window_size] for i in range(len(data) - window_size)])
    y = np.array([data[i + window_size] for i in range(len(data) - window_size)])

    # Конвертируем в тензор один раз для всего набора
    return torch.from_numpy(X).unsqueeze(-1), torch.from_numpy(y)


X, y = create_dataset(data)
print(f"Форма данных: X={X.shape}, y={y.shape}")


class SimpleGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.fc(hidden.squeeze(0))


model = SimpleGRU()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs.squeeze(), y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')


test_seq = torch.tensor([[[0.5], [0.6], [0.7], [0.8], [0.9]]], dtype=torch.float32)
prediction = model(test_seq)
print(f"Предсказание: {prediction.item():.4f}")
