import torch
import torch.nn as nn
import torch.optim as optim


training_data = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=torch.float32)

labels = torch.tensor([[0.0], [1.0], [0.0], [0.0]], dtype=torch.float32)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


model = SimpleNN()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

epochs = 1000
for epoch in range(epochs):
    outputs = model(training_data)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

test_data = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=torch.float32)

with torch.no_grad():
    prediction = model(test_data)
    print('Предсказания модели')
    for i, inputs in enumerate(test_data):
        print(f"Вход: {inputs.tolist()}, Выход: {prediction[i].item():.4f}")
