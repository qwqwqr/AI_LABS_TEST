import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Загрузка данных
data = pd.read_csv('mushrooms.csv')

# Создаем и сохраняем кодировщики
encoders = {}
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Разделяем на признаки и класс
X = data.drop('class', axis=1).to_numpy()
y = data['class'].to_numpy()

# Разделяем данные на тестовые и проверочные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)


class MushroomClassifier(nn.Module):
    """Модель на основе алгоритма обратного распространения ошибки backpropagation c оптимизацией Adam"""
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


input_size = X_train.shape[1]
model = MushroomClassifier(input_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
train_losses = []
test_accuracies = []
# Обучение модели 50 эпох
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs.data, 1)
        correct = (predicted == y_test).sum().item()
        accuracy = correct / y_test.size(0)
        test_accuracies.append(accuracy)

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Сохранение модели
torch.save(model.state_dict(), 'mushroom_classifier.pth')

# Сохранение метаданных
metadata = {
    'encoders': encoders,
    'feature_order': list(data.columns.drop('class')),
    'target_encoder': encoders['class'],
    'input_size': input_size
}

joblib.dump(metadata, 'model_metadata.pkl')
