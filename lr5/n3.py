import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Загрузка и подготовка данных CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# Классы CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 2. Создание модели (CNN + MaxPool + MLP)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Сверточные слои с MaxPooling
        self.conv_layers = nn.Sequential(
            # Блок 1: Conv -> ReLU -> MaxPool
            nn.Conv2d(3, 16, 3, padding=1),  # 3, 16, 3x3
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Уменьшение в 2 раза

            # Блок 2: Conv -> ReLU -> MaxPool
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Блок 3: Conv -> ReLU -> MaxPool
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Полносвязнй слой (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),  # 64 канала * 4x4
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 выходных классов
        )

    def forward(self, x):
        # Применяем сверточные слои
        x = self.conv_layers(x)

        # Выравниваем для MLP
        x = x.view(x.size(0), -1)

        # Применяем полносвязные слои
        x = self.mlp(x)
        return x


# 3. Инициализация модели, функции потерь и оптимизатора
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 4. Обучение модели
def train(model, trainloader, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499:  # Печатаем каждые 500 мини-батчей
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 500:.3f}')
                running_loss = 0.0


# Обучаем 5 эпох
train(model, trainloader, epochs=5)


# 5. Оценка на тестовых данных
def test(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test images: {100 * correct / total:.2f}%')


test(model, testloader)


# 6. Визуализация предсказаний
def visualize_predictions(model, testloader, num_images=6):
    model.eval()
    images, labels = next(iter(testloader))
    outputs = model(images[:num_images])
    _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(2, 3, i + 1)
        img = images[i].numpy().transpose((1, 2, 0))
        img = img * 0.5 + 0.5  # Денормализация
        plt.imshow(img)
        plt.title(f'True: {classes[labels[i]]}\nPred: {classes[preds[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


visualize_predictions(model, testloader)