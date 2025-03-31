import torch
import torch.nn as nn
import joblib
import pandas as pd


# Модуль для запуска нейронной сети
class MushroomClassifier(nn.Module):
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


# Модель и метаданные
model = MushroomClassifier(input_size=22)
model.load_state_dict(torch.load('mushroom_classifier.pth', weights_only=False))
model.eval()

metadata = joblib.load('model_metadata.pkl')
encoders = metadata['encoders']

test_data = pd.read_csv('mushrooms.csv')

X_test_raw = test_data.drop('class', axis=1)
y_test_raw = test_data['class']

X_test_encoded = pd.DataFrame()
for col in X_test_raw.columns:
    le = encoders[col]
    X_test_encoded[col] = le.transform(X_test_raw[col])

y_test_encoded = encoders['class'].transform(y_test_raw)

X_test_tensor = torch.FloatTensor(X_test_encoded.values)
y_test_tensor = torch.LongTensor(y_test_encoded)

with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predictions = torch.max(outputs.data, 1)
    predictions = predictions.numpy()

y_test_decoded = encoders['class'].inverse_transform(y_test_encoded)
predictions_decoded = encoders['class'].inverse_transform(predictions)

sample_indexes = range(1, 11)
for idx in sample_indexes:
    print(f"\nN:{idx}")
    print("Истинное значение:", y_test_decoded[idx])
    print("Предсказание:", predictions_decoded[idx])
