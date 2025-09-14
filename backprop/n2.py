import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = torch.sigmoid(layer(x))
        x = self.linears[-1](x)
        return x


def get_int_input(prompt, min_val=1):
    while True:
        try:
            value = int(input(prompt))
            if value >= min_val:
                return value
            print(f"Значение должно быть ≥ {min_val}")
        except ValueError:
            print("Введите целое число")


def get_float_list_input(prompt, expected_len):
    while True:
        try:
            values = list(map(float, input(prompt).split(',')))
            if len(values) == expected_len:
                return values
            print(f"Ожидается {expected_len} значений")
        except ValueError:
            print("Некорректный ввод. Пример: 1.0,0.5,-2.3")


def main():
    print("\n--- Конфигурация нейросети ---")
    layers = []

    # Входной слой
    inputs = get_int_input("Количество входов: ")
    layers.append(inputs)

    # Скрытые слои
    hidden_layers = get_int_input("Количество скрытых слоев (0 если нет): ", 0)
    for i in range(hidden_layers):
        neurons = get_int_input(f"Нейронов в скрытом слое {i + 1}: ")
        layers.append(neurons)

    # Выходной слой
    outputs = get_int_input("Количество выходов: ")
    layers.append(outputs)

    # Создаем модель
    model = MLP(layers)
    print(f"\nСоздана модель с архитектурой: {layers}")

    # Тестирование
    while True:
        test = input("\nПротестировать сеть? (y/n): ").lower()
        if test != 'y':
            break

        inputs = get_float_list_input(
            f"Введите {layers[0]} входных значений через запятую: ",
            layers[0]
        )

        with torch.no_grad():
            tensor = torch.FloatTensor(inputs).unsqueeze(0)
            output = model(tensor)
            print("Результат:", output.numpy())


if __name__ == "__main__":
    main()