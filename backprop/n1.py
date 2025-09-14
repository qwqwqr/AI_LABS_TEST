import numpy as np

class FormalNeuron:
    def __init__(self, activation_func='linear'):
        self.weights = np.array([1.0, -1.0])  # Веса: левый вход (белый) +1, правый (черный) -1
        self.bias = -0.5  # Порог

        if activation_func == 'linear':
            self.activation = lambda x: x
        elif activation_func == 'step':
            self.activation = lambda x: 1 if x >= 0 else 0
        elif activation_func == 'sigmoid':
            self.activation = lambda x: 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unknown activation function")

    def predict(self, inputs):
        # inputs: [левый_пиксель, правый_пиксель], где 0 - белый, 1 - черный
        net_input = np.dot(inputs, self.weights) + self.bias
        return self.activation(net_input)


# Тестирование нейрона
def test_neuron(activation_func):
    print(f"\nTesting with {activation_func} activation:")
    neuron = FormalNeuron(activation_func)

    # Тестовые случаи: [левый, правый]
    test_cases = [
        ([0, 0], "Белый | Белый - Нет границы"),
        ([0, 1], "Белый | Черный - Граница (должен сработать)"),
        ([1, 0], "Черный | Белый - Обратная граница"),
        ([1, 1], "Черный | Черный - Нет границы")
    ]

    for inputs, description in test_cases:
        output = neuron.predict(inputs)
        print(f"{description}: {output:.2f}")


# Проведем эксперименты с разными функциями активации
test_neuron('linear')
test_neuron('step')
test_neuron('sigmoid')