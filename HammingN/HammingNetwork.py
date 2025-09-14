import random
import matplotlib.pyplot as plt

import numpy as np


def visualization(test_vec, max_neurons=4):
    plt.figure(figsize=(10, 5))

    for num_neurons in range(1, max_neurons + 1):
        subset_prototypes = prototypes[:num_neurons]

        net = HammingNetwork(num_neurons, subset_prototypes)
        net.train()

        winner = net.predict(test_vec)

        plt.subplot(1, max_neurons, num_neurons)
        plt.bar(range(num_neurons), net.bias, color='lightblue', label='Другой вариант')
        plt.bar(winner, net.bias[winner], color='blue', label='Не спам')
        plt.title(f'{num_neurons} нейрона\nКласс: {winner}')
        plt.xlabel('Нейрон')
        plt.ylabel('Активация')
        plt.ylim(0, 6)
        if num_neurons == 1:
            plt.legend()

    plt.tight_layout()
    plt.show()


class HammingNetwork:
    """ Сеть Хемминга"""
    def __init__(self, num_neurons, prototype_vectors):
        # Инициализация
        self.num_neurons = num_neurons
        self.prototypes = np.array(prototype_vectors)
        self.weights = None
        self.bias = None

    def train(self):
        # Обучение сети
        n_features = self.prototypes.shape[1]
        self.weights = self.prototypes.T / 2
        self.bias = np.ones(self.num_neurons) * n_features / 2

    def predict(self, input_vector, max_iter=10):
        # Слой сравнения
        similarity = np.dot(self.weights.T, input_vector) + self.bias

        # Конкурентный слой
        activations = np.zeros(self.num_neurons)
        winner = np.argmax(similarity)
        activations[winner] = 1

        for _ in range(max_iter):
            new_activations = np.copy(activations)
            new_activations[winner] = 0
            next_winner = np.argmax(similarity * (new_activations == 0))

            if similarity[next_winner] < similarity[winner]:
                break
            winner = next_winner

        return winner


keywords = ["кошка", "собака", "тигр", "лев", "волк", "лиса", "медведь", "выиграй", "бесплатно", "приз"]

normal_message = [1, 1, -1, -1, -1, -1, -1, -1, -1, -1]
spam_message = [-1, -1, -1, -1, -1, -1, -1, 1, 1, 1]
mixed_message = [1, -1, 1, -1, -1, -1, -1, 1, -1, -1]
t = [random.choice([-1, 1]) for i in range(10)]

prototypes = [
    normal_message,  # Индекс 0 — не спам
    spam_message,  # Индекс 1 — спам
    mixed_message,  # Индекс 2 — смешанное
]

test_vec = [1, -1, -1, 1, -1, -1, -1, 1, -1, -1]
visualization(test_vec, max_neurons=3)
print(HammingNetwork(3, test_vec))
