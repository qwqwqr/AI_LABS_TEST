import numpy as np
import matplotlib.pyplot as plt


def conv2d(input, kernel, stride=1, padding=0):
    """Реализация 2D свертки"""
    # Добавление padding
    if padding > 0:
        input = np.pad(input, ((padding, padding), (padding, padding)),
                       mode='constant')

    # Размеры входных данных и ядра
    in_h, in_w = input.shape
    k_h, k_w = kernel.shape

    # Расчет выходных размеров
    out_h = (in_h - k_h) // stride + 1
    out_w = (in_w - k_w) // stride + 1

    # Инициализация выхода
    output = np.zeros((out_h, out_w))

    # Применение свертки
    for i in range(0, out_h):
        for j in range(0, out_w):
            h_start = i * stride
            w_start = j * stride
            window = input[h_start:h_start + k_h, w_start:w_start + k_w]
            output[i, j] = np.sum(window * kernel)

    return output


def maxpool2d(input, pool_size=2, stride=2):
    """Реализация 2D Max Pooling"""
    # Размеры входных данных
    in_h, in_w = input.shape

    # Расчет выходных размеров
    out_h = (in_h - pool_size) // stride + 1
    out_w = (in_w - pool_size) // stride + 1

    # Инициализация выхода
    output = np.zeros((out_h, out_w))

    # Применение пулинга
    for i in range(0, out_h):
        for j in range(0, out_w):
            h_start = i * stride
            w_start = j * stride
            window = input[h_start:h_start + pool_size, w_start:w_start + pool_size]
            output[i, j] = np.max(window)

    return output


# Пример использования
input_img = np.array([[1, 2, 3, 4, 5],
                      [6, 7, 8, 9, 10],
                      [11, 12, 13, 14, 15],
                      [16, 17, 18, 19, 20],
                      [21, 22, 23, 24, 25]])

kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

# Применяем свертку
conv_result = conv2d(input_img, kernel, stride=1, padding=1)
print("Результат свертки:")
print(conv_result)

# Применяем MaxPooling
pool_result = maxpool2d(conv_result, pool_size=2, stride=2)
print("\nРезультат MaxPooling:")
print(pool_result)