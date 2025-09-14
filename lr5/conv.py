import numpy as np


def convolution_2d(input_matrix, kernel, stride=1, padding=0):
    """Реализация свертки    """
    # Добавляем padding
    if padding > 0:
        input_padded = np.pad(input_matrix, pad_width=padding, mode='constant')
    else:
        input_padded = input_matrix

    # Размеры изображения и ядра
    input_h, input_w = input_padded.shape
    kernel_h, kernel_w = kernel.shape

    # Вычисляем размеры изображения
    output_h = (input_h - kernel_h) // stride + 1
    output_w = (input_w - kernel_w) // stride + 1

    # Инициализируем массив
    output = np.zeros((output_h, output_w))

    # Применяем свертку
    for i in range(0, output_h):
        for j in range(0, output_w):
            # Вычисляем позиции
            h_start = i * stride
            h_end = h_start + kernel_h
            w_start = j * stride
            w_end = w_start + kernel_w

            # Извлекаем окно
            window = input_padded[h_start:h_end, w_start:w_end]

            # Вычисляем свертку
            output[i, j] = np.sum(window * kernel)

    return output


input_matrix = np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 16]])

kernel = np.array([[1, 0],
                   [0, -1]])

result = convolution_2d(input_matrix, kernel, stride=1, padding=0)
print("Результат свертки:")
print(result)
