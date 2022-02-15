'''
---------Лабораторная работа 1-----------

ЛОКАЛЬНАЯ ЛИНЕЙНАЯ ФИЛЬТРАЦИЯ ИЗОБРАЖЕНИЙ

--------------Вариант 1------------------

Весовые коэффициенты
    [1, 1, 1,
1/9  1, 1, 1,
     1, 1, 1]

Коэффициенты 
k1,   k2,   k3
0,1   0,6   0,5

'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def showHist_GrayImage(image: np.array) -> None:
    x_brightness, y_frequencies  = np.unique(image, return_counts=True)
    plt.fill_between(x_brightness, 0, y_frequencies, alpha=0.7)
    # plt.plot(x_brightness, y_frequencies)
    plt.show()

if __name__ == '__main__':
    images = [
        '480px-Lenna.png',
        '480px-Lenna_Noise10%.png',
        '480px-Lenna_Noise15%.png',
    ]
    for i in images:
        image = np.array(Image.open(i))
        showHist_GrayImage(image)