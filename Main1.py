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
from tracemalloc import start
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def showHist_GrayImage(image: np.array) -> None:
    # hist = np.histogram(image, bins=range(image.min(), image.max()))
    hist = np.histogram(image, bins=range(256))
    plt.plot(hist[0], hist[1][:-1])
    plt.show()

if __name__ == '__main__':
    images = [
        '480px-Lenna.png',
        '480px-Lenna_Noise10%.png',
        '480px-Lenna_Noise15%.png',
    ]
    image = np.array(Image.open(images[0]))
    showHist_GrayImage(image)