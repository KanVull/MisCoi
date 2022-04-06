'''
ЛАБОРАТОРНАЯ РАБОТА №3
СКЕЛЕТИЗАЦИЯ И УТОНЬШЕНИЕ БИНАРНЫХ ИЗОБРАЖЕНИЙ

Тестируемое изображение: Figures.png
Фигуры по варианту:
 - Квадрат
 - Крест
 - Решетка
 - Буква А
 - Буква D
 - Цифра 2

'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def image_to_binary(image: np.ndarray, threshold=127) -> np.ndarray:
    '''
    Convert image to binary representation
    0 - is above threshold
    1 - is below or equal threshold

    input:
    image:      np.ndarray
    threshold:  pixel brightness value, that make program 
                to understand, place 1 or 0 in that place

    return:
        np.ndarray with shape equal to input image  
        None if input image is not np.ndarray or threshold below 0          
    '''
    if not type(image) == np.ndarray:
        return None
    if threshold < 0:
        return None
    return np.where(image > threshold, 1, 0)

def show_binary_image(image: np.ndarray) -> None:
    '''
    Create and show matplotlib plot with image only

    input:
        image: np.ndarray must contain only 0 and 1 values

    return:
        None
    '''
    img = np.copy(image)
    img[img > 0] = 255
    axe = plt.subplot()
    axe.imshow(image, cmap='gray')
    axe.axis('off')
    plt.show()



if __name__ == '__main__':
    image = np.array(Image.open('Figures.png').convert('L'))
    binimage = image_to_binary(image)
    show_binary_image(binimage)
