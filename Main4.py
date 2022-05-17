'''
ЛАБОРАТОРНАЯ РАБОТА №4
СЕГМЕНТАЦИЯ ИЗОБРАЖЕНИЯ И ВЫДЕЛЕНИЕ КОНТУРОВ

Тестируемое изображение: 480px-Lenna.png

Метод raise_area -  это выращивание областей по признаку вхождения в диапазон
                    +- некоторое значение от значения первого проверяемого
                    пикселя в этой области.
                    (почти) Рекурсивный метод, который уходит от начального
                    пикселя в 4 стороны и проверяет их на условие вхождения
                    в диапазон +- число от значения первого пикселя в области

Метод get_contours -    это реализованный алгоритм обработки изображения
                        опертором Собеля     

'''

import numpy as np
from PIL import Image

def raise_area(image, pixel_value_range = (10,10)) -> np.ndarray:
    checked = np.zeros_like(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if checked[x,y] != 0:
                continue
            else:
                stack_of_coordinates = [(x,y)]
                value = image[x, y]
                while len(stack_of_coordinates) != 0:
                    x_st, y_st = stack_of_coordinates.pop(0)
                    checked[x_st, y_st] = 1
                    if value - pixel_value_range[0] <= image[x_st, y_st]  <= value + pixel_value_range[1]:
                        image[x_st, y_st] = value
                        if x_st != 0:
                            if checked[x_st-1, y_st] == 0:
                                stack_of_coordinates.append((x_st-1, y_st))
                                checked[x_st-1, y_st] = 1
                        if x_st != image.shape[0] - 1:
                            if checked[x_st+1, y_st] == 0:
                                stack_of_coordinates.append((x_st+1, y_st)) 
                                checked[x_st+1, y_st] = 1
                        if y_st != 0:
                            if checked[x_st, y_st-1] == 0:
                                stack_of_coordinates.append((x_st, y_st-1))
                                checked[x_st, y_st-1] = 1
                        if y_st != image.shape[1] - 1:
                            if checked[x_st, y_st+1] == 0:
                                stack_of_coordinates.append((x_st, y_st+1))
                                checked[x_st, y_st+1] = 1

    return image                

def get_contours(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.int32)
    width = image.shape[0] - 3
    height = image.shape[1] - 3
    for x in range(width):
        for y in range(height):
            image[x, y] = 0.5 * (
                (image[x, y] - image[x, y+2] + 2 * (image[x+1, y] - image[x+1, y+2]) + image[x+2, y] - image[x+2, y+2])**2 + 
                (image[x, y] - image[x+2, y] + 2 * (image[x, y+1] - image[x+2, y+1]) + image[x, y+2] - image[x+2, y+2])**2
            )**0.5
    image[image > 255] = 255
    image[image < 0]   = 0
    image = image.astype(np.uint8)
    return image        


if __name__ == '__main__':
    image = Image.open('480px-Lenna.png').convert('L')
    image = np.array(image)
    raised_area_image = raise_area(image.copy(), pixel_value_range=(40,40))
    Image.fromarray(raised_area_image).show()
    contours_image = get_contours(image.copy())
    Image.fromarray(contours_image).show()