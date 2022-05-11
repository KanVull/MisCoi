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

import cv2
import numpy as np
from PIL import Image

# def image_to_binary(image: np.ndarray, threshold=127) -> np.ndarray:
#     '''
#     Convert image to binary representation
#     0 - is above threshold
#     1 - is below or equal threshold

#     input:
#     image:      np.ndarray
#     threshold:  pixel brightness value, that make program 
#                 to understand, place 1 or 0 in that place

#     return:
#         np.ndarray with shape equal to input image  
#         None if input image is not np.ndarray or threshold below 0          
#     '''
#     if not type(image) == np.ndarray:
#         return None
#     if threshold < 0:
#         return None
#     return np.where(image > threshold, 0, 1)

# def show_binary_image(image: np.ndarray) -> None:
#     '''
#     Show image like png object

#     input:
#         image: np.ndarray must contain only 0 and 1 values

#     return:
#         None
#     '''
#     img = np.copy(image)
#     img[img > 0] = 255
#     Image.fromarray(img).convert('L').show()

def get_contoursImage(image: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursImage = np.zeros_like(image)
    cv2.drawContours(contoursImage, contours, -1, (255), 1)
    return contoursImage                      
from test import get_skeleton

def get_ABCD(contoursImage, i, j):
    '''
    returns (a,b,c,d) of distance from edges
    A - Top
    B - Right
    C - Bottom
    D - Left
    '''
    A = B = C = D = 0

    test_i = i
    while contoursImage[test_i, j] == 0:
        test_i += 1
    A = test_i - i

    test_j = j
    while contoursImage[i, test_j] == 0:
        test_j += 1
    B = test_j - j

    test_i = i
    while contoursImage[test_i, j] == 0:
        test_i -= 1
    C = i - test_i

    test_j = j
    while contoursImage[i, test_j] == 0:
        test_j -= 1
    D = j - test_j

    return A,B,C,D    


if __name__ == '__main__':
    img = cv2.imread('FiguresR.png', 0)
    ckeleton = cv2.ximgproc.thinning(img)                
    # imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, image = cv2.threshold(imgray, 127, 255, 0)
    # skeleton = np.zeros_like(image)
    # ckeleton = get_skeleton('FiguresR.png')
    # contoursImage = get_contoursImage(image)
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         if image[i,j] == 0:
    #             A,B,C,D = get_ABCD(contoursImage, i, j)
    #             if -1 < A-C <= 1 or -1 < B-D <= 1:
    #                 skeleton[i,j] = 255

    Image.fromarray(ckeleton).show()

