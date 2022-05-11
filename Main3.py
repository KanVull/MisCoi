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
from PIL import Image
import cv2

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
    return np.where(image > threshold, 0, 1)

def show_binary_image(image: np.ndarray) -> None:
    '''
    Show image like png object

    input:
        image: np.ndarray must contain only 0 and 1 values

    return:
        None
    '''
    img = np.copy(image)
    img[img > 0] = 255
    Image.fromarray(img).convert('L').show()

def B(direction: int, n_array: list) -> int:
    '''
        n_array
        n3 - n2 - n1
        |    |     |
        n4 - P  - n0
        |    |     |
        n5 - n6 - n7

        Calculate boolian number of skeletonization
        B0 = n4 (n2 v n3 v n5 v n6)(n6 v ~n7)(~n1 v n2)
        B2 = n6 (n0 v n4 v n5 v n7)(n0 v ~n1)(~n3 v n4)
        B4 = n0 (n1 v n2 v n6 v n7)(n2 v ~n3)(~n5 v n6)
        B6 = n2 (n0 v n1 v n3 v n4)(n4 v ~n5)(~n6 v n7) 

        input:
            direction: int - 0, 2, 4 or 6 
            n_array: list - one-dementional array of 0 and 1
                            from n0 to n7

        return:
            0 or 1 in case of direction B
    '''
    n = n_array
    match direction:
        case 0:
            return n[4] and (n[2] or n[3] or n[5] or n[6]) and (n[6] or not n[7]) and (not n[1] or n[2])
        case 2:
            return n[6] and (n[0] or n[4] or n[5] or n[7]) and (n[0] or not n[1]) and (not n[3] or n[4])
        case 4:
            return n[0] and (n[1] or n[2] or n[6] or n[7]) and (n[2] or not n[3]) and (not n[5] or n[6])
        case 6:
            return n[2] and (n[0] or n[1] or n[3] or n[4]) and (n[4] or not n[5]) and (not n[6] or n[7])
        case _:
            return None

def get_contoursImage(image: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursImage = np.zeros_like(image)
    cv2.drawContours(contoursImage, contours, -1, (255), 1)
    return contoursImage                      

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
    image = np.array(Image.open('Figures.png').convert('L'))
    binimage = image_to_binary(image)
    for i in range(1, binimage.shape[0]-1):
        for j in range(1, binimage.shape[1]-1):
            if binimage[i,j] == 1:
                n_array = [ 
                    binimage[i,   j+1], #n0
                    binimage[i-1, j+1], #n1
                    binimage[i-1, j  ], #n2
                    binimage[i-1, j-1], #n3
                    binimage[i,   j-1], #n4
                    binimage[i+1, j-1], #n5
                    binimage[i+1, j  ], #n6
                    binimage[i+1, j+1], #n7
                ]
                binimage[i,j] = B(0, n_array) or B(4, n_array)      
    skeleton = cv2.ximgproc.thinning(img) 
    # show_binary_image(binimage)
    Image.fromarray(skeleton).show()
