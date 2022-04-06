'''
ЛАБОРАТОРНАЯ РАБОТА №2
ВИДОИЗМЕНЕНИЕ ГИСТОГРАММ

Тестируемое изображение: 480px-Lenna.png
Значения параметров по варианту:
    g_min:  10
    g_max:  250
    a:      ±0.6
    k:      ±0.25

Реализованы формулы:
    1. Линейное функциональное отображение
        -- linear_function_mapping
    2. Равномерное распределение
        -- uniform_distribution
    3. Экспоненциальное распределение
        -- exponential_distribution
    4. Распределение Рэлея
        -- Rayleigh_distribution
    5. Распределение степени 2/3
        -- degree_2_dev_3_distribution
    6. Гиперболическое распределение
        -- hyperbolic_distribution
    7. Степенная интенсификация
        -- power_intensification
'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def showHist_GrayImage(image: np.array, showImage=False) -> tuple:
    if showImage:
        fig, axes = plt.subplots(1, 2)
        axes[1].imshow(image, cmap='gray')
        axes[1].axis('off')
    else:
        fig, axes = plt.subplot(1, 1)
    hist = np.unique(image, return_counts=True)
    axes[0].fill_between(hist[0], 0, hist[1], alpha=1)
    plt.show()

def linear_function_mapping(image: np.array, g: dict) -> np.array:
    f = {'min': image.min(), 'max': image.max()}
    return (((g['max'] - g['min']) / (f['max'] - f['min'])) * (image - f['min'])).astype(np.uint8)

def probability(image: np.array) -> dict:
    p = list(np.unique(image, return_counts=True))
    p[1] = p[1].astype(np.float64)
    s = p[1].sum()
    for i in range(len(p[1])):
        p[1][i] = p[1][i] / s
        if i>0:
            p[1][i] += p[1][i-1] 
    p[1][-1] -= 0.0000001        
    return {p[0][i]: p[1][i] for i in range(len(p[0]))}        

def uniform_distribution(image: np.array, g: dict) -> np.array:
    p = probability(image)
    img = np.zeros_like(image).astype(np.float64)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            img[x][y] = (g['max'] - g['min']) * p[image[x][y]] + g['min']
    return img.astype(np.uint8)        

def exponential_distribution(image: np.array, g: dict, a: float) -> np.array:
    p = probability(image)
    img = np.zeros_like(image).astype(np.float64)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            img[x][y] = g['min'] - 1/a * np.log(1 - p[image[x][y]])
    return img.astype(np.uint8)    

def Rayleigh_distribution(image: np.array, g: dict, a: float) -> np.array:
    p = probability(image)
    img = np.zeros_like(image).astype(np.float64)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            img[x][y] = g['min'] + (2 * a**2 * np.log( 1/( 1-p[image[x][y]] ) ))**0.5
    return img.astype(np.uint8)      

def degree_2_dev_3_distribution(image: np.array, g: dict) -> np.array:
    p = probability(image)
    img = np.zeros_like(image).astype(np.float64)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            img[x][y] = ( (g['max']**0.33 - g['min']**0.33) * p[image[x][y]] + g['min']**0.33 )**3
    return img.astype(np.uint8)      

def hyperbolic_distribution(image: np.array, g: dict) -> np.array:
    p = probability(image)
    img = np.zeros_like(image).astype(np.float64)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            img[x][y] = g['min'] * (g['max'] / g['min'])**p[image[x][y]] 
    return img.astype(np.uint8)  

def power_intensification(image: np.array, g: dict, k: float) -> np.array:
    p = probability(image)
    img = np.zeros_like(image).astype(np.float64)
    dev = sum([pr**k for pr in p.values()])
    f_min = image.min()
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            ch = 0
            for i in range(f_min, image[x][y] + 1):
                if i in p.keys():
                    ch += p[i]**k
            img[x][y] = ((g['max'] - g['min']) * ch) / dev + g['min']
    return img.astype(np.uint8) 

if __name__ == '__main__':
    g = {'min': 10.0, 'max': 250.0}
    # g = {'min': 30.0, 'max': 200.0}
    a = 0.6
    # a = 2
    k = 0.25
    k = 1.5
    image = np.array(Image.open('480px-Lenna.png').convert('L'))
    # showHist_GrayImage(image, showImage=True)
    # lfm_image = linear_function_mapping(image, g)
    # showHist_GrayImage(lfm_image, showImage=True)
    # ud_image = uniform_distribution(image, g)
    # showHist_GrayImage(ud_image, showImage=True)
    # ed_image = exponential_distribution(image, g, a)
    # showHist_GrayImage(ed_image, showImage=True)
    # Rd_image = Rayleigh_distribution(image, g, a)
    # showHist_GrayImage(Rd_image, showImage=True)
    # d2d3d_image = degree_2_dev_3_distribution(image, g)
    # showHist_GrayImage(d2d3d_image, showImage=True)
    # hd_image = hyperbolic_distribution(image, g)
    # showHist_GrayImage(hd_image, showImage=True)
    pi_image = power_intensification(image, g, k)
    showHist_GrayImage(pi_image, showImage=True)