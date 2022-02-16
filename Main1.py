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

def getHist_GrayImage(image: np.array) -> tuple:
    return np.unique(image, return_counts=True)

def deNoise_nonRec(image: np.array, mask: tuple) -> np.array:
    newImage = image.copy()
    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):
            newImage[x,y] = np.trunc(np.sum(image[x-1:x+2, y-1:y+2] * mask) / np.sum(mask))
    return newImage  

def deNoise_Rec(image: np.array, mask: tuple) -> np.array:
    newImage = image.copy()
    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):
            newImage[x,y] = np.trunc(np.sum(newImage[x-1:x+2, y-1:y+2] * mask) / np.sum(mask))
    return newImage

def deNoise_Rec2(image: np.array, mask: tuple, k: float) -> np.array:
    newImage = image.copy()
    nonRec = deNoise_nonRec(newImage, mask)
    Rec = deNoise_Rec(newImage, mask)
    newImage = np.trunc(k * nonRec + (1 - k) * Rec)
    return newImage     


if __name__ == '__main__':
    mask = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    k1, k2, k3 = 0.1, 0.6, 0.5

    images = [
        '480px-Lenna.png',
        '480px-Lenna_Noise10%.png',
        '480px-Lenna_Noise15%.png',
    ]
    image1 = np.array(Image.open(images[0]).convert('L'))
    image2 = np.array(Image.open(images[1]).convert('L'))
    image3 = np.array(Image.open(images[2]).convert('L'))
    deNoise_image1 = deNoise_Rec2(image1, mask, k3)
    deNoise_image2 = deNoise_Rec2(image2, mask, k3)
    deNoise_image3 = deNoise_Rec2(image3, mask, k3)
    hist_image1 = getHist_GrayImage(image1)
    hist_image2 = getHist_GrayImage(image2)
    hist_image3 = getHist_GrayImage(image3)
    hist_deNoise1 = getHist_GrayImage(deNoise_image1)
    hist_deNoise2 = getHist_GrayImage(deNoise_image2)
    hist_deNoise3 = getHist_GrayImage(deNoise_image3)

    fig, axes = plt.subplots(4, 3)
    axes[0, 0].imshow(image1, cmap='gray')
    axes[0, 1].imshow(image2, cmap='gray')
    axes[0, 2].imshow(image3, cmap='gray')
    axes[0, 0].axis('off')
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')
    axes[2, 0].imshow(deNoise_image1, cmap='gray')
    axes[2, 1].imshow(deNoise_image2, cmap='gray')
    axes[2, 2].imshow(deNoise_image3, cmap='gray')
    axes[2, 0].axis('off')
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')
    axes[1, 0].fill_between(hist_image1[0], 0, hist_image1[1], alpha=1)
    axes[1, 1].fill_between(hist_image2[0], 0, hist_image2[1], alpha=1)
    axes[1, 2].fill_between(hist_image3[0], 0, hist_image3[1], alpha=1)
    axes[3, 0].fill_between(hist_deNoise1[0], 0, hist_deNoise1[1], alpha=1)
    axes[3, 1].fill_between(hist_deNoise2[0], 0, hist_deNoise2[1], alpha=1)
    axes[3, 2].fill_between(hist_deNoise3[0], 0, hist_deNoise3[1], alpha=1)
    plt.show()