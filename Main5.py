'''
ЛАБОРАТОРНАЯ РАБОТА №5
ВЫЧИСЛЕНИЕ ПРИЗНАКОВ ИЗОБРАЖЕНИЙ

Тестируемое изображение: 480px-Lenna.png
 

'''
import math
import numpy as np
from PIL import Image

def P_ST(image, sigma=1, teta=0):
    f_max = image.max()
    f_min = image.min()
    answer = np.zeros((f_max-f_min+1, f_max-f_min+1))
    for i in range(image.shape[0]):
        for j in range(image.shape[1] - 1):
            b_1 = image[i, j]
            b_2 = image[i, j+1]
            answer[b_1 - f_min, b_2 - f_min] += 1
    answer += answer.T
    return answer / answer.sum()         


image = Image.open('480px-Lenna.png').convert('L')
image = np.array(image)
m = image.max() - image.min()
if m == 0: m = 1
P = P_ST(image)
P[P==0] = 0.00000000000000000000001
#### K - контраст
K = 0
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        K += (i - j)**2 * P[i,j] 
#### M - Второй угловой момент
M = (P**2).sum()
#### R - коэффициент корреляции
pass
#### H - энтропия
H = -((P * np.log2(P)).sum())
#### D - дисперсия
D = 0
G = P.sum() / (m**2)
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        D += (i - G)**2 * P[i,j] 
#### Mg - инверстный дифф момент
Mg = 0
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        Mg += P[i,j] / ((i - j)**2 + 1)

del P

image = Image.open('car.jpg').convert('L')
image = np.array(image)
##### Геометрия
s = image.sum()
#### x_center y_center - центр тяжести
x_center = 0.0
y_center = 0.0
for i in range(image.shape[0]):
    x_center += i * image[i].sum()
r_image = np.rot90(image, 3)
for j in range(r_image.shape[0]):
    y_center += j * r_image[j].sum()
x_center /= s    
y_center /= s
del r_image
#### Mx My - момент инерции объекта
Mx = 0.0
My = 0.0
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        Mx += (i - x_center)**2 * image[i,j] 
        My += (j - y_center)**2 * image[i,j]        
#### Mxy - смешанный момент инерции
Mxy = 0.0
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        Mxy += (i - x_center)*(j - y_center) * image[i,j]   
#### M1 M2 - Главный момент инерции
M1 = (Mx + My) / 2.0 + ((Mx-My)**2 / 4 + Mxy**2)**0.5
M2 = (Mx + My) / 2.0 - ((Mx-My)**2 / 4 + Mxy**2)**0.5              
print()



