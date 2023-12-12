import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.fftpack import fft

M = 512
dw = 1 / M
s = 500 * 1e-6

def MTF(img, slope):
    
    for i in range(512):
        if img[10, i] > 50:
            x11 = i
            y11 = img[10, i]
            break

    for i in range(512):
        if img[500, i] > 50:
            x21 = i
            y21 = img[500, i]
            break

    for i in range(512):
        if img[10, i] > 200:
            x12 = i
            y12 = img[10, i]
            break

    for i in range(512):
        if img[500, i] > 200:
            x22 = i
            y22 = img[500, i]
            break
    
    maksimum = np.max(img) 
    minimum = np.min(img)

    k1 = (y12 - y11) / (x12 - x11)
    m1 = (maksimum - y11) / k1 + x11

    k2 = (y22 - y21) / (x22 - x21)
    m2 = (maksimum - y21) / k2 + x21
    
    d = np.abs(m1 - m2)
    
    dx = d / 489; 

    if x11 == x12 and x21 == x22:
        N = 1
    else:
        N = int(np.floor(1 / dx))

    # mala slika
    img_cut = img[:N, :]

    if slope == 1:
        ESF = np.reshape(img_cut, N * 512)
        w = np.arange(0, N * M / 2 * dw, dw)
    else:
        ESF = img[250, :]
        w = np.arange(0, (M / 2 - 1) * dw, dw)

    # print(x11, x12, x21, x22)
    # print(y11, y12, y21, y22)
    # print(k1, k2, m1, m2)
    # print(f'X11 = {x11} Y11 = {y11} X21 = {x21} Y21 = {y21}')

    # LSF
    LSF = np.gradient(ESF, dw)

    # MTF
    MTF = fft(LSF)
    MTF /= MTF[0]
    MTF = np.abs(MTF)
    MTF = MTF[:len(w)]
    return MTF

def img_desc(img):
    contrast = []
    blur = []
    snr = []

    # slope-> poredimo prvu pojavu belog piksela u prvom i poslednjem redu
    first = 0
    last = 0

    for i in range(512):
        if img[0, i] < 100: # prag za beli pixel je 100
            first = i
        if img[511, i] < 100:
            last = i
        if first != last: # ????
            slope = 1
            break

    # Contrast
    black_region = img[30:120, 30:120]
    white_region = img[390:480, 390:480]
    
    contrast = np.mean(black_region) - np.mean(white_region)

    black_var = np.var(black_region)
    white_var = np.var(white_region)

    if black_var == 0:
        snr = 0
    else:
        snr = 20 * np.log10(np.sum(white_region) / black_var) 
    # print(snr)

    # [mtf, _] = MTF(img, slope)
    # index = np.where(mtf < 0.1)[0][0]

if __name__ == '__main__':
    img = cv2.imread('images/fantom01.bmp', 0)
    MTF(img, 1)
    img_desc(img)
