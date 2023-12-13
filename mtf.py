import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.fftpack import fft
import os

M = 512
dw = 1 / M
s = 500 * 1e-6

def MTF(img):
    # nalazenje pragova 
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
    
    k1 = (y12 - y11) / (x12 - x11)
    # m1 = (maksimum - y11) / k1 + x11
    m1 = (125 - y11) / k1 + x11

    k2 = (y22 - y21) / (x22 - x21)
    # m2 = (maksimum - y21) / k2 + x21
    m2 = (125 - y21) / k2 + x21
    
    br = 0
    for i in range(512):
        if img[500, i] > 150:
            br += 1
        if img[10, i] > 150:
            break

    # d = np.abs(m1 - m2)
    d = br * 0.5
    dx = d / 489; 

    if x11 == x12 and x21 == x22:
        N = 1
    else:
        N = int(np.floor(1 / dx))

    # ESF
    img_cut = img[:N, :]

    if N == 1:
        esf = np.reshape(img_cut.T, N * 512)
        # w = np.arange(0, N * M / 2 * dw, dw)
    else:
        # ako nemamo slope uzimamo samo jedan red na sredini
        esf = img[250, :]
        # w = np.arange(0, (M / 2 - 1) * dw, dw)

    # LSF
    lsf = np.gradient(esf)

    # MTF
    mtf = fft(lsf)
    mtf_normalized = np.abs(mtf[: ((N * 512) // 2)] / mtf[0])
    mtf_normalized = mtf_normalized[:M//2]
    return mtf_normalized


def img_params(img, img_number):
    # slope-> poredimo prvu pojavu belog piksela u prvom i poslednjem redu
    first = 0
    last = 0
    slope = 0

    for i in range(512):
        if img[0, i] > 100: # prag za beli pixel je 100
            first = i
            break
        if img[511, i] > 100:
            last = i
            break
    if first != last:
        slope = 1

    # Blur
    for i in range(512):
        if img[100, i] > 40:
            first = i
            break
    for i in range(first, 512):
        if img[100, i] > 220:
            second = i
            break
    blur = second - first

    # Contrast
    black_region = img[30:120, 30:120]
    white_region = img[390:480, 390:480]

    contrast = np.mean(black_region) - np.mean(white_region)

    black_var = (black_region - np.mean(black_region))**2
    white_var = (white_region - np.mean(white_region))**2

    std_black = np.sqrt(np.sum(black_var) / black_region.size)

    snr = 20 * np.log10(np.sum(white_var) / std_black) 

    return {
        'img_number': img_number,
        'slope': slope,
        'contrast': contrast,
        'snr': snr,
        'blur': blur,
    }

def subplot_img_mtf(img, x, y, title: str = ''):
    ideal_mtf = np.abs(np.sinc(x))
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title(title)
    plt.plot(x, y)
    plt.plot(x, ideal_mtf)
    plt.xlabel('ciklusa/mm')
    plt.ylabel('MTF')
    plt.grid()
    plt.show()

def test():
    img = cv2.imread(f'images/fantom11.bmp', 0)
    params = img_params(img)
    mtf = MTF(img)
    plt.plot(mtf)
    plt.show()

if __name__ == '__main__':

    file_list = os.listdir('images')
    params_list = []

    w = np.linspace(0, 2, 256)
    ideal_mtf = np.sinc(w)

    for file in file_list:
        img = cv2.imread(f'images/{file}', 0)
        img_number = int(file[6:8])
        params = img_params(img, img_number)
        params_list.append(params)

        mtf = MTF(img)
        subplot_img_mtf(img, w, mtf, '')

   
    # ideal = np.sinc(np.linspace(0, 512, 100))
    # plt.plot(ideal)
    # plt.show()


