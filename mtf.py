import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.fftpack import fft
import os
import random

M = 512
dw = 1 / M
s = 500 * 1e-6
w_param = 1

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
    m1 = (125 - y11) / k1 + x11

    k2 = (y22 - y21) / (x22 - x21)
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
    else:
        # ako nemamo slope uzimamo samo jedan red na sredini
        esf = img[250, :]

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

    contrast = (np.max(img) - np.min(img)) / (np.max(img) + np.min(img))
    snr = 20 * np.log10(np.mean(img) / np.std(img))
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    blur = np.var(laplacian)

    return {
        'num': img_number,
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

def analyze(params_list):
    contrasts = [p['contrast'] for p in params_list]
    snrs = [p['snr'] for p in params_list]
    blurs = [p['blur'] for p in params_list]

    mean_contrast = np.sum(contrasts) / 16
    mean_snr = np.sum(snrs) / len(snrs)
    mean_blur = np.sum(blurs) / 16

    contrasts_desc = ['high' if c >= mean_contrast else 'low' for c in contrasts]
    snrs_desc = ['high' if s >= mean_snr else 'low' for s in snrs]
    blurs_desc = ['high' if b >= mean_blur else 'low' for b in blurs]

    w = np.linspace(0, w_param, 256)
    cutoff_f = [w[np.argmax(p['mtf'] < 0.1)] for p in params_list]

    params_desc = []
    for i in range(16):
        param_desc = {
            'img': params_list[i]['img'],
            'mtf': params_list[i]['mtf'],
            'cutoff': cutoff_f[i],
            'num': params_list[i]['num'],
            'slope': params_list[i]['slope'],
            'contrast': contrasts_desc[i],
            'snr': snrs_desc[i],
            'blur': blurs_desc[i],
        }
        params_desc.append(param_desc)
    return params_desc

def print_params(params):
    print('phantom number: ', params['num'])
    print('cutoff freq: ', params['cutoff'])
    print('slope: ', params['slope'])
    print('blur: ', params['blur'])
    print('snr: ', params['snr'])

if __name__ == '__main__':

    file_list = os.listdir('images')
    params_list = []

    w = np.linspace(0, w_param, 256)
    ideal_mtf = np.abs(np.sinc(w/2))

    for file in file_list:
        img = cv2.imread(f'images/{file}', 0)
        img_number = int(file[6:8])
        params = img_params(img, img_number)
        params['img'] = img
        params['mtf'] = MTF(img)
        params_list.append(params)
        # subplot_img_mtf(params['img'], w, params['mtf'], '')

    params_list_desc = analyze(params_list)

    choice = 1
    print('unesi -1 za izlaz')
    choice = int(input('Izaberi broj fantoma >> '))
    while choice != -1:
        subplot_img_mtf(params_list_desc[choice]['img'], w, params_list_desc[choice]['mtf'], f'fantom{choice}')

        # find random image
        idx = random.choice(range(16))
        print(params_list_desc[choice]['snr'])

        while (params_list_desc[choice]['contrast'] != params_list_desc[idx]['contrast'] and params_list_desc[choice]['blur'] == params_list_desc[idx]['blur'] and params_list_desc[choice]['snr'] == params_list_desc[idx]['snr']):
            idx = random.choice(range(16))

        # print_params(params_list_desc[choice])
        # print_params(params_list_desc[idx])

        plt.figure()
        plt.title(f'mtf {choice} i {idx}')
        plt.plot(w, params_list_desc[choice]['mtf'])
        plt.plot(w, params_list_desc[idx]['mtf'])
        plt.plot(w, ideal_mtf)
        plt.legend([f'mtf {choice}', f'mtf {idx}'])
        plt.xlabel('ciklusa/mm')
        plt.ylabel('MTF')
        plt.grid()
        plt.show()

        choice = int(input('Izaberi broj fantoma >> '))

