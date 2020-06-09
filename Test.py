import numpy as np
import cv2
import skimage
import skimage.measure
import skimage.metrics
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from math import log10, sqrt

# гаусс
def MSE(noise, clear):
    np_noise = np.asarray(noise, dtype="int32").ravel()
    np_clear = np.asarray(clear, dtype="int32").ravel()
    return mean_squared_error(np_noise, np_clear, multioutput='uniform_average')
# гаусс
def PSNR(noise,clear):
    psnr = skimage.metrics.peak_signal_noise_ratio(clear,noise,data_range=255)
    return psnr
# гаусс+пуассон
def SSIM(noise, clear):
    np_noise = np.asarray(noise, dtype="int32").ravel()
    np_clear = np.asarray(clear, dtype="int32").ravel()
    return skimage.metrics.structural_similarity(np_clear,np_noise)

# пуассон
def MLEP(noise, clear):
    x = np.float64(np.ravel(noise).T)
    z = np.float64(np.ravel(clear).T)
    r = np.argwhere(x == 0)
    x = np.delete(x, r)
    z = np.delete(z, r)
    k = x / z

    MLEP = 2 * np.sum(x * np.log(k) - x + z) / (len(x) - 1)
    return MLEP

def calc_gauss(noise,clear):
    mse = MSE(noise,clear)
    psnr = PSNR(noise,clear)
    ssim = SSIM(noise,clear)
    return mse,psnr,ssim

def calc_puasson(noise, clear):
    ssim = SSIM(noise, clear)
    mlep= MLEP(noise,clear)
    return ssim,mlep


def GAUSS(image,mu,s):
    h,w,ch = image.shape
    rand = np.random.normal(mu,s,size=[h,w,ch])
    return image+np.int8(np.fix(rand))


def plot_estimates_gauss(image,rng):
    list_mse = []
    list_psnr = []
    list_ssim = []

    list_mse_filter = []
    list_psnr_filter = []
    list_ssim_filter = []

    mu = 0

    for s in rng:
        if s % 10 == 0:
            print(s)
        # noise = GAUSS(image, mu, s)
        mode_noise = "poisson"
        noise = skimage.util.random_noise(image, mode=mode_noise)
        cv2.imwrite("tmp.png", noise)
        tmp = cv2.imread("tmp.png")
        filter_img = cv2.bilateralFilter(tmp, s, 50, cv2.BORDER_REFLECT)

        mse, psnr, ssim = calc_gauss(noise, image)
        list_mse.append(mse)
        list_psnr.append(psnr)
        list_ssim.append(ssim)
        mse, psnr, ssim = calc_gauss(filter_img, image)
        list_mse_filter.append(mse)
        list_psnr_filter.append(psnr)
        list_ssim_filter.append(ssim)


    print(list_mse)
    print(list_psnr)
    print(list_ssim)

    fig,(p_mse,p_psnr,p_ssim) = plt.subplots(3,1,figsize=(12, 12))
    # p_mse.plot(rng,list_mse)
    p_mse.plot(rng,list_mse_filter)
    # p_psnr.plot(rng,list_psnr)
    p_psnr.plot(rng,list_psnr_filter)
    # p_ssim.plot(rng,list_ssim)

    p_ssim.plot(rng,list_ssim_filter)

    plt.show()

print("Init")
cell_path  = r'Images\Clear\3.jpeg'
mel_path = r'Images\Clear\2.jpg'
cell_image = cv2.imread(cell_path)
mel_image = cv2.imread(mel_path)


# mode_noise = "gaussian"
# cell_noise = skimage.util.random_noise(cell_image, mode=mode_noise,var=100)
# mel_noise = skimage.util.random_noise(mel_image, mode=mode_noise,var=100)


rng = range(1,100)

plot_estimates_gauss(cell_image,rng)









