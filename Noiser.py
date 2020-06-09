
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def puasson(clear,to_check):
    x = np.float64(np.ravel(to_check).T)
    z = np.float64(np.ravel(clear).T)
    r = np.argwhere(x ==0)
    x = np.delete(x,r)
    z = np.delete(z,r)
    k = x/z

    MLEP = 2*np.sum(x*np.log(k) - x + z)/(len(x)-1)
    return MLEP

def GAUSS(image,mu,s):
    h,w,ch = image.shape
    rand = np.random.normal(mu,s,size=[h,w,ch])
    return image+np.int8(np.fix(rand))

mu = 0
# s = 9
file = r'Images\Clear\2.jpeg'

image = cv2.imread(file)
h,w,ch = image.shape

np_clear = np.asarray(image, dtype="int32" ).ravel()

MSE_list_n =[]
MSE_list_f =[]

PSNR_list_n =[]
PSNR_list_f =[]

rng = range(1,150)

for s in rng:
    if(s%10==0):
        print(s)
    noise_img = GAUSS(image,mu,s)

    cv2.imwrite("tmp.png",noise_img)
    tmp = cv2.imread("tmp.png")

    filter_img = cv2.bilateralFilter(tmp, 1, 1, cv2.BORDER_REFLECT)

    # cv2.imwrite(r'Images\Noise\3_s={s}.png'.format(s=s), noise_img)
    # cv2.imwrite(r'Images\Filter\3_s={s}.png'.format(s=s), filter_img)

    np_noise = np.asarray(noise_img, dtype="int32").ravel()
    mse_n = mean_squared_error(np_noise, np_clear, multioutput='uniform_average')

    res_noise = noise_img - image
    PSNR_n = 10*np.log10(255**2*h*w/np.sum(res_noise**2))

    np_filter = np.asarray(filter_img, dtype="int32").ravel()
    mse_f = mean_squared_error(np_filter, np_clear, multioutput='uniform_average')

    res_noise = filter_img - image
    PSNR_f = 10*np.log10(255**2*h*w/np.sum(res_noise**2))

    MSE_list_n.append(mse_n)
    MSE_list_f.append(mse_f)

    PSNR_list_n.append(PSNR_n)
    PSNR_list_f.append(PSNR_f)



plt.plot(rng,MSE_list_n)
plt.plot(rng,MSE_list_f)
plt.grid()
plt.show()


# plt.plot(rng,PSNR_list_n)
# plt.plot(rng,PSNR_list_f)
# plt.grid()
# plt.show()





