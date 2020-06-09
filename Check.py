import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


clear = cv2.imread(r'Images/Clear/1.jpeg')
noise = cv2.imread(r'Images/Noise/1.png')
filtered = cv2.imread(r'Images/Filter/1.png')

def puasson(clear,to_check):
    x = np.float64(np.ravel(to_check).T)
    z = np.float64(np.ravel(clear).T)
    r = np.argwhere(x ==0)
    x = np.delete(x,r)
    z = np.delete(z,r)
    k = x/z

    MLEP = 2*np.sum(x*np.log(k) - x + z)/(len(x)-1)
    return MLEP

res_noise = noise - clear
res_filtered = filtered - clear

h,w,ch = res_noise.shape

MSE_noise = np.sum(res_noise**2)/(h*w*ch)
MSE_filtered = np.sum(res_filtered**2)/(h*w*ch)

PSNR_noise = 10*np.log10(255**2*h*w/np.sum(res_noise**2))
PSNR_filtered = 10*np.log10(255**2*h*w/np.sum(res_filtered**2))

MLEP_noise = puasson(clear,noise)
MLEP_filtered = puasson(clear,filtered)



np_noise = np.asarray( noise, dtype="int32" ).ravel()
np_clear = np.asarray( clear, dtype="int32" ).ravel()
np_filter = np.asarray( filtered, dtype="int32" ).ravel()

MSE_noise2 = mean_squared_error(np_noise,np_clear, multioutput='uniform_average')
MSE_filtered2 = mean_squared_error(np_filter,np_clear, multioutput='uniform_average' )

print("MSE_2",MSE_noise2,MSE_filtered2)

y = []
y1 = []
x = []

for i in range(len(np_noise)):
    x.append(i)
    y.append(np_noise[i]-MSE_noise2)
    y1.append(np_filter[i]-MSE_filtered2)


# print("MSE",MSE_noise, MSE_filtered)
# print("PSNR",PSNR_noise, PSNR_filtered)
# print("MLEP",MLEP_noise, MLEP_filtered)

plt.plot(x,y)
plt.plot(x,y1)
plt.grid()
plt.show()


# cv2.imshow("clear",clear)
# cv2.imshow("noise  ",noise)
# cv2.imshow("filtered",filtered)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


