import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

def GAUSS(image,mu,s):

    h,w,ch = image.shape
    rand = np.random.normal(mu,s,size=[h,w,ch])
    return image+np.int8(np.fix(rand))



cell_path  = r'Images\Clear\3.jpeg'
mel_path = r'Images\Clear\2.jpg'


image = cv2.imread(mel_path)

mu =0
s = 20

rng = range(1, 30)
rng2 = range(1, 100)

print("!")
noise_img = GAUSS(image, mu, s)
cv2.imwrite("tmp.png",noise_img)
tmp = []

for num in rng2:
    tmp.append(cv2.imread("tmp.png"))

sigma_color =1
sigma_space =1

times = []


print("!")
for sigma_color in rng:
    print(sigma_color,end=" - ")

    start_time = time.time()
    for img in tmp:
        image = cv2.bilateralFilter(img, sigma_color, sigma_space, cv2.BORDER_REFLECT)
    times.append(time.time()-start_time)
    print(time.time()-start_time)


print("!!")

plt.plot(rng,times)
plt.grid()
plt.show()