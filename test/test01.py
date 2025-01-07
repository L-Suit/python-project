import random
import cv2
import numpy as np


def get_noise(img, value=10):
    noise = np.random.uniform(0, 256, img.shape[0:2])
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)
    return noise

def rain_blur(noise, length=10, angle=0, w=1):
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))
    k = cv2.warpAffine(dig, trans, (length, length))
    k = cv2.GaussianBlur(k, (w, w), 0)
    blurred = cv2.filter2D(noise, -1, k)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred
def alpha_rain(rain, img, beta=0.8):
    rain = np.expand_dims(rain, 2)
    rain_effect = np.concatenate((img, rain), axis=2)  # add alpha channel
    rain_result = img.copy()
    rain = np.array(rain, dtype=np.float32)
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    return rain_result

def main():
    image = cv2.imread(r'/root/python-project/data/img/5(7).jpg')
    cv2.imshow("Display Window",image)
    cv2.waitKey(0)

    # 应用高斯模糊效果
    #blurred_image = cv2.GaussianBlur(image, (9, 9), 0)  # (15, 15) 是卷积核的大小，可以调整

    # noise = get_noise(image, value=500)
    # rain = rain_blur(noise, length=50, angle=-25, w=3)
    # rain_result = alpha_rain(rain, image, beta=0.6)
    #
    # img_processed = rain_result / 255
    # img_processed = np.clip(img_processed * 255, 0, 255)
    # img_processed = img_processed.astype(np.uint8)
    #
    # cv2.imshow("Display Window",img_processed)
    # cv2.waitKey(0)

if __name__ == '__main__':
    main()