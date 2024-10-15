import random

import cv2


if __name__ == "__main__":
    # model = YOLO(r"D:\Pycharm-project\pythonProject\目标检测yolo\runs\detect\yolov8n_weather_test5\weights\last.pt")
    # for param in model.parameters():
    #     print(param.dtype)  # 这将打印出每个参数的数据类型
    #     break
    # #model = YOLO(r"D:\Pycharm-project\pythonProject\目标检测yolo\runs\detect\train\weights\last.pt")
    # model.val(data="mydataset-IP102.yaml")

    # img
    image = cv2.imread(r"D:\dataset\ip102\Detection\VOC2007\images\IP000000200.jpg")
    dark_factor = random.uniform(0.7, 0.8)
    image = cv2.convertScaleAbs(image, alpha=dark_factor, beta=0)

    cv2.imshow("image", image)
    cv2.waitKey(0)


