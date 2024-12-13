##从数据集中随机加上 阴天、雨天、雾天的效果，并保存到新的数据集中
import math
import os
import random
from shutil import copyfile

import cv2
import numpy as np
from PIL import Image

# 设置随机种子，以保证每次运行得到相同的随机序列
# random.seed(42)  # 使用固定的随机种子


# 定义图片处理函数
def process_image(image_path, output_folder, process_type):
    image = cv2.imread(image_path)
    # 如果输出目录中已经有该图片，则跳过处理
    if os.path.exists(os.path.join(output_folder, os.path.basename(image_path))):
        print(f"Skip {os.path.join(output_folder, os.path.basename(image_path))}")
        return

    # 根据process_type定义不同的处理方式
    if process_type == 'origin':                    # 保持原始图片
        img_processed = image
    elif process_type == 'rain':                    # 雨天效果
        # 降低亮度对比度
        image = cv2.convertScaleAbs(image, alpha=0.7, beta=-10)
        # value = random.randint(300, 600)
        noise = get_noise(image, value=500)
        rain = rain_blur(noise, length=50, angle=-25, w=3)
        rain_result = alpha_rain(rain, image, beta=0.6)

        img_processed = rain_result / 255
        img_processed = np.clip(img_processed * 255, 0, 255)
        img_processed = img_processed.astype(np.uint8)
    elif process_type == 'fog':                    # 雾天效果
        i = random.randint(0, 9)
        img_processed = image / 255
        (row, col, chs) = image.shape
        A = 0.5

        beta = 0.01 * i + 0.05
        size = math.sqrt(max(row, col))
        center = (row // 2, col // 2)
        if row * col > 810000:
            foggy_image = AddHaz_loop_largeimg(img_processed, center, size, beta, A)
        else:
            foggy_image = AddHaz_loop(img_processed, center, size, beta, A)
        img_processed = np.clip(foggy_image * 255, 0, 255)
        img_processed = img_processed.astype(np.uint8)
    elif process_type == 'dark':                      # 低光效果
        # 降低亮度
        dark_factor = random.uniform(0.75, 0.85)
        image = cv2.convertScaleAbs(image, alpha=dark_factor, beta=0)
        img_processed = image / 255  # 归一化

        # 伽马变换
        r=random.uniform(1.5,5)
        dark_image=Dark_loop(img_processed,r)
        img_processed=np.clip(dark_image*255, 0, 255) # 限制范围在(0,255)内
        img_processed = img_processed.astype(np.uint8)
    elif process_type == 'fuzzy':                   # 添加模糊效果
        # 应用高斯模糊效果
        img_processed = cv2.GaussianBlur(image, (9, 9), 0)  # (15, 15) 是卷积核的大小，可以调整

    # 构建输出路径
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, img_processed)


def record_allocation(record_file, allocation):
    with open(record_file, 'w') as file:
        for image_path, group in allocation.items():
            file.write(f"{image_path},{group}\n")


def read_allocation(record_file):
    allocation = {}
    if os.path.exists(record_file):
        with open(record_file, 'r') as file:
            for line in file:
                path, group = line.strip().split(',')
                allocation[path] = int(group)
    return allocation

def AddHaz_loop(img_f, center, size, beta, A):
    (row, col, chs) = img_f.shape  # (H,W,C)
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    return img_f

# 针对高分辨率图片调整参数
def AddHaz_loop_largeimg(img_f, center, size, beta, A):
    beta = 0.01 * random.randint(0, 4) + 0.04
    print('fog beta=', beta)
    (row, col, chs) = img_f.shape  # (H,W,C)
    for j in range(row):
        for l in range(col):
            d = -0.024 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    return img_f
def Dark_loop(img_f, r):
    (row, col, chs) = img_f.shape  # (H,W,C)
    for j in range(row):  # 遍历每一行
        for l in range(col):  # 遍历每一列
            img_f[j][l][:] = img_f[j][l][:] ** r
    return img_f

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
    original_images_folder = r'D:\dataset\forest-31-pests\train2017'  # 原始图片所在的文件夹
    new_dataset_folder = r'D:\dataset\mypest-test\images\train'  # 新的数据集存放位置
    allocation_record = './for31_mypest_test_train_record.csv'  # 分配记录文件路径

    # 获取所有图片文件的路径`
    image_paths = [os.path.join(original_images_folder, f) for f in os.listdir(original_images_folder) if
                   f.endswith('.jpg')]

    # 读取分配记录或重新分配
    allocation = read_allocation(allocation_record)
    if not allocation:
        # 随机打乱图片路径列表
        random.shuffle(image_paths)

        # 分成5部分
        lenth = len(image_paths)

        # 分割比例 0.16:0.21：0.21:0.21:0.21
        p1 = int(lenth*0.16)
        p2 = int(lenth*0.37)
        p3 = int(lenth*0.58)
        p4 = int(lenth*0.79)


        parts = [
            image_paths[:p1],
            image_paths[p1:p2],
            image_paths[p2:p3],
            image_paths[p3:p4],
            image_paths[p4:]
        ]

        # 记录分配情况
        for i, part in enumerate(parts):
            for image_path in part:
                allocation[image_path] = i

        # 记录分配
        record_allocation(allocation_record, allocation)

    # 对每部分图片进行处理
    count = 0
    process_types = ['origin', 'rain', 'fog', 'dark', 'fuzzy']
    for i in range(5):
        os.makedirs(new_dataset_folder, exist_ok=True)

        for image_path, group in allocation.items():
            if group == i:
                process_image(image_path, new_dataset_folder, process_types[i])
                count += 1
                if count %100 == 0:
                    print("处理进度：",count)

    print(f"处理完成！,共处理{count}张图片")


if __name__ == "__main__":
    main()