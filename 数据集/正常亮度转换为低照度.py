import os
import shutil
import random
import cv2
from pathlib import Path
import numpy as np


# 设置你的源文件夹和目标文件夹
source_folder = Path('D:\dataset\ip102\Detection\VOC2007\images')  # 替换为你的源文件夹路径
dest_folder_unaltered = Path('D:\dataset\ip102-low-light/high')  # 替换为未调整图像的目标文件夹路径
dest_folder_altered = Path('D:\dataset\ip102-low-light/low')  # 替换为调整过图像的目标文件夹路径

# 创建目标文件夹如果它们不存在
dest_folder_unaltered.mkdir(parents=True, exist_ok=True)
dest_folder_altered.mkdir(parents=True, exist_ok=True)

# 获取所有图片文件
image_files = list(source_folder.glob('*.jpg'))  # 假设图片是jpg格式，根据需要调整
random.shuffle(image_files)  # 随机打乱图片顺序

# 确定一半的图片数量
half_size = len(image_files) // 2

# 复制未调整的图像
for image_path in image_files[:half_size]:
    shutil.copy(str(image_path), str(dest_folder_unaltered / image_path.name))
    print("复制",image_path)


gamma = 0.7  # 伽马值小于1会使图像变暗
invGamma = 1.0 / gamma
table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

# 调整并复制剩余的图像
for image_path in image_files[half_size:]:
    # 读取图像
    image = cv2.imread(str(image_path))
    # alpha < 1 降低对比度，beta < 0 降低亮度
    dark_image = cv2.convertScaleAbs(image, alpha=0.8, beta=-40)
    # gamma校正
    dark_gamma_image = cv2.LUT(dark_image, table)
    # 保存调整后的图像
    cv2.imwrite(str(dest_folder_altered / image_path.name), dark_gamma_image)
    print("调整并复制",image_path)

print("图像处理完成。未调整的图像已复制到：", dest_folder_unaltered)
print("调整过的图像已复制到：", dest_folder_altered)
