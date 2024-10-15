import os
import shutil
import random
import cv2
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance

# 设置你的源文件夹和目标文件夹
source_folder = Path('D:\dataset\ip102\Detection\VOC2007\images')  # 替换为你的源文件夹路径
dest_folder_unaltered = Path('D:\dataset/ip102-low-light-1000/trainB')  # 替换为未调整图像的目标文件夹路径
dest_folder_altered = Path('D:\dataset/ip102-low-light-1000/trainA')  # 替换为调整过图像的目标文件夹路径

# 创建目标文件夹如果它们不存在
dest_folder_unaltered.mkdir(parents=True, exist_ok=True)
dest_folder_altered.mkdir(parents=True, exist_ok=True)

# 获取所有图片文件
image_files = list(source_folder.glob('*.jpg'))  # 假设图片是jpg格式，根据需要调整
random.shuffle(image_files)  # 随机打乱图片顺序

# 确定需要处理的图片数量
# half_size = len(image_files) // 2
size = 1000

# 复制未调整的图像
for image_path in image_files[:size]:
    shutil.copy(str(image_path), str(dest_folder_unaltered / image_path.name))
    print("复制",image_path)

# 设置亮度和对比度的降低因子，小于1的值会降低亮度和对比度
# 生成随机数，要求随机数在0到2之间

brightness_factor = 0.25
brightness_change_factor = 0.1 # 亮度降低为原来的25%，再上下浮动±0.1
contrast_factor = 0.7    # 对比度降低为原来的 %

gamma = 0.9  # 伽马值小于1会使图像变暗
invGamma = 1.0 / gamma
table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

# 调整并复制剩余的图像
for image_path in image_files[len(image_files)-size:]:
    # # 读取图像
    # image = cv2.imread(str(image_path))
    # if image is None:
    #     print(f'Error: {image_path} not found or unable to read')
    # # alpha < 1 降低对比度，beta < 0 降低亮度
    # dark_image = cv2.convertScaleAbs(image, alpha=0.6, beta=-40)
    # # gamma校正
    # #dark_gamma_image = cv2.LUT(dark_image, table)
    # # 保存调整后的图像
    # cv2.imwrite(str(dest_folder_altered / image_path.name), dark_image)

    # 打开图片
    image = Image.open(str(image_path))

    # 调整亮度
    enhancer = ImageEnhance.Brightness(image)
    final_brightness_factor = random.uniform(brightness_factor-brightness_change_factor, brightness_factor+brightness_change_factor)
    image = enhancer.enhance(final_brightness_factor)

    # 调整对比度
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    # 保存图片
    image.save(str(dest_folder_altered / image_path.name))
    print("调整并复制",image_path)

print("图像处理完成。未调整的图像已复制到：", dest_folder_unaltered)
print("调整过的图像已复制到：", dest_folder_altered)
