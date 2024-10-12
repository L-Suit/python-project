# 实际使用的脚本
import os

original_file = r"D:\dataset\ip102\Detection\VOC2007\test.txt"  # 原始文件名
new_file = r"D:\dataset\ip102_weather\test.txt"      # 修改后的文件名

# 原来路径
original_prefix = 'D:/dataset/ip102/Detection/VOC2007/images/'
# 要替换的新路径前缀
new_prefix = 'D:/dataset/ip102_weather/images/'

with open(original_file, 'r') as file:
    # 读取所有行
    lines = file.readlines()

# 处理每一行
modified_lines = [
    line.replace(original_prefix, new_prefix) for line in lines
]

# 如果没有文件就新建文件


# 写入新文件
with open(new_file, 'w') as file:
    file.writelines(modified_lines)

# 如果希望覆盖原文件，可以使用如下代码
# with open(original_file, 'w') as file:
#     file.writelines(modified_lines)