from PIL import Image


# 示例用法
input_image_path = r'D:\Pycharm_project\python-project\data\img\2(27).jpg'
out_image_path = r'D:\Pycharm_project\python-project\data\img\re1.jpg'
# 打开图像
pil_image = Image.open(input_image_path)

# 定义最大尺寸
max_size = (900, 900)

# 使用thumbnail保持宽高比
pil_image.thumbnail(max_size)


pil_image.save(out_image_path)