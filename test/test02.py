import cv2

image_path = 'D:\dataset/forest-31-pests/train2017/1.jpg'


image = cv2.imread(image_path)
if image is None:
    print('Error: Image not found or unable to read')

cv2.imshow('image', image)
# 等比例缩放图片到480*480像素
resized_image = cv2.resize(image, (480, 480))
cv2.imshow('resized_image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()