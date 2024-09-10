import json
import shutil
import os


json_file_path = 'D:\dataset\林业害虫30类/annotations/instances_val2017.json'
#原数据集图片路径
original_path = 'D:\dataset\林业害虫30类/val2017/'
# 新路径
new_path = 'D:/dataset/自组建新数据集/val2017/'
newjson_path = 'D:/dataset/自组建新数据集/annotations/instances_val2017.json'

# 读取JSON文件
with open(json_file_path, 'r') as file:
    data = json.load(file)


# 定义要移动的类别
categoriesID_to_move = [0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 9, 12, 14, 16, 17, 18, 19, 20, 21, 22,
                        23, 26, 27, 28, 29, 30]


# 确保新路径存在
if not os.path.exists(new_path):
    os.makedirs(new_path)

all_categories = data['categories']
images = data['images']



new_categories = []
new_annatations = []
new_json = {}
# 原images不用动

# 建立ID到文件名的映射
imgID_to_name = {}
for img in images:
    imgID_to_name[img['id']] = img['file_name']

# 建立旧类ID到新类ID的映射
oricateID_to_newID = {}

# 新类ID
print("待移动的类别:")
i = 0
for category in all_categories:
    if category['id'] in categoriesID_to_move:
        oricateID_to_newID[category['id']] = i
        category['id'] = i
        i += 1
        new_categories.append(category)
        print(category['name'])
print("新类别:", new_categories)

# 遍历数据，移动图片
j = 0
for item in data['annotations']:
    # print(item['image_id'])
    if item['category_id'] in categoriesID_to_move:
        # 构建原始图片路径
        ori_img_path =  original_path + imgID_to_name[item['image_id']]
        # 构建新图片路径
        new_image_path = os.path.join(new_path, os.path.basename(ori_img_path))
        # 移动图片
        shutil.copy(ori_img_path, new_image_path)
        print(f'Copy {ori_img_path} to {new_image_path}')
        # 更新图片标注的cateID和id
        item['category_id'] = oricateID_to_newID[item['category_id']]
        item['id'] = j
        j += 1
        new_annatations.append(item)

new_json['categories'] = new_categories
new_json['annotations'] = new_annatations
new_json['images'] = data['images']
# new_json.append({'categories': new_categories, 'annotations': new_annatations, 'images': data['images']})

# 写入新的JSON文件
with open(newjson_path, 'w') as file:
    json.dump(new_json, file, indent=4)


print('Images have been moved successfully!!')