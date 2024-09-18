import json
import os
from PIL import Image
from tqdm import tqdm

def process_annotations(json_file, images_folder, labels_folder):
    # 读取JSON文件
    with open(json_file, 'r',encoding='utf-8') as file:
        data = json.load(file)

    annotations = data['annotations']

    # 建立image_id2name映射
    images = data['images']
    imgID_to_name = {}
    for img in images:
        imgID_to_name[img['id']] = img['file_name']

    for annotation in tqdm(annotations):
        category_id = annotation['category_id']
        image_id = annotation['image_id']

        # 获取图片尺寸
        file_path = os.path.join(images_folder, str(image_id) + '.jpg')
        try:
            with Image.open(file_path) as img:
                width, height = img.size
        except FileNotFoundError:
            continue

        # 计算归一化的坐标
        bbox = annotation['bbox']
        x_center = (bbox[0] + bbox[2] / 2) / width
        y_center = (bbox[1] + bbox[3] / 2) / height
        w_norm = bbox[2] / width
        h_norm = bbox[3] / height

        # 写入标签文件，标签编号减1
        label = category_id - 1
        label_content = f"{label} {x_center} {y_center} {w_norm} {h_norm}"

        label_file_path = os.path.join(labels_folder, str(imgID_to_name[image_id]) + '.txt')
        with open(label_file_path, 'a') as file:
            file.write(label_content + '\n')

def create_labels_folder():
    # 创建labels文件夹及其子文件夹
    labels_folder = 'D:\dataset\林业害虫31类/yolo/labels'
    train_labels_folder = os.path.join(labels_folder, 'train')
    val_labels_folder = os.path.join(labels_folder, 'val')
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)
    return train_labels_folder, val_labels_folder

def main():
    train_labels_folder, val_labels_folder = create_labels_folder()

    # 处理训练和验证数据
    process_annotations('D:\dataset\林业害虫31类/annotations/instances_train2017.json',
                        'D:\dataset\林业害虫31类/train2017', train_labels_folder)
    process_annotations('D:\dataset\林业害虫31类/annotations/instances_val2017.json',
                        'D:\dataset\林业害虫31类/val2017', val_labels_folder)

    print("完成处理。")

if __name__ == "__main__":
    main()