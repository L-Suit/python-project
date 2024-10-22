import os
import shutil

def remove_extension(filename):
    """去除文件名的后缀"""
    return os.path.splitext(filename)[0]

def move_files(src_dir, target_dir, label_names):
    """移动文件"""
    for filename in os.listdir(src_dir):
        if remove_extension(filename) in label_names:
            src_file = os.path.join(src_dir, filename)
            shutil.move(src_file, target_dir)
            print(f"Moved '{src_file}' to '{target_dir}'")

def main():
    # 源目录和目标目录
    label_dir = r'C:\ProgramData\lsh-dataset\forest-31pests-weather\labels\train'
    src_dir = r'C:\ProgramData\lsh-dataset\forest-31pests-weather\images'
    target_dir = r'C:\ProgramData\lsh-dataset\forest-31pests-weather\images\train'

    # 读取源目录下所有文件的名称，并去除后缀
    labels_files = set(remove_extension(file) for file in os.listdir(label_dir))

    # 在目标目录下移动与源目录同名的文件
    move_files(src_dir, target_dir, labels_files)

if __name__ == '__main__':
    main()