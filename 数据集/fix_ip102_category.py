import csv

file_path = 'D:\dataset\ip102\Classification/classes.txt'
data = []

with open(file_path, 'r') as file:
    reader = csv.reader(file, delimiter='\t')  # 假设数据是由制表符分隔的
    for row in reader:
        if len(row) == 2:
            data.append((row[0], row[1]))  # 将每行的数据作为一个元组添加到列表中

# 打印读取的数据
for item in data:
    print(item)