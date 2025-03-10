import json
import pandas as pd

# 输入和输出文件路径
input_file = r"D:/实验室/小论文/实验数据/dynamic-rcnn_r50/20250228_100103/vis_data/20250228_100103.json"  # 替换为你的 JSON 文件路径
output_file = "results.csv"

# 初始化变量
results = []
current_epoch = 1
epoch_data = []
coco_map = None
coco_map_50 = None

# 逐行读取 JSON 文件
with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # 解析 JSON 数据
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        # 检查是否是每轮的 COCO mAP 结果
        if "coco/bbox_mAP" in data:
            coco_map = data["coco/bbox_mAP"]
            coco_map_50 = data["coco/bbox_mAP_50"]
        else:
            # 如果是普通日志条目，检查是否属于当前轮次
            if data["epoch"] == current_epoch:
                epoch_data.append(data)
            else:
                # 新轮次，提取上一轮的数据
                if len(epoch_data) >= 7:
                    loss_cls = epoch_data[6]["loss_cls"]
                    loss_bbox = epoch_data[6]["loss_bbox"]
                    results.append([current_epoch, coco_map, coco_map_50, loss_cls, loss_bbox])

                # 重置变量
                current_epoch = data["epoch"]
                epoch_data = [data]
                coco_map = None
                coco_map_50 = None

# 处理最后一轮的数据
if len(epoch_data) >= 7:
    loss_cls = epoch_data[6]["loss_cls"]
    loss_bbox = epoch_data[6]["loss_bbox"]
    results.append([current_epoch, coco_map, coco_map_50, loss_cls, loss_bbox])

# 将数据保存到 Excel 文件
df = pd.DataFrame(results, columns=["Epoch", "metrics/mAP50-95(B)", "metrics/mAP50(B)", "val/cls_loss", "val/box_loss"])
df.to_csv(output_file, index=False)

print(f"数据已成功提取并保存到 {output_file}")