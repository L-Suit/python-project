from mmdet.apis import init_detector, inference_detector,DetInferencer
import argparse
# config_file = '../data/rtmdet_tiny_8xb32-300e_coco.py'
# checkpoint_file = '../data/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
# model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
# inference_detector(model, '../data/img/deng.png')


# # 初始化模型
# inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco')
#
# # 推理示例图片
# inferencer('../data/img/deng.png', show=True)



# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="这是一个示例程序")

# 添加一个可选参数
parser.add_argument("-n", "--name", type=str, help="你的名字")

# 添加一个位置参数
parser.add_argument("age", type=int, help="你的年龄")

# 解析命令行参数
args = parser.parse_args()

# 使用参数
print(f"你好，{args.name}！")
print(f"你的年龄是：{args.age}")