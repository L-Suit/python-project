from mmdet.apis import init_detector, inference_detector,DetInferencer

# config_file = '../data/rtmdet_tiny_8xb32-300e_coco.py'
# checkpoint_file = '../data/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
# model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
# inference_detector(model, '../data/img/deng.png')


# 初始化模型
inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco')

# 推理示例图片
inferencer('../data/img/deng.png', show=True)