from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG


# if __name__ == '__main__':
#     args = dict(model='yolov8n.pt', data='mydataset-forest-31pest.yaml', epochs=20, batch=16, workers=2)
#     # data='mydataset-IP102.yaml'  'mydataset-forest-31pest.yaml'
#
#     # 设置模型保存路径
#     DEFAULT_CFG.save_dir = f"runs/detect/yolov8n_31pests"
#
#     trainer = DetectionTrainer(overrides=args)
#     trainer.train()


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-weather.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=r'mydataset-forest-31pest.yaml',
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache=True,
                imgsz=640,
                epochs=10,
                single_cls=False,  # 是否是单类别检测
                batch=4,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # 如过想续训就设置last.pt的地址
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                # half=True,
                project='runs/detect',
                name='yolov8n_weather_31for_test',
                )