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
    model = YOLO(r'./cfg/models/v8/yolov8.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    imgsz = 480
    epoch = 200
    batch = 16
    optimizer = 'SGD'
    lr0 = 0.01
    patience = 15



    model.train(data=r'mydataset-IP102.yaml',
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache=True,
                imgsz=imgsz,
                epochs=epoch,
                pretrained=False,
                single_cls=False,  # 是否是单类别检测
                batch=batch,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer=optimizer, # 优化器设置
                lr0=lr0,
                #resume=True, # 如过想续训,此处设置true，model不用.yaml改为last.pt的位置
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                # half=True,
                project='runs/detect',
                name=f'yolov8n_ip102weather_epo{epoch}_{optimizer}_{batch}',
                )