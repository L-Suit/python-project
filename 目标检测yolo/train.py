from ultralytics8 import YOLO


from ultralytics8.models.yolo.detect import DetectionTrainer
from ultralytics8.utils import DEFAULT_CFG


# if __name__ == '__main__':
#     args = dict(model='yolov8n.pt', data='mydataset-for31.yaml', epochs=20, batch=16, workers=2)
#     # data='mydataset-IP102.yaml'  'mydataset-for31.yaml'
#
#     # 设置模型保存路径
#     DEFAULT_CFG.save_dir = f"runs/detect/yolov8n_31pests"
#
#     trainer = DetectionTrainer(overrides=args)
#     trainer.train()


if __name__ == '__main__':
    model = YOLO(r'./cfg/models/v8/yolov8n.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    imgsz = 544
    epoch = 200
    batch = 16
    optimizer = 'AdamW'
    lr0 = 0.001
    patience = 15
    weight_decay = 0.0005
    workers = 6



    model.train(data=r'mydataset-for31.yaml',
                # 如果大家任务是其它的'ultralytics11/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                imgsz=imgsz,
                epochs=epoch,
                lr0=lr0,
                batch=batch,
                optimizer=optimizer,  # 优化器设置
                workers= workers,
                patience=patience,
                #dropout=dropout,
                weight_decay=weight_decay,

                mosaic=0,
                pretrained=False,
                single_cls=False,  # 是否是单类别检测
                close_mosaic=10,
                device='0',
                cache=True,
                resume=True, # 如过想续训,此处设置true，model不用.yaml改为last.pt的位置
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                # half=True,
                project='runs/detect',
                name=f'yolov8n_for31weather_epo{epoch}_lr{lr0}_{batch}_{optimizer}_wk{workers}_wd{weight_decay}_sz{imgsz}_',
                )