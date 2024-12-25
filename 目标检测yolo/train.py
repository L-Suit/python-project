from ultralytics import YOLO


from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG



if __name__ == '__main__':

    model = YOLO(r'./cfg/models/v8/yolov8-weather-ADown.yaml')
    #model = YOLO(r'/root/python-project/目标检测yolo/runs/detect/'
    #             r'yolov8n-CPA_for31v2_epo200_lr0.01_16_SGD_wk6_wd0.0005_sz544_/weights/last.pt')
    imgsz = 544
    epoch = 200
    batch = 16
    optimizer = 'AdamW'
    lr0 = 0.001
    patience = 15
    weight_decay = 0.0005
    workers = 6


    model.train(data=r'mydataset-for31.yaml',
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                imgsz=imgsz,
                epochs=epoch,
                lr0=lr0,
                batch=batch,
                optimizer=optimizer,  # 优化器设置
                workers=workers,
                patience=patience,
                #dropout=dropout,
                weight_decay=weight_decay,

                mosaic=0,
                pretrained=False,
                single_cls=False,  # 是否是单类别检测
                close_mosaic=0,
                device='0',
                cache=True,
                resume=True, # 如过想续训,此处设置true，model不用.yaml改为last.pt的位置
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                # half=True,
                project='runs/detect',
                name=f'yolov8n-ADown_for31v2_epo{epoch}_lr{lr0}_{batch}_{optimizer}_wk{workers}_wd{weight_decay}_sz{imgsz}_',
                )