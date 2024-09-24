from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG


if __name__ == '__main__':
    args = dict(model='yolov8n.pt', data='mydataset-forest-31pest.yaml', epochs=20, batch=16, workers=2)
    # data='mydataset-IP102.yaml'  'mydataset-forest-31pest.yaml'

    # 设置模型保存路径
    DEFAULT_CFG.save_dir = f"runs/detect/yolov8n_31pests"

    trainer = DetectionTrainer(overrides=args)
    trainer.train()