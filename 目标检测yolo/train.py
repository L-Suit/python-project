from ultralytics.models.yolo.detect import DetectionTrainer


if __name__ == '__main__':
    args = dict(model='yolov8n.pt', data='mydataset-forest-31pest.yaml', epochs=10, batch=16, workers=2)
    # data='mydataset-IP102.yaml'  'mydataset-forest-31pest.yaml'
    trainer = DetectionTrainer(overrides=args)
    trainer.train()