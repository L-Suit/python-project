from ultralytics.models.yolo.detect import DetectionTrainer


if __name__ == '__main__':
    args = dict(model='yolov8n.pt', data='mydataset-IP102.yaml', epochs=5, batch=64, workers=2)
    trainer = DetectionTrainer(overrides=args)
    trainer.train()