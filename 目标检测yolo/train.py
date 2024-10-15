from ultralytics.models.yolo.detect import DetectionTrainer


if __name__ == '__main__':
    args = dict(model='yolov8n.pt', data='mydataset-林业害虫31类.yaml', epochs=100, batch=32, workers=4)
    trainer = DetectionTrainer(overrides=args)
    trainer.train()