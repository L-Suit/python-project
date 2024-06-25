from ultralytics.models.yolo.detect import DetectionTrainer

args = dict(model='yolov8n.pt', data='mydataset-IP102.yaml', epochs=5)
trainer = DetectionTrainer(overrides=args)
trainer.train()