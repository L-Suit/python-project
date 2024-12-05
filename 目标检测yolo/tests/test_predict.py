from ultralytics.utils import ASSETS
from ultralytics.models.yolo.detect import DetectionPredictor

args = dict(model='yolov8n.pt', source=ASSETS)
predictor = DetectionPredictor(overrides=args)
predictor.predict_cli()