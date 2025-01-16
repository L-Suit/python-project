from ultralytics import YOLO
import requests
import itertools
import string
from concurrent.futures import ThreadPoolExecutor, as_completed

if __name__ == "__main__":
    #model_path = r"/root/python-project/目标检测yolo/runs/detect/yolov8n_for31weather-new_epo200_lr0.001_16_AdamW_wk6_wd0.0005_sz544_mosaic0_/weights/last.pt"
    #model_path = r"/root/python-project/目标检测yolo/runs/detect/yolov8n-CPA_for31weather-new_epo200_lr0.001_16_AdamW_wk4_wd0.0005_sz544_mosaic0_/weights/best.pt"

    #本地
    model_path = r"C:\Users\16256\Desktop\fsdownload\yolov8n-CPA_for31weather-new_epo200_lr0.001_16_AdamW_wk4_wd0.0005_sz544_mosaic0_\weights/best.pt"
    #model_path = r"C:\Users\16256\Desktop\fsdownload\yolov8n_for31weather-new_epo200_lr0.001_16_AdamW_wk6_wd0.0005_sz544_mosaic0_\weights/best.pt"

    #img_path = r"/root/dataset/pest-test/for31wea-new-img"
    img_path = r"D:\dataset\pest-test\for31wea-new-img"

    # 结果保存路径在default.yaml下修改


    # Load a model
    model = YOLO(model_path)  # load a custom model

    # Predict with the model
    results = model.predict(img_path,
                            save=True)