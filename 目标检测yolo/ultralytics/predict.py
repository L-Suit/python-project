from ultralytics import YOLO

pth_path = r"E:\lsh\python-project\目标检测yolo\runs\detect\yolov8n_for31_epo200_AdamW_640_0.002\weights\best.pt"

test_path = r"C:\ProgramData\lsh-dataset\test-pest"
# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO(pth_path)  # load a custom model

# Predict with the model
results = model(test_path, save=True, conf=0.5)  # predict on an image