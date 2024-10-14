from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"D:\Pycharm-project\pythonProject\目标检测yolo\runs\detect\yolov8n_weather_test5\weights\last.pt")
    for param in model.parameters():
        print(param.dtype)  # 这将打印出每个参数的数据类型
        break
    #model = YOLO(r"D:\Pycharm-project\pythonProject\目标检测yolo\runs\detect\train\weights\last.pt")
    model.val(data="mydataset-IP102.yaml")