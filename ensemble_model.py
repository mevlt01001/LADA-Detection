from ultralytics import YOLO, RTDETR

yolo = YOLO("pt_folder/yolo12l.pt")
rtdetr = RTDETR("pt_folder/rtdetr-l.pt")
print(yolo)


