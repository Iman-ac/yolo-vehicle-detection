from ultralytics import YOLO


model = YOLO('runs/train/yolov8m_vehicles/weights/best.pt')


results = model('data/test/images/image46.JPG')  # example
results[0].show()  
results[0].save('output_inference.jpg')  

print("sample detections:", results[0].boxes.cls.tolist())  