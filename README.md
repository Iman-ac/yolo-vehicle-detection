 YOLOv8_Vehicle_Detection_Project

 Explain
-  دیتاست عکس 100 وسیله نقلیه (5 کلاس: Car, Bicycle, Bus, Truck, Motorcycle)
- مدل‌ها: Fine-tune YOLOv8n/s/m, mAP@0.5 Val=0.256 (بهترین).
- مقایسه:  model_comparison.csv.
- WandB: https://wandb.ai/[Iman-ac]/yolo-vehicle-detection
- PDF: yolo_report.pdf (روال, جدول, تصاویر)

Run
pip install -r requirements.txt
python train_yolo.py  

 Requirements
ultralytics wandb pandas reportlab