YOLOv8\_Vehicle\_Detection\_Project

Explain

* دیتاست عکس 100 وسیله نقلیه (5 کلاس: Car, Bicycle, Bus, Truck, Motorcycle)
* مدل‌ها: Fine-tune YOLOv8n/s/m, mAP@0.5 Val=0.256 (بهترین).
* مقایسه:  model\_comparison.csv.
* WandB: https://wandb.ai/tajmiri-iman-engineer?shareProfileType=copy
* PDF: yolo\_report.pdf (روال, جدول, تصاویر)

Run
pip install -r requirements.txt
python train\_yolo.py

Requirements
ultralytics wandb pandas reportlab



