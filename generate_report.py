from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import pandas as pd
import os

"""PDF setup"""
pdf_file = "yolo_report.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=letter)
styles = getSampleStyleSheet()
story = []  # elements to add


title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=30)
story.append(Paragraph("project report: Fine-tuning YOLOv8 for detecting vihecles", title_style))
story.append(Spacer(1, 12))

# text
roal_text = """
روال پروژه:
1.  100 pictures with Label Studio labeling (5 classes: Car, Bicycle, Bus, Truck, Motorcycle).
2. split: 70% train (70 image), 20% val (20 image), 10% test (10 image).
3. Fine-tune YOLOv8n/s/m with 50 epochs, batch=8, imgsz=640 (CPU).
4. results: mAP@0.5 Val best 0.256 (m), Test 0.212. Car best class  (mAP=0.422).
5. WandB: https://wandb.ai/[tajmiri-iman-engineer]/yolo-vehicle-detection (charts mAP/loss).
"""
story.append(Paragraph(roal_text, styles['Normal']))
story.append(Spacer(1, 12))

"""comparation chart from csv"""
df = pd.read_csv('model_comparison.csv', index_col=0)  
table_data = [df.columns.tolist()] + df.values.tolist()
table = Table(table_data)
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 12),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black)
]))
story.append(table)
story.append(Spacer(1, 12))


metrics_text = """
accuracy & error:
- Precision: mid 0.58 (low false positive).
- Recall: mid 0.24 .
- mAP@0.5: n=0.12, s=0.16, m=0.26 (Val).
- best model: yolov8m (25M params, 78 GFLOPs).
"""
story.append(Paragraph(metrics_text, styles['Normal']))
story.append(Spacer(1, 12))


images = [
    ('runs/train/yolov8m_vehicles/confusion_matrix.png', 'Confusion Matrix'),
    ('runs/train/yolov8m_vehicles/results.png', 'Training Curves'),
    ('output_inference.jpg', 'Inference Output')
]
for img_path, caption in images:
    if os.path.exists(img_path):
        story.append(Paragraph(caption, styles['Heading2']))
        story.append(Spacer(1, 6))
        img = Image(img_path, width=4*inch, height=2*inch)
        story.append(img)
        story.append(Spacer(1, 12))
    else:
        print(f"Warning: {img_path} not found!")

# Build PDF
doc.build(story)
print(f"PDF saved: {pdf_file}")