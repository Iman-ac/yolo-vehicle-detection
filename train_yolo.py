from ultralytics import YOLO
import wandb
import torch
import os
import pandas as pd  

"""WandB init (for all runs)""" 
wandb.init(project="yolo-vehicle-detection", name="yolov8-all-models")

""" Models (n=fast/small, s=balanced, m=accurate)"""
models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']

results_summary = {}  # for compsrison

for model_name in models:
    print(f"\n Starting training {model_name}...")
    model = YOLO(model_name)  
    
    """Fine-tune with optimal settings for small dataset"""
    result = model.train(
        data='data.yaml', 
        epochs=50,
        imgsz=640,  # standard size
        batch=8,  
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=4,  
        project='runs/train',
        name=model_name.replace('.pt', '_vehicles'),  
        plots=True,  
        save=True,  
        val=True,  
        patience=10  # early stop if no improvement
    )
    
    """Extract val metrics"""
    final_val_map = result.results_dict['metrics/mAP50(B)']
    results_summary[model_name] = {
        'mAP@0.5 (Val)': final_val_map,
        'Precision (Val)': result.results_dict['metrics/precision(B)'],
        'Recall (Val)': result.results_dict['metrics/recall(B)']
    }
    
    """Test on test set"""
    save_dir = result.save_dir  
    best_pt_path = os.path.join(save_dir, 'weights/best.pt')
    if os.path.exists(best_pt_path):
        test_model = YOLO(best_pt_path)
        test_results = test_model.val(data='data.yaml', split='test')
        results_summary[model_name]['mAP@0.5 (Test)'] = test_results.box.map50
        print(f" {model_name}: Val mAP@0.5 = {final_val_map:.3f}, Test mAP@0.5 = {test_results.box.map50:.3f}")
    else:
        print(f" {model_name}: best.pt not found, skipping test.")

"""Summary table (for PDF)"""
df = pd.DataFrame(results_summary).T
print("\n Model comparison:")
print(df)
df.to_csv('model_comparison.csv')  

wandb.finish()