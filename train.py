from ultralytics import YOLO
import os

print('='*50)
print('YOLO Training Started')
print('='*50)

train_imgs = len(os.listdir('datasets/mydata/images/train'))
val_imgs = len(os.listdir('datasets/mydata/images/val'))
print(f'Train images: {train_imgs}')
print(f'Val images: {val_imgs}')
print()

model = YOLO('yolov8n.pt')
print('Loading model: yolov8n.pt')
print()

print('Starting training...')
results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=416,
    device='cpu',
    patience=10,
    save=True,
    verbose=False
)

print()
print('='*50)
print('Training Completed!')
print('='*50)
best_weights = 'runs/detect/train/weights/best.pt'
if os.path.exists(best_weights):
    print(f'Best model saved to: {best_weights}')
else:
    print('Model saved in runs/detect/ directory')
