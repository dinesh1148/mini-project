from ultralytics import YOLO
import os

print('='*60)
print('YOLO Training on VOC2012 Dataset')
print('='*60)

train_imgs = len(os.listdir('datasets/voc2012/images/train'))
val_imgs = len(os.listdir('datasets/voc2012/images/val'))
print(f'Train images: {train_imgs}')
print(f'Val images: {val_imgs}')
print()

model = YOLO('yolov8n.pt')
print('Loading model: yolov8n.pt (pretrained on COCO)')
print()

print('Starting training on VOC2012...')
print('Device: CPU (adjust to GPU if available)')
print()

results = model.train(
    data='data_voc2012.yaml',
    epochs=100,
    imgsz=416,
    device='cpu',
    patience=15,
    save=True,
    verbose=True,
    project='runs/voc2012',
    name='train'
)

print()
print('='*60)
print('Training Completed!')
print('='*60)
best_weights = 'runs/voc2012/train/weights/best.pt'
if os.path.exists(best_weights):
    print(f'Best model saved to: {best_weights}')
else:
    print('Model saved in runs/voc2012/ directory')
