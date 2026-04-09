import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import cv2

# VOC classes
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def convert_voc_to_yolo(xml_file, output_dir, image_width, image_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    yolo_lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in classes:
            continue
        class_id = classes.index(class_name) + 1  # Add 1 for background class at index 0

        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # Convert to YOLO format
        x_center = (xmin + xmax) / 2 / image_width
        y_center = (ymin + ymax) / 2 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Write to txt file
    txt_file = output_dir / (Path(xml_file).stem + '.txt')
    with open(txt_file, 'w') as f:
        f.write('\n'.join(yolo_lines))

def main():
    voc_dir = Path('VOC2012')  # Adjusted to current directory
    output_labels_dir = Path('datasets/voc2012/labels')
    output_images_dir = Path('datasets/voc2012/images')

    # Create dirs
    for split in ['train', 'val']:
        (output_labels_dir / split).mkdir(parents=True, exist_ok=True)
        (output_images_dir / split).mkdir(parents=True, exist_ok=True)

    # Assume standard VOC structure
    annotations_dir = voc_dir / 'Annotations'
    images_dir = voc_dir / 'JPEGImages'

    # VOC2012 uses class-specific split files, use aeroplane as reference
    trainval_file = voc_dir / 'ImageSets' / 'Main' / 'aeroplane_trainval.txt'
    val_file = voc_dir / 'ImageSets' / 'Main' / 'aeroplane_val.txt'

    if not trainval_file.exists():
        print("VOC dataset not found. Looking for aeroplane_trainval.txt in ImageSets/Main/")
        return

    trainval_ids = []
    with open(trainval_file, 'r') as f:
        for line in f.readlines():
            img_id = line.split()[0]
            trainval_ids.append(img_id)

    val_ids = []
    with open(val_file, 'r') as f:
        for line in f.readlines():
            img_id = line.split()[0]
            val_ids.append(img_id)

    train_ids = [id for id in trainval_ids if id not in val_ids]

    print(f"Total trainval: {len(trainval_ids)}, Train: {len(train_ids)}, Val: {len(val_ids)}")

    # Process train
    train_count = 0
    for img_id in train_ids:
        xml_file = annotations_dir / f'{img_id}.xml'
        img_file = images_dir / f'{img_id}.jpg'
        if xml_file.exists() and img_file.exists():
            img = cv2.imread(str(img_file))
            if img is not None:
                h, w = img.shape[:2]
                convert_voc_to_yolo(xml_file, output_labels_dir / 'train', w, h)
                # Copy image using shutil
                dest = output_images_dir / 'train' / f'{img_id}.jpg'
                shutil.copy2(str(img_file), str(dest))
                train_count += 1

    # Process val
    val_count = 0
    for img_id in val_ids:
        xml_file = annotations_dir / f'{img_id}.xml'
        img_file = images_dir / f'{img_id}.jpg'
        if xml_file.exists() and img_file.exists():
            img = cv2.imread(str(img_file))
            if img is not None:
                h, w = img.shape[:2]
                convert_voc_to_yolo(xml_file, output_labels_dir / 'val', w, h)
                dest = output_images_dir / 'val' / f'{img_id}.jpg'
                shutil.copy2(str(img_file), str(dest))
                val_count += 1

    print(f"Conversion complete!")
    print(f"Train: {train_count} images, Val: {val_count} images")

if __name__ == '__main__':
    main()