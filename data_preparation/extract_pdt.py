import os
import xml.etree.ElementTree as ET
import random
import math
import cv2

# --- CONFIGURATION ---
IMAGE_DIR = r"D:\Projects\Plant Leaf Disease\PDT dataset\PDT dataset\LH\images"        
ANNOTATION_DIR = r"PDT dataset/PDT dataset/LH/Annotations"  
OUTPUT_DIR = r"PDT dataset/PDT_CNN"

# Target Size for the classification dataset
TARGET_SIZE = (224, 224)
MIN_CROP_SIZE = 256  # Prevents pixelation by enforcing a context window around tiny objects

# Create target directories
HEALTHY_DIR = os.path.join(OUTPUT_DIR, "healthy")
UNHEALTHY_DIR = os.path.join(OUTPUT_DIR, "stressed")
os.makedirs(HEALTHY_DIR, exist_ok=True)
# os.makedirs(STRESSED_DIR, exist_ok=True) # Note: matching folder name setup
os.makedirs(UNHEALTHY_DIR, exist_ok=True)

def make_square_box(xmin, ymin, xmax, ymax, img_w, img_h, min_crop_size=256):
    """Adjusts a bounding box to be a perfect square with a minimum size to avoid pixelation."""
    width = xmax - xmin
    height = ymax - ymin
    
    size = max(width, height, min_crop_size)
    cx, cy = xmin + width // 2, ymin + height // 2
    
    new_xmin = max(0, cx - size // 2)
    new_ymin = max(0, cy - size // 2)
    new_xmax = min(img_w, new_xmin + size)
    new_ymax = min(img_h, new_ymin + size)
    
    if (new_xmax - new_xmin) < size:
        new_xmin = max(0, new_xmax - size)
    if (new_ymax - new_ymin) < size:
        new_ymin = max(0, new_ymax - size)
        
    return new_xmin, new_ymin, new_xmax, new_ymax

def is_overlapping(box1, box2):
    """Checks if two bounding boxes [xmin, ymin, xmax, ymax] overlap."""
    return not (box1[2] <= box2[0] or box1[0] >= box2[2] or box1[3] <= box2[1] or box1[1] >= box2[3])

def generate_healthy_patches(img_w, img_h, unhealthy_boxes, patch_size=256, num_patches=3):
    """Generates random square patches that do not overlap with unhealthy bounding boxes."""
    healthy_boxes = []
    max_attempts = 300  # High attempts to ensure clean background discovery
    attempts = 0
    
    while len(healthy_boxes) < num_patches and attempts < max_attempts:
        attempts += 1
        xmin = random.randint(0, img_w - patch_size)
        ymin = random.randint(0, img_h - patch_size)
        xmax = xmin + patch_size
        ymax = ymin + patch_size
        candidate = [xmin, ymin, xmax, ymax]
        
        overlap = any(is_overlapping(candidate, ub) for ub in unhealthy_boxes) or \
                  any(is_overlapping(candidate, hb) for hb in healthy_boxes)
                  
        if not overlap:
            healthy_boxes.append(candidate)
            
    return healthy_boxes

def process_dataset_automated():
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No valid images found in {IMAGE_DIR}")
        return

    print(f"Starting completely automated run for {len(image_files)} images...\n---")
    saved_unhealthy_count = 0
    saved_healthy_count = 0

    for img_filename in image_files:
        base_name = os.path.splitext(img_filename)[0]
        image_path = os.path.join(IMAGE_DIR, img_filename)
        annotation_path = os.path.join(ANNOTATION_DIR, f"{base_name}.xml")
        
        if not os.path.exists(annotation_path):
            continue
            
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading: {img_filename}")
            continue
            
        if len(img.shape) < 3 or img.shape[2] != 3:
            continue
            
        img_h, img_w, _ = img.shape
        
        # 1. Parse Pascal VOC XML Annotation File
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        all_unhealthy_boxes = []
        for obj in root.findall('object'):
            name = obj.find('name').text.strip().lower()
            if name == 'unhealthy':
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                all_unhealthy_boxes.append([xmin, ymin, xmax, ymax])

        total_found = len(all_unhealthy_boxes)
        if total_found == 0:
            continue

        # 2. Apply Dynamic Capping (Take 25% if total objects > 5)
        if total_found > 5:
            sample_size = max(5, math.ceil(total_found * 0.25))
            selected_unhealthy_boxes = random.sample(all_unhealthy_boxes, sample_size)
        else:
            selected_unhealthy_boxes = all_unhealthy_boxes

        # 3. Generate EXACTLY EQUAL Number of Healthy Patches
        healthy_boxes = generate_healthy_patches(img_w, img_h, all_unhealthy_boxes, patch_size=MIN_CROP_SIZE, num_patches=len(selected_unhealthy_boxes))
        
        # Force strict 1:1 fallback balance if background real-estate is too crowded
        if len(healthy_boxes) < len(selected_unhealthy_boxes):
            selected_unhealthy_boxes = selected_unhealthy_boxes[:len(healthy_boxes)]

        # 4. Extract and Save Unhealthy Patches (No Popups)
        for idx, box in enumerate(selected_unhealthy_boxes):
            sq_xmin, sq_ymin, sq_xmax, sq_ymax = make_square_box(box[0], box[1], box[2], box[3], img_w, img_h, min_crop_size=MIN_CROP_SIZE)
            crop = img[sq_ymin:sq_ymax, sq_xmin:sq_xmax]
            resized_crop = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            output_name = f"{base_name}_unhealthy_box{idx+1}.jpg"
            cv2.imwrite(os.path.join(UNHEALTHY_DIR, output_name), resized_crop)
            saved_unhealthy_count += 1

        # 5. Extract and Save Healthy Patches (No Popups)
        for idx, box in enumerate(healthy_boxes):
            crop = img[box[1]:box[3], box[0]:box[2]]
            resized_crop = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            output_name = f"{base_name}_healthy_patch{idx+1}.jpg"
            cv2.imwrite(os.path.join(HEALTHY_DIR, output_name), resized_crop)
            saved_healthy_count += 1

    print("---")
    print("Automation Process Completed Successfully!")
    print(f"Total Unhealthy Class Images Saved: {saved_unhealthy_count}")
    print(f"Total Healthy Class Images Saved: {saved_healthy_count}")

if __name__ == "__main__":
    process_dataset_automated()