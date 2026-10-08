import os
import json
import cv2

# --- CONFIGURATION ---
IMAGE_DIR = r"D:\Projects\Plant Leaf Disease\MSPI Dataset\train\img"        # Replace with your images folder
ANNOTATION_DIR = r"D:\Projects\Plant Leaf Disease\MSPI Dataset\train\ann"  # Replace with your JSON folder
OUTPUT_DIR = r"D:\Projects\Plant Leaf Disease\MSPI Dataset\MSPI_CNN"
TARGET_SIZE = (224, 224)

# Create target directories
HEALTHY_DIR = os.path.join(OUTPUT_DIR, "healthy")
STRESSED_DIR = os.path.join(OUTPUT_DIR, "stressed")
os.makedirs(HEALTHY_DIR, exist_ok=True)
os.makedirs(STRESSED_DIR, exist_ok=True)

def extract_square_patches(img, xmin, ymin, xmax, ymax):
    """
    Sliding window approach inside a rectangular bounding box.
    Returns a list of square cropped image arrays without adding padding.
    """
    width = xmax - xmin
    height = ymax - ymin
    patches = []

    # If it's already a square or taller than wide
    if width <= height:
        size = width
        # Slide vertically if height is significantly larger
        stride = max(1, size // 2) # 50% overlap for extracting multiple distinct patches
        current_ymin = ymin
        while current_ymin + size <= ymax:
            crop = img[current_ymin:current_ymin+size, xmin:xmin+size]
            if crop.size > 0:
                patches.append(crop)
            if current_ymin + size == ymax:
                break
            current_ymin = min(ymax - size, current_ymin + stride)
    else:
        # Tider/wider than tall: Slide horizontally
        size = height
        stride = max(1, size // 2)
        current_xmin = xmin
        while current_xmin + size <= xmax:
            crop = img[ymin:ymin+size, current_xmin:current_xmin+size]
            if crop.size > 0:
                patches.append(crop)
            if current_xmin + size == xmax:
                break
            current_xmin = min(xmax - size, current_xmin + stride)

    return patches

def process_dataset_automated():
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    skip_keywords = ["green", "channel", "red", "near", "infrared", "edge"]
    
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No valid images found in {IMAGE_DIR}")
        return

    print(f"Starting automated processing for {len(image_files)} potential images...\n---")
    total_saved_crops = 0

    for img_filename in image_files:
        filename_lower = img_filename.lower()
        
        # 1. Filename Keyword Filter
        if any(keyword in filename_lower for keyword in skip_keywords):
            continue
            
        base_name = os.path.splitext(img_filename)[0]
        image_path = os.path.join(IMAGE_DIR, img_filename)
        annotation_path = os.path.join(ANNOTATION_DIR, f"{base_name}.jpg.json")
        
        # 2. Check Annotation Matching
        if not os.path.exists(annotation_path):
            continue
            
        # 3. Load Image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading: {img_filename}")
            continue
            
        # 4. RGB Channel Check
        if len(img.shape) < 3 or img.shape[2] != 3:
            continue
            
        # 5. Load Annotation JSON
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        annotations = data if isinstance(data, list) else data.get("objects", [data])
        
        # 6. Extract and Save Crops Automatically
        for idx, ann in enumerate(annotations):
            label = ann["classTitle"]
            points = ann["points"]["exterior"]
            
            xmin, ymin = points[0]
            xmax, ymax = points[1]
            
            # Generate the natural square patches within coordinates
            square_crops = extract_square_patches(img, xmin, ymin, xmax, ymax)
            
            # Determine directory path
            save_dir = HEALTHY_DIR if label == "healthy" else STRESSED_DIR
            
            for patch_idx, crop in enumerate(square_crops):
                # Resize seamlessly to standard 224x224 shape
                resized_crop = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                
                # Format output name to prevent overwrites
                output_filename = f"{base_name}_ann{idx+1}_patch{patch_idx+1}.jpg"
                save_path = os.path.join(save_dir, output_filename)
                
                cv2.imwrite(save_path, resized_crop)
                total_saved_crops += 1

    print("---")
    print(f"Automation Complete!")
    print(f"Total processed square images (224x224) exported: {total_saved_crops}")
    print(f"Healthy folder: {len(os.listdir(HEALTHY_DIR))} images")
    print(f"Stressed folder: {len(os.listdir(STRESSED_DIR))} images")

if __name__ == "__main__":
    process_dataset_automated()