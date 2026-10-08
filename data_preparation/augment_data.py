import os
import random
import glob
from PIL import Image
import numpy as np
import albumentations as A

# 1. Define transformation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.75),  # Handles 90, 180, 270 deg rotations
    A.RandomBrightnessContrast(
        brightness_limit=0.15, 
        contrast_limit=0.10, 
        p=0.5
    ),
])

def augment_class_directory(input_dir, target_count=6500):
    images = glob.glob(os.path.join(input_dir, "*.png")) + glob.glob(os.path.join(input_dir, "*.jpg"))
    current_count = len(images)
    
    if current_count >= target_count:
        print(f"Directory {input_dir} already has {current_count} images.")
        return

    needed = target_count - current_count
    print(f"Generating {needed} augmented images for {input_dir}...")

    for i in range(needed):
        # Pick a random source image from the class
        img_path = random.choice(images)
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Apply transformation
        augmented = transform(image=image)["image"]
        
        # Save augmented image
        save_path = os.path.join(input_dir, f"aug_{i}_{os.path.basename(img_path)}")
        Image.fromarray(augmented).save(save_path)

# Example usage on train directories:
augment_class_directory(r"D:\Projects\Plant Leaf Disease\Unified_Data\Healthy", target_count=7000)
augment_class_directory(r"D:\Projects\Plant Leaf Disease\Unified_Data\Unhealthy", target_count=7000)