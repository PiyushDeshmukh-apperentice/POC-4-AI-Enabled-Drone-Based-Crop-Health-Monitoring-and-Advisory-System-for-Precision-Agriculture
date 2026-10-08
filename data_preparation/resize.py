import os
from PIL import Image

def resize_images_in_folder(folder_path, target_width, target_height):
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
    
    # Check if the directory actually exists
    if not os.path.exists(folder_path):
        print(f"Error: The path '{folder_path}' does not exist.")
        return

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with Image.open(file_path) as img:
                    # Resizes the image using a high-quality resampling filter
                    resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    
                    # Overwrites the original image. 
                    # If you want to keep originals, save to a different folder or rename it.
                    resized_img.save(file_path)
                    print(f"Resized: {filename} to {target_width}x{target_height}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    # --- HARDCODED CONFIGURATION ---
    # Replace this with the actual path to your folder
    # Note: On Windows, use forward slashes (/) or a raw string r"C:\path\to\folder"
    IMAGE_FOLDER = r"D:\Projects\Plant Leaf Disease\WeedyRice_Uneven\unhealthy\G" 
    
    DESIRED_WIDTH = 224
    DESIRED_HEIGHT = 224
    # -------------------------------

    resize_images_in_folder(IMAGE_FOLDER, DESIRED_WIDTH, DESIRED_HEIGHT)
    print("Processing complete!")