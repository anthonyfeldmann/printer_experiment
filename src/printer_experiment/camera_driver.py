import cv2
import numpy as np
import os
import time

def get_single_measurement(image_path: str, target_bucket: int = 1) -> float:
    """
    Reads a saved image, crops it using explicit hardcoded pixel coordinates,
    applies a two-step morphological filter to handle transparent fluid reflections
    and remove 3D printer stringing, and measures the "Empty Gap" from the top 
    of the bucket down to the highest water pixel.
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: OpenCV could not load the image at {image_path}")
        return None

    try:
        base_name, ext = os.path.splitext(image_path)
        if ext == '':
            ext = '.jpg'

        timestamp = int(time.time())

        # 1. SAVE ORIGINAL
        original_path = f"{base_name}_{timestamp}_original{ext}"
        cv2.imwrite(original_path, image)
        print(f"[Driver] Saved original uncropped view to: {original_path}")

        # --- EXPLICIT DICTIONARY CROPPING ---
        # Format: (Y_start, Y_end, X_start, X_end)
        crop_regions = {
            1: (216, 241, 380, 397),  
            2: (216, 242, 405, 421),  
            3: (216, 241, 428, 443)   
        }

        if target_bucket not in crop_regions:
            print(f"Error: Bucket {target_bucket} is not defined.")
            return None

        y_start, y_end, x_start, x_end = crop_regions[target_bucket]

        # Apply the explicit pixel crop
        image = image[y_start:y_end, x_start:x_end]
        
        cropped_path = f"{base_name}_{timestamp}_bucket_{target_bucket}_cropped{ext}"
        cv2.imwrite(cropped_path, image)

        # Convert to grayscale and blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. SAVE THRESHOLD
        # Tuned to 80 to account for ambient lighting interference
        _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV) 
        
        thresh_path = f"{base_name}_{timestamp}_bucket_{target_bucket}_thresh{ext}"
        cv2.imwrite(thresh_path, thresh)
        print(f"[Driver] Saved raw threshold view to: {thresh_path}")

        # --- THE FLUID STRINGING FIX (TWO-STEP FILTER) ---
        # Step 1: CLOSING. Fill in the holes and plump up the water reflections into a solid block.
        close_kernel = np.ones((5, 5), np.uint8)
        solid_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        # Step 2: OPENING. Use a smaller, gentle 3x3 eraser to wipe out the thin printer strings.
        open_kernel = np.ones((3, 3), np.uint8)
        clean_thresh = cv2.morphologyEx(solid_thresh, cv2.MORPH_OPEN, open_kernel)

        # Save this cleaned view so you can visually verify the water survived but the streaks are gone!
        clean_path = f"{base_
