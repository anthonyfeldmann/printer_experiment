import cv2
import numpy as np
import os
import time

def get_single_measurement(image_path: str, target_bucket: int = 1) -> float:
    """
    Scans all three buckets simultaneously. 
    Applies Morphological image wiping to handle transparent liquid and printer strings.
    Maximizes liquid height in the target bucket minus spills.
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

        mm_per_pixel = 0.264
        bucket_liquid_heights = {}

        # Tight Center Column Crop Coordinates
        crop_regions = {
            1: (216, 238, 383, 394),  
            2: (216, 239, 408, 418),  
            3: (216, 238, 431, 440)   
        }

        print(f"[Driver] Analyzing fluid distribution across all 3 buckets...")

        for bucket_id, (y_start, y_end, x_start, x_end) in crop_regions.items():
            
            # Crop to the specific bucket
            crop = image[y_start:y_end, x_start:x_end]
            
            # Grayscale and blur
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Thresholding (Raised to 130 to catch lighter liquid)
            _, thresh = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY_INV) 

            # --- THE IMAGE WIPING FIX ---
            # Step 1: CLOSING. Fill in the holes and plump up the water reflections.
            close_kernel = np.ones((5, 5), np.uint8)
            solid_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
            
            # Step 2: OPENING. Use a 3x3 eraser to wipe out the thin 3D printer strings.
            open_kernel = np.ones((3, 3), np.uint8)
            clean_thresh = cv2.morphologyEx(solid_thresh, cv2.MORPH_OPEN, open_kernel)
            # ----------------------------

            # Find the white water pixels in the wiped image
            y_coords, x_coords = np.where(clean_thresh == 255)
            height, width = clean_thresh.shape

            if len(y_coords) == 0:
                bucket_liquid_heights[bucket_id] = 0.0
            else:
                top_y_local = int(np.percentile(y_coords, 5))
                liquid_pixels = height - top_y_local
                liquid_mm = liquid_pixels * mm_per_pixel
                bucket_liquid_heights[bucket_id] = liquid_mm

                # Visual Overlay
                global_y_top = y_start + top_y_local
                cv2.line(image, (x_start, global_y_top), (x_end, global_y_top), (0, 255, 0), 2)
                cv2.putText(image, f"{liquid_mm:.1f}mm", (x_start, global_y_top - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        # Master Score Calculation
        target_liquid = bucket_liquid_heights[target_bucket]
        spill_penalty = 0.0

        for b_id, liquid_amt in bucket_liquid_heights.items():
            if b_id != target_bucket:
                spill_penalty += liquid_amt

        total_score = target_liquid - spill_penalty

        measured_path = f"{base_name}_{timestamp}_measured_all{ext}"
        cv2.imwrite(measured_path, image)
        
        print(f"   Target Bucket ({target_bucket}) Liquid: {target_liquid:.2f} mm")
        print(f"   Spill Penalty (Wrong Buckets): -{spill_penalty:.2f} mm")
        print(f"[Driver] Total Recorded Score: {total_score:.2f} mm\n")

        return float(total_score)

    except Exception as e:
        print(f"Error during OpenCV processing: {e}")
        return None

# --- INDEPENDENT EXECUTION BLOCK ---
if __name__ == "__main__":
    test_image_path = "images/run_1719900000_iter_0.jpg" # Adjust to a real image filename
    print(f"--- Running Global Fluid Analysis Test ---")

    result = get_single_measurement(test_image_path, target_bucket=1)
