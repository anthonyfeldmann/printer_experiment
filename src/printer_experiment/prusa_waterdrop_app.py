import cv2
import numpy as np
import os
import time

def get_single_measurement(image_path: str, target_bucket: int = 1) -> float:
    """
    Reads a saved image, crops it using explicit tight pixel coordinates to avoid bucket walls,
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

        # --- EXPLICIT DICTIONARY CROPPING (TIGHT CENTER COLUMN) ---
        # Format: (Y_start, Y_end, X_start, X_end)
        # 3 pixels shaved off the left, right, and bottom to avoid plastic glare.
        # Top edge remains at 216 to maintain the optimizer's zero-target.
        crop_regions = {
            1: (216, 238, 383, 394),  
            2: (216, 239, 408, 418),  
            3: (216, 238, 431, 440)   
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
        clean_path = f"{base_name}_{timestamp}_bucket_{target_bucket}_clean{ext}"
        cv2.imwrite(clean_path, clean_thresh)
        print(f"[Driver] Saved clean (de-stringed) view to: {clean_path}")

        # --- THE GAP MEASUREMENT LOGIC ---
        # IMPORTANT: We are now asking for the white pixels from 'clean_thresh'
        y_coords, x_coords = np.where(clean_thresh == 255)

        height, width = clean_thresh.shape

        if len(y_coords) == 0:
            print("Warning: No white pixels detected. The fluid missed the target.")
            # 50.0 penalty applied so the optimizer knows this ridge length failed
            return 50.0

        # Find the highest pixel of the water (the meniscus) using 5th percentile for noise rejection
        top_y = int(np.percentile(y_coords, 5))

        # The target is the absolute TOP of the crop box (Y = 0)
        target_y = 0

        # Calculate the "Empty Gap" (Distance from the top of the bucket down to the water)
        pixel_distance = top_y - target_y 

        # Convert to millimeters
        mm_per_pixel = 0.264
        error_distance_mm = pixel_distance * mm_per_pixel

        # --- DRAW VISUAL OVERLAYS ---
        # Draw a blue line at the top (target) and a green line at the top of the water
        cv2.line(image, (0, target_y), (width, target_y), (255, 0, 0), 2) 
        cv2.line(image, (0, top_y), (width, top_y), (0, 255, 0), 2)
        
        text = f"Gap: {error_distance_mm:.2f} mm"
        text_y = top_y + 15 if top_y < 15 else top_y - 5
        cv2.putText(image, text, (2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        measured_path = f"{base_name}_{timestamp}_bucket_{target_bucket}_measured{ext}"
        cv2.imwrite(measured_path, image)
        print(f"[Driver] Saved measured view to: {measured_path}")

        return float(error_distance_mm)

    except Exception as e:
        print(f"Error during OpenCV processing: {e}")
        return None

# --- INDEPENDENT EXECUTION BLOCK ---
if __name__ == "__main__":
    test_image_path = "images/run_1719900000_iter_0.jpg" # Adjust to a real image filename
    print(f"--- Running Independent Test on {test_image_path} ---")

    # Forcing target_bucket=1 for local testing
    result = get_single_measurement(test_image_path, target_bucket=1)

    if result is not None:
        if result == 50.0:
            print("\nTest failed: Fluid missed target. 50.0 mm Penalty applied.")
        else:
            print(f"\nSuccess! Calculated Gap Distance: {result:.3f} mm")
    else:
        print("\nTest encountered a fatal error.")
