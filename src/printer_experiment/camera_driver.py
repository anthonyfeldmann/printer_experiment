import cv2
import numpy as np
import os
import time

def get_single_measurement(image_path: str, target_bucket: int = 1) -> float:
    """
    Reads a saved image, crops it using explicit hardcoded pixel coordinates,
    and measures the distance from the bottom of the bucket to the highest water pixel.
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
        # Note: You may need to adjust the '60' based on your recent lighting tuning!
        _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV) 
        
        thresh_path = f"{base_name}_{timestamp}_bucket_{target_bucket}_thresh{ext}"
        cv2.imwrite(thresh_path, thresh)
        print(f"[Driver] Saved threshold view to: {thresh_path}")

        # --- THE NEW MEASUREMENT LOGIC ---
        # Find the Y and X coordinates of EVERY white pixel in the cropped image
        y_coords, x_coords = np.where(thresh == 255)

        height, width = thresh.shape

        if len(y_coords) == 0:
            print("Warning: No white pixels detected. The fluid missed the crop box entirely.")
            return 50.0

        # NOISE REJECTION: Take the 5th percentile of the top pixels to find the solid water line.
        # This ignores a random single speck of dust or splash.
        top_y = int(np.percentile(y_coords, 5))

        # The target is the absolute bottom of the crop box (which is just 'height' in local coordinates)
        target_y = height

        # Calculate straight vertical pixel distance
        pixel_distance = target_y - top_y

        # Convert to millimeters
        mm_per_pixel = 0.264
        error_distance_mm = pixel_distance * mm_per_pixel

        # --- DRAW VISUAL OVERLAYS ---
        # Draw a blue line at the bottom (target) and a green line at the top of the water
        cv2.line(image, (0, target_y), (width, target_y), (255, 0, 0), 2) 
        cv2.line(image, (0, top_y), (width, top_y), (0, 255, 0), 2)
        
        text = f"{error_distance_mm:.2f} mm"
        # Shift text down slightly if the water is right at the top of the frame
        text_y = top_y - 5 if top_y > 15 else top_y + 15
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
    test_image_path = "images/run_1719900000_iter_0.jpg" # Adjust to your real file
    print(f"--- Running Independent Test on {test_image_path} ---")

    result = get_single_measurement(test_image_path)

    if result is not None:
        if result == 50.0:
            print("\nTest failed: Fluid missed target. Penalty applied.")
        else:
            print(f"\nSuccess! Calculated Error Distance: {result:.3f} mm")
    else:
        print("\nTest encountered a fatal error.")
