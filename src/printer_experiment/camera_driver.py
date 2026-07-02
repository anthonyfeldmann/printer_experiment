import cv2
import numpy as np
import os
import time

def get_single_measurement(image_path: str, target_bucket: int = 2) -> float:
    """
    Reads a saved image, crops it using explicit hardcoded pixel coordinates,
    and measures the drop distance.
    Saves EVERY stage of the vision pipeline for debugging.
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: OpenCV could not load the image at {image_path}")
        return None

    try:
        base_name, ext = os.path.splitext(image_path)
        if ext == '':
            ext = '.jpg'

        # Generate a unique timestamp for this specific run
        timestamp = int(time.time())

        # 1. SAVE THE ORIGINAL UNCROPPED IMAGE
        original_path = f"{base_name}_{timestamp}_original{ext}"
        cv2.imwrite(original_path, image)
        print(f"[Driver] Saved original uncropped view to: {original_path}")

        # --- THE EXPLICIT DICTIONARY CROPPING METHOD ---
        # Format is (Y_start, Y_end, X_start, X_end)
        crop_regions = {
            1: (216, 241, 380, 397),  
            2: (216, 242, 405, 421),  
            3: (216, 241, 428, 443)   
        }

        if target_bucket not in crop_regions:
            print(f"Error: Bucket {target_bucket} is not defined.")
            return None

        # Fetch the exact coordinates you hardcoded
        y_start, y_end, x_start, x_end = crop_regions[target_bucket]

        # Apply the explicit pixel crop
        image = image[y_start:y_end, x_start:x_end]

        # 2. SAVE THE CROPPED VIEW
        cropped_path = f"{base_name}_{timestamp}_bucket_{target_bucket}_cropped{ext}"
        cv2.imwrite(cropped_path, image)
        print(f"[Driver] Saved cropped view to: {cropped_path}")
        # ------------------------------------------------

        # Convert to grayscale and apply a blur to reduce noise
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. SAVE THE THRESHOLD
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV) 
        
        thresh_path = f"{base_name}_{timestamp}_bucket_{target_bucket}_thresh{ext}"
        cv2.imwrite(thresh_path, thresh)
        print(f"[Driver] Saved threshold view to: {thresh_path}")

        # Find the contours (shapes) in the image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 2:
            print("Warning: Could not detect both the ridge and the drop clearly in the cropped frame.")
            return 50.0

        # Sort the contours by area to identify which is which
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        ridge_contour = contours[0]
        drop_contour = contours[1]

        # Calculate the center points (Centroids) of both shapes
        M_ridge = cv2.moments(ridge_contour)
        cX_ridge = int(M_ridge["m10"] / (M_ridge["m00"] + 1e-5))
        cY_ridge = int(M_ridge["m01"] / (M_ridge["m00"] + 1e-5))

        M_drop = cv2.moments(drop_contour)
        cX_drop = int(M_drop["m10"] / (M_drop["m00"] + 1e-5))
        cY_drop = int(M_drop["m01"] / (M_drop["m00"] + 1e-5))

        # Calculate the straight-line pixel distance between the two centers
        pixel_distance = np.sqrt((cX_drop - cX_ridge)**2 + (cY_drop - cY_ridge)**2)

        # Convert pixel distance to millimeters
        mm_per_pixel = 0.264
        error_distance_mm = pixel_distance * mm_per_pixel

        text = f"{error_distance_mm:.2f} mm"
        mid_x = (cX_ridge + cX_drop) // 2
        mid_y = (cY_ridge + cY_drop) // 2

        # 4. SAVE THE MEASURED OVERLAY
        cv2.line(image, (cX_ridge, cY_ridge), (cX_drop, cY_drop), (0, 255, 0), 2)
        cv2.circle(image, (cX_ridge, cY_ridge), 2, (0, 0, 255), -1)
        cv2.circle(image, (cX_drop, cY_drop), 2, (0, 0, 255), -1)
        cv2.putText(image, text, (mid_x - 15, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        measured_path = f"{base_name}_{timestamp}_bucket_{target_bucket}_measured{ext}"
        cv2.imwrite(measured_path, image)
        print(f"[Driver] Saved measured view to: {measured_path}")

        # 5. SAVE THE "LINES ONLY" DIAGNOSTIC VIEW
        black_canvas = np.zeros_like(image)

        cv2.drawContours(black_canvas, [ridge_contour], -1, (255, 255, 255), 1)
        cv2.drawContours(black_canvas, [drop_contour], -1, (255, 255, 255), 1)

        cv2.line(black_canvas, (cX_ridge, cY_ridge), (cX_drop, cY_drop), (0, 255, 0), 2)
        cv2.circle(black_canvas, (cX_ridge, cY_ridge), 2, (0, 0, 255), -1)
        cv2.circle(black_canvas, (cX_drop, cY_drop), 2, (0, 0, 255), -1)
        cv2.putText(black_canvas, text, (mid_x - 15, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        lines_path = f"{base_name}_{timestamp}_bucket_{target_bucket}_lines{ext}"
        cv2.imwrite(lines_path, black_canvas)
        print(f"[Driver] Saved lines-only view to: {lines_path}")

        return float(error_distance_mm)

    except Exception as e:
        print(f"Error during OpenCV processing: {e}")
        return None

# --- INDEPENDENT EXECUTION BLOCK ---
if __name__ == "__main__":
    test_image_path = "images/run_1719900000_iter_0.jpg" # Adjust this to a real filename in your images/ folder
    print(f"--- Running Independent Test on {test_image_path} ---")

    result = get_single_measurement(test_image_path)

    if result is not None:
        if result == 50.0:
            print("\nTest failed to find the shapes. Check the diagnostic images!")
        else:
            print(f"\nSuccess! Calculated Error Distance: {result:.3f} mm")
    else:
        print("\nTest encountered a fatal error.")
