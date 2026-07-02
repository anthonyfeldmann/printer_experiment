import cv2
import numpy as np
import os
import time

def get_single_measurement(image_path: str, target_bucket: int = 2) -> float:
    """
    Reads a saved image, mathematically slices the plate into 3 equal sections,
    shrinks the crop box to ignore borders, and measures the drop distance.
    Saves all diagnostic images with a unique timestamp.
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

        # 1. Define the Master Bounding Box that covers ALL THREE buckets
        y_start = 215
        y_end = 240
        
        master_x_start = 375
        master_x_end = 650 
        
        # 2. Calculate the exact width of one single bucket
        total_width = master_x_end - master_x_start
        bucket_width = total_width // 3

        # 3. Find the true boundaries for the requested bucket
        shift_multiplier = target_bucket - 1
        true_x_start = master_x_start + (bucket_width * shift_multiplier)
        true_x_end = true_x_start + bucket_width

        # --- IGNORE THE BORDERS ---
        x_margin = 5
        
        # Shrink the bounding box inwards
        center_x_start = true_x_start + x_margin
        center_x_end = true_x_end - x_margin

        # Apply the mathematically centered crop
        image = image[y_start:y_end, center_x_start:center_x_end]
        
        # Add the timestamp to the filename
        cropped_path = f"{base_name}_{timestamp}_bucket_{target_bucket}_centered_cropped{ext}"
        cv2.imwrite(cropped_path, image)
        print(f"[Driver] Saved centered view of Bucket {target_bucket} to: {cropped_path}")

        # --------------------------------

        # 2. Convert to grayscale and apply a blur to reduce noise
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Threshold the image to isolate the dark objects (ridge and drop)
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Add the timestamp to the threshold diagnostic
        thresh_path = f"{base_name}_{timestamp}_thresh{ext}"
        cv2.imwrite(thresh_path, thresh)
        print(f"[Driver] Saved threshold view to: {thresh_path}")
        # --------------------------------------

        # 4. Find the contours (shapes) in the image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 2:
            print("Warning: Could not detect both the ridge and the drop clearly in the cropped frame.")
            print(f"-> Open {thresh_path} to see what OpenCV is detecting.")
            return 50.0

        # Sort the contours by area to identify which is which
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        ridge_contour = contours[0]
        drop_contour = contours[1]

        # 5. Calculate the center points (Centroids) of both shapes
        M_ridge = cv2.moments(ridge_contour)
        cX_ridge = int(M_ridge["m10"] / (M_ridge["m00"] + 1e-5))
        cY_ridge = int(M_ridge["m01"] / (M_ridge["m00"] + 1e-5))

        M_drop = cv2.moments(drop_contour)
        cX_drop = int(M_drop["m10"] / (M_drop["m00"] + 1e-5))
        cY_drop = int(M_drop["m01"] / (M_drop["m00"] + 1e-5))

        # 6. Calculate the straight-line pixel distance between the two centers
        pixel_distance = np.sqrt((cX_drop - cX_ridge)**2 + (cY_drop - cY_ridge)**2)

        # 7. Convert pixel distance to millimeters
        mm_per_pixel = 0.264 
        error_distance_mm = pixel_distance * mm_per_pixel

        # --- VISUAL ANNOTATION & SAVING ---
        # A) Save the standard overlay
        cv2.line(image, (cX_ridge, cY_ridge), (cX_drop, cY_drop), (0, 255, 0), 2)
        cv2.circle(image, (cX_ridge, cY_ridge), 2, (0, 0, 255), -1)
        cv2.circle(image, (cX_drop, cY_drop), 2, (0, 0, 255), -1)
        
        text = f"{error_distance_mm:.2f} mm"
        mid_x = (cX_ridge + cX_drop) // 2
        mid_y = (cY_ridge + cY_drop) // 2
        cv2.putText(image, text, (mid_x - 15, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Add the timestamp to the measured output
        measured_path = f"{base_name}_{timestamp}_measured{ext}"
        cv2.imwrite(measured_path, image)
        print(f"[Driver] Saved measured view to: {measured_path}")

        # B) Save the "Lines Only" diagnostic view
        black_canvas = np.zeros_like(image)
        
        cv2.drawContours(black_canvas, [ridge_contour], -1, (255, 255, 255), 1)
        cv2.drawContours(black_canvas, [drop_contour], -1, (255, 255, 255), 1)
        
        cv2.line(black_canvas, (cX_ridge, cY_ridge), (cX_drop, cY_drop), (0, 255, 0), 2)
        cv2.circle(black_canvas, (cX_ridge, cY_ridge), 2, (0, 0, 255), -1)
        cv2.circle(black_canvas, (cX_drop, cY_drop), 2, (0, 0, 255), -1)
        cv2.putText(black_canvas, text, (mid_x - 15, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Add the timestamp to the lines output
        lines_path = f"{base_name}_{timestamp}_lines{ext}"
        cv2.imwrite(lines_path, black_canvas)
        print(f"[Driver] Saved lines-only view to: {lines_path}")
        # ----------------------------------

        return float(error_distance_mm)

    except Exception as e:
        print(f"Error during OpenCV processing: {e}")
        return None

# --- INDEPENDENT EXECUTION BLOCK ---
if __name__ == "__main__":
    test_image_path = "images/plate_image_iter_0.jpg" 
    print(f"--- Running Independent Test on {test_image_path} ---")
    
    result = get_single_measurement(test_image_path)
    
    if result is not None:
        if result == 50.0:
            print("\nTest failed to find the shapes. Check the _thresh.jpg file!")
        else:
            print(f"\nSuccess! Calculated Error Distance: {result:.3f} mm")
    else:
        print("\nTest encountered a fatal error.")
