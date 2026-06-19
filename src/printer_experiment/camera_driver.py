import cv2
import numpy as np

def get_single_measurement(image_path: str) -> float:
    """
    Reads a saved image from the MADSci workflow, processes it to find the
    distance between the target (printed ridge) and the water drop.
    """
    # 1. Load the image from the file path
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: OpenCV could not load the image at {image_path}")
        return None

    try:
        # 2. Convert to grayscale and apply a blur to reduce noise
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Threshold the image to isolate the dark objects (ridge and drop)
        # You may need to tune the '60' value depending on your lab's lighting
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

        # 4. Find the contours (shapes) in the image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 2:
            print("Warning: Could not detect both the ridge and the drop clearly.")
            # If the drop completely missed the frame or blended in, return a high error penalty
            return 50.0

        # Sort the contours by area to identify which is which
        # Assuming the printed ridge is the largest shape, and the water drop is the second largest
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
        # IMPORTANT: You will need to tune this calibration factor based on how high your camera is mounted!
        mm_per_pixel = 0.264 
        error_distance_mm = pixel_distance * mm_per_pixel

        # --- OPTIONAL DEBUGGING ---
        # Uncomment the three lines below if you want the script to save a visual copy 
        # showing exactly where it drew the measurement line, which is great for tuning!
        
        # cv2.line(image, (cX_ridge, cY_ridge), (cX_drop, cY_drop), (0, 255, 0), 2)
        # debug_path = image_path.replace(".jpg", "_DEBUG.jpg")
        # cv2.imwrite(debug_path, image)

        return float(error_distance_mm)

    except Exception as e:
        print(f"Error during OpenCV processing: {e}")
        return None