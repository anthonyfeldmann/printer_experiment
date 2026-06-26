import cv2
import numpy as np

def get_single_measurement(image_path: str) -> float:
    """
    Reads a saved image from the MADSci workflow, crops it, processes it to find 
    the distance, and saves a visual overlay of the OpenCV calculations.
    """
    # 1. Load the image from the file path
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: OpenCV could not load the image at {image_path}")
        return None

    try:
        # --- CROPPING LOGIC (REGION OF INTEREST) ---
        y_start = 215
        y_end = 240
        x_start = 360
        x_end = 450

        # Apply the exact pixel crop
        image = image[y_start:y_end, x_start:x_end]
        
        # Save the raw cropped frame to disk
        cropped_path = image_path.replace(".jpg", "_cropped.jpg")
        cv2.imwrite(cropped_path, image)
        # --------------------------------

        # 2. Convert to grayscale and apply a blur to reduce noise
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Threshold the image to isolate the dark objects (ridge and drop)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

        # 4. Find the contours (shapes) in the image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 2:
            print("Warning: Could not detect both the ridge and the drop clearly in the cropped frame.")
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
        # Draw a green line connecting the two centers
        cv2.line(image, (cX_ridge, cY_ridge), (cX_drop, cY_drop), (0, 255, 0), 2)
        
        # Draw red dots at the exact centroid coordinates so you can verify the tracking
        cv2.circle(image, (cX_ridge, cY_ridge), 2, (0, 0, 255), -1)
        cv2.circle(image, (cX_drop, cY_drop), 2, (0, 0, 255), -1)
        
        # Format the text and find the midpoint of the line to place it
        text = f"{error_distance_mm:.2f} mm"
        mid_x = (cX_ridge + cX_drop) // 2
        mid_y = (cY_ridge + cY_drop) // 2
        
        # Put the measurement text slightly above the line
        cv2.putText(image, text, (mid_x - 15, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Save the fully annotated image to your images folder
        measured_path = image_path.replace(".jpg", "_measured.jpg")
        cv2.imwrite(measured_path, image)
        # ----------------------------------

        return float(error_distance_mm)

    except Exception as e:
        print(f"Error during OpenCV processing: {e}")
        return None


if __name__ == "__main__":
    # Put the path to an existing raw image you want to test here:
    test_image_path = "images/plate_image_iter_0.jpg" 
    
    print(f"--- Running Independent Test on {test_image_path} ---")
    
    # Call the function
    result = get_single_measurement(test_image_path)
    
    if result is not None:
        print(f"Success! Calculated Error Distance: {result:.3f} mm")
        print("Check your folder for the _cropped.jpg and _measured.jpg files.")
    else:
        print("Test failed. Check the error messages above.")
