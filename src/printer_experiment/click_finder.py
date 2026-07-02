import cv2

# --- CHANGE THIS TO YOUR IMAGE NAME ---
image_path = "images/plate_image_iter_0_1783002252_original.jpg"

def click_event(event, x, y, flags, param):
    # If you click the Left Mouse Button
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked Coordinates -> X: {x}, Y: {y}")
        # Draw a tiny red dot where you clicked so you don't lose your place
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("Coordinate Finder", img)

# Load the image
img = cv2.imread(image_path)

if img is None:
    print(f"Could not load {image_path}. Check the file name!")
else:
    print("Image loaded! Click anywhere to print coordinates.")
    print("Press the 'q' key on your keyboard to close the window.")
    
    cv2.imshow("Coordinate Finder", img)
    
    # Listen for mouse clicks and send them to our function
    cv2.setMouseCallback("Coordinate Finder", click_event)

    # Keep the window open until you press 'q'
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
