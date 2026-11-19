# Simple OpenCV script to generate ground truth pixel coordinates 
# Based on cv2.select ROI method

import cv2

image_path = "reference/carola68-stud-4999947.jpg"

image = cv2.imread(image_path)

height, width = image.shape[:2]

# Use OpenCV selectROI method to select the button
x, y, w, h = cv2.selectROI(
    "Select green button",
    image,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()

# Calculate geometric center of bounding box
center_x = x + w / 2
center_y = y + h / 2

print("Image dimensions:", width, height)
print("Bounding box:", x, y, w, h)
print(f"Ground truth center: ({center_x:.2f}, {center_y:.2f})")
print(f"Pixel coordinates: ({round(center_x)}, {round(center_y)})")
