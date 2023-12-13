import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to rotate an image
def rotate_image(image, angle):
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))

# Function to resize an image
def resize_image(image, scale):
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

# Load the image
img_rgb = cv2.imread('./Images/messi5.jpg')

# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Load the template
template = cv2.imread('./Images/Template.jpg', 0)

# Store width and height of the template
w, h = template.shape[::-1]

# Original template matching
res_original = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_original)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img_rgb, top_left, bottom_right, (0, 255, 0), 2)

# Display the original template matching result
plt.subplot(331), plt.imshow(res_original, cmap='gray')
plt.title('Original Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(332), plt.imshow(img_rgb, cmap='gray')
plt.title('Original Detected Point'), plt.xticks([]), plt.yticks([])

# Rotation variant template matching
for angle in range(-45, 46, 15):
    rotated_template = rotate_image(template, angle)
    res_rotation = cv2.matchTemplate(img_gray, rotated_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_rotation)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img_rgb, top_left, bottom_right, (0, 255, 0), 2)

# Display the rotation variant template matching result
plt.subplot(333), plt.imshow(res_rotation, cmap='gray')
plt.title('Rotation Variant Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(334), plt.imshow(img_rgb, cmap='gray')
plt.title('Rotation Variant Detected Point'), plt.xticks([]), plt.yticks([])

# Scale variant template matching
for scale in np.linspace(0.5, 2.0, 5):
    scaled_template = resize_image(template, scale)
    res_scale = cv2.matchTemplate(img_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_scale)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img_rgb, top_left, bottom_right, (0, 255, 0), 2)

# Display the scale variant template matching result
plt.subplot(335), plt.imshow(res_scale, cmap='gray')
plt.title('Scale Variant Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(336), plt.imshow(img_rgb, cmap='gray')
plt.title('Scale Variant Detected Point'), plt.xticks([]), plt.yticks([])

# Multi-object detection
threshold = 0.8
locations = np.where(res_original >= threshold)

for pt in zip(*locations[::-1]):
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, bottom_right, (0, 255, 0), 2)

# Display the multi-object detection result
plt.subplot(337), plt.imshow(res_original, cmap='gray')
plt.title('Multi-Object Detection Result'), plt.xticks([]), plt.yticks([])
plt.subplot(338), plt.imshow(img_rgb, cmap='gray')
plt.title('Multi-Object Detected Points'), plt.xticks([]), plt.yticks([])

# Show all plots
plt.show()
