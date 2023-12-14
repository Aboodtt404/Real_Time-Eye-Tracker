import cv2
import numpy as np

def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated_image

img = cv2.imread('Images/mario.png')
template = cv2.imread('Images/mario_coin3.png')
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
h, w = template_gray.shape

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

angles_to_check = [0, 90, 180, 270]

for angle in angles_to_check:
    rotated_template = rotate_image(template_gray, angle)

    cv2.imshow(f'Rotated Template (Angle: {angle})', rotated_template)

    res = cv2.matchTemplate(img_gray, rotated_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv2.imshow('Template Matching with Rotation (cv2.TM_CCOEFF_NORMED)', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
