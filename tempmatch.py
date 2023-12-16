import cv2 
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('./Images/soccer_practice.jpg')
img2 = cv2.imread('./Images/rotated_ball.png')  
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                     
sift = cv2.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

if descriptors_2 is not None:
    descriptors_1 = descriptors_1.astype(np.float32)
    descriptors_2 = descriptors_2.astype(np.float32)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
    plt.imshow(img3), plt.show()
else:
    print("Feature detection and computation failed for the second image.")
