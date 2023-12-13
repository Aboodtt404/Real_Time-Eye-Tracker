# import cv2
# import dlib
# import numpy as np

# # Constants
# EYE_AR_THRESH = 0.18  # Adjusted eye aspect ratio threshold for detecting a blink
# MOVING_AVERAGE_FRAMES = 5  # Number of frames for moving average

# # Helper function to calculate the eye aspect ratio (EAR)
# def eye_aspect_ratio(eye):
#     # Ensure we have enough points for calculation
#     if len(eye) == 6:
#         # Calculate the distance between vertical eye landmarks
#         v1 = np.linalg.norm(eye[1] - eye[5])
#         v2 = np.linalg.norm(eye[2] - eye[4])

#         # Calculate the distance between the horizontal eye landmarks
#         h = np.linalg.norm(eye[0] - eye[3])

#         # Calculate the eye aspect ratio
#         ear = (v1 + v2) / (2.0 * h)

#         return ear
#     else:
#         return 0  # Return 0 if there are not enough points for EAR calculation

# # Initialize the video capture
# cam = cv2.VideoCapture(0)

# # Initialize dlib's face detector and facial landmarks predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Initialize variables
# eye_positions = []

# while True:
#     ret, frame = cam.read()

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces using dlib
#     faces = detector(gray)

#     for face in faces:
#         # Get the landmarks for the face
#         shape = predictor(gray, face)

#         # Extract coordinates of the right eye (change the indices based on your landmarks)
#         right_eye = shape.part(42).x, shape.part(43).y, shape.part(45).x, shape.part(47).y

#         # Draw a rectangle around the right eye
#         cv2.rectangle(frame, (right_eye[0], right_eye[1]), (right_eye[2], right_eye[3]), (255, 0, 0), 2)

#         # Calculate eye aspect ratio for the right eye
#         ear_right = eye_aspect_ratio(right_eye)

#         # Smooth eye movements with a moving average
#         eye_positions.append(ear_right)
#         if len(eye_positions) > MOVING_AVERAGE_FRAMES:
#             ear_right_smoothed = sum(eye_positions[-MOVING_AVERAGE_FRAMES:]) / MOVING_AVERAGE_FRAMES
#             eye_positions.pop(0)  # Remove the oldest value
#         else:
#             ear_right_smoothed = ear_right

#         # Check if the user is looking left, right, up, or down
#         if ear_right_smoothed < EYE_AR_THRESH:
#             print("Looking right")
#         else:
#             print("Not looking right")

#     # Display the frame
#     cv2.imshow("Eye Tracking", frame)

#     # Break the loop if 'Esc' key is pressed
#     if cv2.waitKey(1) == 27:
#         break

# # Release the video capture when the loop exits
# cam.release()
# cv2.destroyAllWindows()

# import cv2
# import pyautogui
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# moving_average_frames = 5
# eye_positions = []
# pyautogui.FAILSAFE = False
# cam = cv2.VideoCapture(1)
# screen_w, screen_h = pyautogui.size()
# while True:
#     _, image = cam.read()
#     image = cv2.flip(image, 1)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         roi_gray = gray_image[y:y + h, x:x + w]
#         _, thresholded = cv2.threshold(roi_gray, 30, 255, cv2.THRESH_BINARY)
#         contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             max_contour = max(contours, key=cv2.contourArea)
#             moment = cv2.moments(max_contour)
#             if moment["m00"] != 0:
#                 cx = int(moment["m10"] / moment["m00"])
#                 cy = int(moment["m01"] / moment["m00"])
#                 eye_positions.append((x + cx, y + cy))
#                 if len(eye_positions) > moving_average_frames:
#                     avg_x = sum(p[0] for p in eye_positions[-moving_average_frames:]) / moving_average_frames
#                     avg_y = sum(p[1] for p in eye_positions[-moving_average_frames:]) / moving_average_frames
#                     mouse_x = max(0, min(int(screen_w / w * avg_x), screen_w))
#                     mouse_y = max(0, min(int(screen_h / h * avg_y), screen_h))

#                     pyautogui.moveTo(mouse_x, mouse_y)

#                 cv2.circle(image, (x + cx, y + cy), 3, (0, 0, 255))

#     cv2.imshow("Eye controlled mouse", image)

#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# cam.release()
# cv2.destroyAllWindows()

import cv2
import dlib
import numpy as np

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
    except:
        pass

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(1)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:

        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2) #1
        thresh = cv2.dilate(thresh, None, iterations=4) #2
        thresh = cv2.medianBlur(thresh, 3) #3
        thresh = cv2.bitwise_not(thresh)
        contouring(thresh[:, 0:mid], mid, img)
        contouring(thresh[:, mid:], mid, img, True)
    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
