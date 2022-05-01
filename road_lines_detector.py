import cv2
import numpy as np
import matplotlib.pyplot as plt

LINES_COLOR_RGB = (255, 0, 0)
LINE_THICKNESS = 10

# canny contrast
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# region of interest triangle
TRIANGLE_VERTEX_X_DOWN_LEFT = 200
TRIANGLE_VERTEX_X_DOWN_RIGHT = 1100
TRIANGLE_VERTEX_UPPER  = (550, 250)

# HoughLinesP precision
PIXELS_PRECISION = 2
DEGREE_PRECISION = np.pi / 180 # 1 degree
POINTS_THRESHOLD = 100 
MIN_LINE_LENGTH = 40
MAX_LINE_GAP = 5

def canny(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0) 
    canny = cv2.Canny(blur, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(TRIANGLE_VERTEX_X_DOWN_LEFT, height),
        (TRIANGLE_VERTEX_X_DOWN_RIGHT, height),
        TRIANGLE_VERTEX_UPPER]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), LINES_COLOR_RGB, LINE_THICKNESS)
    return line_image

image = cv2.imread('test_image.png')
lane_image = np.copy(image)
canny = canny(lane_image)

cropped_image = region_of_interest(canny)

lines = cv2.HoughLinesP(cropped_image, PIXELS_PRECISION, DEGREE_PRECISION, POINTS_THRESHOLD, np.array([]), minLineLength = MIN_LINE_LENGTH, maxLineGap = MAX_LINE_GAP)
line_image = display_lines(lane_image, lines)

combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# plt.imshow(roi)
# plt.show()
cv2.imshow("image", combo_image)
cv2.waitKey(0)
