import cv2
import numpy as np
import matplotlib.pyplot as plt

LINES_COLOR_RGB = (255, 0, 0)
LINE_THICKNESS = 10

# canny contrast
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# region of interest triangle
TRIANGLE_VERTEX_X_DOWN_LEFT = 180
TRIANGLE_VERTEX_X_DOWN_RIGHT = 1660
TRIANGLE_VERTEX_UPPER  = (1000, 550)

# HoughLinesP precision
PIXELS_PRECISION = 2
DEGREE_PRECISION = np.pi / 180 # 1 degree
POINTS_THRESHOLD = 100 
MIN_LINE_LENGTH = 40
MAX_LINE_GAP = 5

LINE_START_RATIO = 2 / 5

def make_coordinates(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1 * LINE_START_RATIO)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return np.array([np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])])
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    if len(left_fit) and len(right_fit):
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])
    return np.array([np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])])


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

# image = cv2.imread('road2.png')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)

# cropped_image = region_of_interest(canny_image)

# lines = cv2.HoughLinesP(cropped_image, PIXELS_PRECISION, DEGREE_PRECISION, POINTS_THRESHOLD, np.array([]), minLineLength = MIN_LINE_LENGTH, maxLineGap = MAX_LINE_GAP)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, lines)

# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# plt.imshow(combo_image)
# plt.show()
# cv2.imshow("image", cropped_image)
# cv2.waitKey(0)

cap = cv2.VideoCapture("test4.mp4")
while(cap.isOpened()):
    _, frame =  cap.read()
    canny_image = canny(frame)

    cropped_image = region_of_interest(canny_image)

    lines = cv2.HoughLinesP(cropped_image, PIXELS_PRECISION, DEGREE_PRECISION, POINTS_THRESHOLD, np.array([]), minLineLength = MIN_LINE_LENGTH, maxLineGap = MAX_LINE_GAP)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)

    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    # plt.imshow(combo_image)
    # plt.show()
    cv2.imshow("image", combo_image)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()