import cv2
import numpy as np
import matplotlib.pyplot as plt

LINES_COLOR_RGB = (255, 0, 0)
LINE_THICKNESS = 10

TEST_VIDEO = "test_LA.mp4"

# Canny contrast
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# Region of interest triangle testLA.mp4 (1280x720)
TRIANGLE_VERTEX_X_DOWN_LEFT = 300
TRIANGLE_VERTEX_X_DOWN_RIGHT = 1200
TRIANGLE_VERTEX_UPPER = (630, 370)

# Slopes
MIN_POSITIVE_SLOPE = 0.8
MAX_NEGATIVE_SLOPE = -0.8

# HoughLinesP precision
PIXELS_PRECISION = 2
DEGREE_PRECISION = np.pi / 180 # 1 degree
POINTS_THRESHOLD = 100 
MIN_LINE_LENGTH = 40
MAX_LINE_GAP = 5

LINE_START_RATIO = 3 / 5

def make_coordinates(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1 * LINE_START_RATIO)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < MAX_NEGATIVE_SLOPE:
            left_fit.append((slope, intercept))
        elif slope > MIN_POSITIVE_SLOPE:
            right_fit.append((slope, intercept))

    result = []

    if len(left_fit):
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        result.append(left_line)
    if len(right_fit):
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
        result.append(right_line)
    return np.array(result)

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

def main():
    cap = cv2.VideoCapture(TEST_VIDEO)
    while(cap.isOpened()):
        ret, frame =  cap.read()
        if ret:
            canny_image = canny(frame)

            cropped_image = region_of_interest(canny_image)

            lines = cv2.HoughLinesP(cropped_image, PIXELS_PRECISION, DEGREE_PRECISION, POINTS_THRESHOLD, np.array([]),
                minLineLength = MIN_LINE_LENGTH, maxLineGap = MAX_LINE_GAP)
            if lines is not None:
                averaged_lines = average_slope_intercept(frame, lines)
                line_image = display_lines(frame, averaged_lines)
                # line_image = display_lines(frame, lines)

                combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
                cv2.imshow("Frame", combo_image) # main
                # cv2.imshow("Frame", cropped_image)
            else:
                cv2.imshow("Frame", frame) # main
                # cv2.imshow("Frame", cropped_image)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()