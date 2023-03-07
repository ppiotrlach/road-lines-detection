# road-lines-detection
assignment for "image recognition and processing" classes - 6 semester


This code is a Python script for detecting and drawing road lines in a video using OpenCV.

To use the code, follow these steps:

Create a virtual environment using Python 3:

```
python3 -m venv venv/
```
<br>

Activate the virtual environment:
```
. venv/bin/activate
```
<br>

Install the required dependencies using pip:

```
pip install -r requirements.txt
```

<br>

Run the Python script:
```
python3 road_lines_detector.py
```
<br><br>
The script will open a video file (specified by the TEST_VIDEO constant), and then use various image processing techniques to detect the road lines in each frame of the video. The detected lines are then drawn onto the original frame, and the resulting image is displayed in a window.

<br><br>
The script uses the following constants to control its behavior:

LINES_COLOR_RGB: The RGB color of the lines to be drawn.
LINE_THICKNESS: The thickness of the lines to be drawn.
TEST_VIDEO: The path to the video file to be processed.
TRIANGLE_VERTEX_X_DOWN_LEFT, TRIANGLE_VERTEX_X_DOWN_RIGHT, and TRIANGLE_VERTEX_UPPER: The vertices of a triangle that defines the region of interest in the image.
CANNY_LOW_THRESHOLD and CANNY_HIGH_THRESHOLD: The low and high thresholds for the Canny edge detection algorithm.
MIN_POSITIVE_SLOPE and MAX_NEGATIVE_SLOPE: The minimum positive and maximum negative slopes for a line to be considered as part of the left and right lanes, respectively.
PIXELS_PRECISION, DEGREE_PRECISION, POINTS_THRESHOLD, MIN_LINE_LENGTH, and MAX_LINE_GAP: Parameters for the Hough transform algorithm used to detect the lines.

<br><br>
The script consists of several functions:

make_coordinates: Given an image and a set of line parameters (slope and intercept), returns the coordinates of the two endpoints of the line.
average_slope_intercept: Given an image and a set of lines detected by the Hough transform, returns the average slope and intercept of the left and right lanes, respectively.
canny: Applies the Canny edge detection algorithm to an image.
region_of_interest: Applies a mask to an image to keep only the region of interest defined by the triangle.
display_lines: Draws lines onto an image.
main: The main function that reads frames from the video file, applies the various image processing techniques, and displays the resulting image.

