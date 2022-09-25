import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])# bottom of the image
    y2 = int(y1*(3/5))         # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])




def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2) ,(y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0: # y is reversed in image
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line  = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line,right_line],np.int32)


# detect edges
def CannyEdgeDetect(lane_image):
    # convert image to grayscale
    gray_lane_image = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)
    # applay Gaussian Blur to reduce noise and canny to detect edges
    blur_image = cv2.GaussianBlur(gray_lane_image,(5,5),0)
    canny_image = cv2.Canny(blur_image,50,150)
    return canny_image


def interest_region(image):
    height = image.shape[0]
    width = image.shape[1]
    mask = np.zeros_like(image)
    triangle = np.array([[
        #(100, height), (width, height), (width-500, int(height/1.9))
        (0, height),(width/2, height/2),(width, height)
        ]],np.int32)
    cv2.fillPoly(mask,triangle,255)
    mask_image = cv2.bitwise_and(image,mask)
    return mask_image

def display_Lines(image,lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lines_image,(x1,y1),(x2,y2),(0,0,255),10)
    return lines_image


Laneoutput = cv2.VideoWriter('C:/Users/ramig/Desktop/deep learning with tensorflow/Jordan Road Object Detection/Data/Output Videos/LaneDetectionOutput.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, (416,416))
cap = cv2.VideoCapture("C:/Users/ramig/Desktop/deep learning with tensorflow/Jordan Road Object Detection/Data/Input Videos/Road For Lanes 2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = CannyEdgeDetect(frame)
    cropped_canny = interest_region(canny_image)
   
    lines = cv2.HoughLinesP(cropped_canny, 3, np.pi/180, 400, np.array([]), minLineLength=150, maxLineGap=60)

    try:
        averaged_lines = average_slope_intercept(frame, lines)
        line_image = display_Lines(frame, averaged_lines)
    except:
        line_image = display_Lines(frame, lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    Laneoutput.write(combo_image)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
Laneoutput.release()
cap.release()






