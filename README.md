# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


Overview
---

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### Reflection

##### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.
We first import packages and load a test image. **Notice that matplotlib read image as RGB, however the cv2 read the image as GBR.**
```
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline

image = cv2.imread('test.jpg')
```

The pipeline mainly contains the following steps:
##### 1). Build a filter to find the white and yellow parts:
The lane lines are always white and yellow. Hence, we first extract all the white and yellow parts from the image by filter_color function. We use the RGB, HSV and HLS to do that. 
```
kernel_size = 5
filtered_image = filter_colors(image)
plt.figure()
plt.imshow(filtered_image)
```
```
def filter_colors(image):
    # Filter the white and yellow lines in images.
    # To reinforce the performance, we use both hls and hsv to find the yellow and white lines.
    
    # white BGR
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_RGB_image = cv2.bitwise_and(image, image, mask=white_mask)

    # yellow hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([90,100,100])
    upper_yellow = np.array([110,255,255])
    # lower_yellow = np.array([20,100,100])
    # upper_yellow = np.array([30,255,255])
    
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_hsv_image = cv2.bitwise_and(image, image, mask=yellow_mask)

    hsv_image = weighted_img(white_RGB_image, 1., yellow_hsv_image, 1., 0.)
    
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    # white hls
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    
    white_hls_mask = cv2.inRange(hls, lower, upper)
    white_hls_image = cv2.bitwise_and(image, image, mask=white_hls_mask)

    # yellow hls
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_hls_mask = cv2.inRange(hls, lower, upper)
    yellow_hls_image = cv2.bitwise_and(image, image, mask=yellow_hls_mask)

    hls_image = weighted_img(white_hls_image, 1., yellow_hls_image, 1., 0.)

    final_image = weighted_img(hls_image, 1., hsv_image, 1., 0.)
    
    return final_image
```
![Filter white and yellow parts](http://upload-images.jianshu.io/upload_images/2528310-b91234924f534566.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 2). Convert image to grey image and remove the noises
cv2.GaussianBlur is used to remove the noises from the gray image. Removing the noise is very important. In the next step, we will show the differences.
```
gray = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2GRAY)
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
```
![Convert image to grey image and remove the noises](http://upload-images.jianshu.io/upload_images/2528310-60d4fea28d717c0b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



##### 3). Canny edges detection
In this part, we use canny edges detection method to find all the lines in images. With noises, the lines make a real mess. 
Without noises, the edges of lane line is very clear.
```
# Canny and apply on original image
# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150

edges = cv2.Canny(image, low_threshold, high_threshold)
plt.figure()
plt.imshow(edges)
```
![Canny edges detection on original image](http://upload-images.jianshu.io/upload_images/2528310-fd46537b42be102a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
plt.figure()
plt.imshow(edges)
```
![Canny edges detection on image without noise](http://upload-images.jianshu.io/upload_images/2528310-3c91d52f68845c4c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 4). Region of interest
In the previous step, we find that most of the edges are not lane lines edges. And the lane lines edges always locate in a trapezoid. Hence, we define a trapezoid to restrict the region of interest.
```
# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)   
ignore_mask_color = 255  

# This time we are defining a four sided polygon to mask
imshape = image.shape
vertices = np.array([[(imshape[1]*0.15,imshape[0]*0.95),(imshape[1]*0.45, imshape[0]*0.6), (imshape[1]*0.55, imshape[0]*0.6), (imshape[1]*0.9,imshape[0]*0.95)]], dtype=np.int32)


cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)
plt.figure()
plt.imshow(masked_edges)
```
![Region of interest: a trapezoid](http://upload-images.jianshu.io/upload_images/2528310-603bc13abff7cb4d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 5). Filter edges and generate lines by HoughLinesP function
Not all edges in the trapezoid are useful. We use HoughLinesP with specific parameters to remove some short and useless edges.
```
# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 3 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 5 #minimum number of pixels making up a line
max_line_gap = 25    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)

```
##### 6. Find the lane line related edges
We deal with the left and right lane line, separately. 
Based on the observation, the slope of lines in left part are larger than 0.5, while the slope of lines in right part are less than -0.5.
```


# Iterate over the output "lines" and draw lines on a blank image
left_lane_lines_x = []
left_lane_lines_y = []
right_lane_lines_x = []
right_lane_lines_y = []
    
x_size = image.shape[1]
y_size = image.shape[0]
for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(x1-x2) == 0:
        slope = float("inf")
    else:
        slope = (y2 - y1)/(x2 - x1)

    if slope > 0.5 and x1 > x_size/2 and x2 > x_size/2:
        right_lane_lines_x.append(x1)
        right_lane_lines_x.append(x2)
        right_lane_lines_y.append(y1)
        right_lane_lines_y.append(y2)
    elif slope < -0.5 and x1 < x_size/2 and x2 < x_size/2:
        left_lane_lines_x.append(x1)
        left_lane_lines_x.append(x2)
        left_lane_lines_y.append(y1)
        left_lane_lines_y.append(y2)
```
##### 7). Fit the lane line and generate the start and end point of the fitted line
Fit the lane line with several points. Then we get the slope and bias of the line. We calculate the start and end point, and draw lines in the image.
```
r_m, r_b = np.polyfit(right_lane_lines_x, right_lane_lines_y, 1)
l_m, l_b = np.polyfit(left_lane_lines_x, left_lane_lines_y, 1)
        
y1 = image.shape[0]
y2 = image.shape[0] * (1 - 0.35)


r_x1 = (y1 - r_b) / r_m
r_x2 = (y2 - r_b) / r_m
    
l_x1 = (y1 - l_b) / l_m
l_x2 = (y2 - l_b) / l_m
```

##### 8). Draw it on the image
```
color = [255,0,0]
thickness = 10
cv2.line(image, (int(r_x1), y1), (int(r_x2), int(y2)), color, thickness)
cv2.line(image, (int(l_x1), y1), (int(l_x2), int(y2)), color, thickness)

color_edges = np.dstack((edges, edges, edges)) 

lines_edges = cv2.addWeighted(image, 1, line_image, 1, 0) 
plt.figure()
plt.imshow(lines_edges)
```

![The lane line on image.](http://upload-images.jianshu.io/upload_images/2528310-43fc657a4b172681.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**In order to draw a single line on the left and right lanes, I modified the draw_lines() function by the following steps:**
###### a. Deal with left and right edges separately, based on 
      i. the absolute value of the slope is larger than 0.5
     ii. the left lane edges should lie in the left part of the image; the right lane edges should lie in the right part of the image
###### b. Fit the points with np.polyfit
###### c. Smooth the slope and bias with a size=10 queue. The idea is that the slope and bias should be similar in the past 10 frames.  The experimental results are very good. After smoothing the slope and bias, the lines won't shake like an earthquake. 

```
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
#     for line in lines:
#         for x1,y1,x2,y2 in line:
#             cv2.line(img, (x1, y1), (x2, y2), color, thickness)
      
    left_lane_lines_x = []
    left_lane_lines_y = []
    right_lane_lines_x = []
    right_lane_lines_y = []

    x_size = img.shape[1]
    y_size = img.shape[0]
    for line in lines:
        if len(line[0])>3:
            x1, y1, x2, y2 = line[0]
        else:
            continue
        if abs(x1-x2) == 0:
            slope = float("inf")
        else:
            slope = (y2 - y1)/(x2 - x1)

        if slope > 0.5 and x1 > x_size/2 and x2 > x_size/2:
            right_lane_lines_x.append(x1)
            right_lane_lines_x.append(x2)
            right_lane_lines_y.append(y1)
            right_lane_lines_y.append(y2)
        elif slope < -0.5 and x1 < x_size/2 and x2 < x_size/2:
            left_lane_lines_x.append(x1)
            left_lane_lines_x.append(x2)
            left_lane_lines_y.append(y1)
            left_lane_lines_y.append(y2)
    right_exist = False
    left_exist = False
    if right_lane_lines_x != [] and right_lane_lines_y!= []:
        r_m, r_b = np.polyfit(right_lane_lines_x, right_lane_lines_y, 1)
        right_exist = True
        
    if left_lane_lines_x != [] and left_lane_lines_y!= []:
        l_m, l_b = np.polyfit(left_lane_lines_x, left_lane_lines_y, 1)
        left_exist = True
    
    # Smoothing the slope and bias to make the lane line detection robust.
    if left_exist:
        if left_q_m.size() < 10:
            left_q_m.put(l_m)
        else:
            left_q_m.get()
            left_q_m.put(l_m)

        if left_q_b.size() < 10:
            left_q_b.put(l_b)
        else:
            left_q_b.get()
            left_q_b.put(l_b)
            
    if right_exist:      
        if right_q_m.size() < 10:
            right_q_m.put(r_m)
        else:
            right_q_m.get()
            right_q_m.put(r_m)

        if right_q_b.size() < 10:
            right_q_b.put(r_b)
        else:
            right_q_b.get()
            right_q_b.put(r_b)
    
    r_m = right_q_m.avg()
    l_m = left_q_m.avg()
    r_b = right_q_b.avg()
    l_b = left_q_b.avg()
    
    y1 = img.shape[0]
    y2 = img.shape[0] * (1 - 0.35)


    r_x1 = (y1 - r_b) / r_m
    r_x2 = (y2 - r_b) / r_m

    l_x1 = (y1 - l_b) / l_m
    l_x2 = (y2 - l_b) / l_m

    cv2.line(img, (int(r_x1), y1), (int(r_x2), int(y2)), color, thickness)
    cv2.line(img, (int(l_x1), y1), (int(l_x2), int(y2)), color, thickness)
```
The queue is defined as follows. We can also use the class Queue in built-in functions.
```
class Queue:
    """
    Define a queue class for smoothing the slope and bias of the lane line.
    Smoothing is necessary for this task. 
    """
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []
    
    def Empty(self):
        while self.isEmpty() == False:
            self.items.get()
        return self.items
    
    def put(self, item):
        self.items.insert(0,item)
        
    def avg(self):
        return np.mean(self.items)

    def get(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
```


### 2. Identify potential shortcomings with your current pipeline
Shortcoming:
In the challenge.mp4, the algorithm works well on almost all the frames except frame 113 and 114. I run the pipeline on the two frames, and it works well. However, it doesn't work well on the video. I spent 2 days to improve the performance on the two frames. Failed...

Potential shortcoming:
1. The test videos are limited. The algorithm may fail on many other situations. For instance, night, rainy, snow and sand storm.

2. Many of the parameters are fixed, which may have great limitations. As the development of machine learning, is it possible for the ML algorithm to learn these automatically?


### 3. Suggest possible improvements to your pipeline

I really want to test the algorithm on a S-Bend road. I hope that the length of the annotated lane line is adaptive. We use the linear regression to fit the lane line. It's more reasonable to fit the lane line as a curve.
