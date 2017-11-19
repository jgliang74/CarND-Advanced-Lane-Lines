## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/undistored_test3.jpg "Road Image Undistorted"
[image3]: ./output_images/binary_combo_test5.jpg "Binary Example"
[image4]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Camera Calibration

##### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "advanced_lane_lines.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
The following image shows undistortion matrix calibrated from previous step was applied on one of test images captured from camera mounted on the car. you can tell the differnce from the shape of tree on middle fo left side of image. but the effect is subtle.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color (s channel of HLS color space) and gradient thresholds to generate a binary image {in cell 3 of "advanced_lane_lines.ipynb" - function color_gradient_threshold()}. The final image color_binary is a combination of binary thresholding the S channel (HLS) and binary thresholding the result of applying the Sobel operator in the x direction on the original image. Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `find_M_Minv()`, which appears in lines 1 through 8 in the 5th code cell of the IPython notebook("advanced_lane_lines")  The `find_M_Minv()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[205,718],
                  [590,450],
                  [680,450],
                  [1100,718]])

dst = np.float32([[320, img.shape[0]],
                  [320, 0],
                  [960, 0],
                  [960, img.shape[0]]])
```
This resulted in the following source and destination points:

| Source         | Destination   | 
|:--------------:|:-------------:| 
| 205,  718      | 320, 0        | 
| 590,  450      | 320, 720      |
| 680,  450      | 960, 720      |
| 1100, 718      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4.Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The functions fit_left_right_curves() and fit_without_slidingwindow_search() are defined in Jupyter notebook to identify lane lines and fit a second order polynomial to both right and left lane lines. fit_left_right_curves() is identical to the reference code which computes a histogram of the bottom half of the image and finds the bottom-most x position (or "base") of the left and right lane lines. By centering each window on the midpoint of the pixels from the window below, the function then identifies nine windows to track potential lane pixels wiht highest possibility. After pixels belonging to each lane line are identified and the Numpy polyfit() method fits a second order polynomial to each set of pixels. The image below demonstrates how this process works:

![alt text][image5]

In order to speed up the process, fit_without_slidingwindow_search() uses previously identified lines (prev_left_fit, prev_right_fit) as the based position to start with.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated in the function cell named calculate_curverad() using this line of code for both left line curvature and right line curvature.

    curve_radius = ((1 + (2*fit[0] * y_0 * y_meters_per_pixel + fit[1])^2)**1.5) /  np.absolute(2*fit[0])

In above equation, fit[0] is the first coefficient (Referred as A in the second order polynominal X = Ay^2 + By +C) of the second order polynomial fit, and fit[1] is the second (B) coefficient. y_0 is the position of the car in y direction of the image. y_meters_per_pixel is the factor used for converting from pixels to meters.

The position of the vehicle with respect to the center of the lane is calculated with the following lines of code:

    lane_center_position = (rightx_bot + leftx_bot)/2
    distance_from_center = abs((lane_center_position - car_center)*3.7/700) 
    
rightx_bot and leftx_bot are the x-intercepts of the right and left fits at the bottom of image, respectively(Evaluate x by setting y = 719). The car position is the difference between these intercept points and the image midpoint (assuming that the camera is mounted at the center of the vehicle).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Function draw_lines_on_original_image() is implemented to plot back the identified lane down onto the road image. In this function, the fitted lane lines are first drawn on a blank image,  this image is then warped back to the original image space through inverse perspective matrix (Minv). The resulting image is shown as the following:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline submitted works reasonably well for the project video. I choose the color thresholding on S channel combined wiht thresholding on the result of applying the Sobel operator in the x direction. Together with region masking, it can pick up both lane lines throughout the entire video. Line object is used for continously frame processing, It only tracks good fits (last 10 frames) and will reject bad fits. Simple sanity checking, like lane width check and fit coefficents difference check, are implemented to reject bad fits. For displaying detected lane lines, best fit is used by averaging polynominal coefficents from last n good fits store in the line objects.
However, when trying to apply this pipeline to the chanllenge video, it didn't plug and play. A few debug plots are shown as the following 

![alt text][image7]
![alt text][image8]

It clear shows the gradient thresholding generated too much noise on the backgroud which was difficult for region masking to get rid of (due to the curly road, hard to define a suitable masking region). Those noise further confused the search algorithm so that wrong lines are detected.

Comparing the project video, the challenge video has following difficulties:
1). lighting condition changes drastically from frame to frame making thresholding a lot more difficult.
2). Curly road with sharp turns lead to massive background noises presented in binary image.
3). Lane lines disappear for long time in bright or sharply tuning conditions.

For the future improvements, I would focus on two major aspects:
1). To explore more on using different color spaces for thresholding and to be less relying on gradient based threshold which tended to generate more noises with inappropriate threshold settings.
2). To explore more advanced sanity checking and rejecting algorithms. Current one is quite rudimentary. Besides using estimated lane width, there are many other information (for instance, smootheness in curveratures along one line or parallelism between right line and left line ) can be used to evaluate detection confidence level. It is beneficil to reject any potential bad detection so that recent detection data will be kept unpolluted in the buffer.

