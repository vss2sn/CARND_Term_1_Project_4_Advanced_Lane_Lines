## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `output_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

### REPORT - NOTES

Notes:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
Required to correct distortion. Created a .p file to store the data to speed up the process of tuning

* Use color transforms, gradients, etc., to create a thresholded binary image.
Used saturation, magnitude and direction of gradient as well as a low and high pass thresholding, as described in lectures. Would like to move the a structure where tuning is a little easier, maybe `interact` method in jupyter notebook. Retuned to improve result.

* Apply a perspective transform to rectify binary image ("birds-eye view").
Using polygon in the original image which will be transformed to a polygon in the `dst points`; hardcoded values:
src = np.float32([[293, 668], [587, 458], [703, 458], [1028, 668]])
dst = np.float32([[310, im_shape[1]], [310, 0],[950, 0], [950, im_shape[1]]])

REF: https://github.com/CYHSM/carnd/blob/master/CarND-Advanced-Lane-Lines seemed a quick fix for the transform.
Having refered to this, I would like to now put my pipeline as a jupyter notebook, but I havent had the time to do so, and I am already behind schedule, so it's a work in progress; will implement after submitting the last project.

* Detect lane pixels and fit to find the lane boundary.
Used sliding histogram for fitting curve based on lectures. Could be better. Maximum peaks in the bottom half of the image corresponding to the start of the lane. Search for the line with the same approach and fit a polynomial,

For perspective transforms, code based on code in the online lessons.

Have activelly avoided adding in things form project 1, adding those would definitely help, especially focusing on the region of interest and dynamically thresholding. This would also help in the  challenge video, where this version of the code fails under the shadow of the bridge.

* Determine the curvature of the lane and vehicle position with respect to center.
Using average of left and right curves fitted on the sides of the lane to get a more accurate representation of the curve from the viewpoint of the car. Using said average to measaure distance from the centre, under the assumption that the car remains in the center.

* Warp the detected lane boundaries back onto the original image.

* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
Pipeline allows video file as inout with output showing aforementioned video file along with curvature of lane and distance from center as well as highlighting the lane.
