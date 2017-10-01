## **Vehicle Detection**

### Writeup by Hannes Bergler
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/writeup_images/training_data.jpg
[image2]: ./output_images/writeup_images/YUV_HOG_example.jpg
[image3]: ./output_images/writeup_images/search_windows.jpg
[image4]: ./output_images/writeup_images/hot_window1.jpg
[image5]: ./output_images/writeup_images/hot_window2.jpg
[image6]: ./output_images/writeup_images/heatmap_examples.JPG
[image7]: ./output_images/writeup_images/labels_and_bboxes.JPG
[video1]: ./output_videos/project_video.mp4

## Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation. 
---

### I. Submission Files

My project includes the following files:
* writeup.md - summarizing the project results (You're reading it!)
* Vehicle_Detection_P5.ipynb - Jupyter notebook containing the project's python code
* output_images - folder containing all output images
* output_videos - folder containing the output videos

### II. Histogram of Oriented Gradients (HOG)

#### 1. Extracting HOG features from the training images

The code for this step is contained in the second code cell of the Jupyter notebook `Vehicle_Detection_P5.ipynb` located in the project's base directory.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

![alt text][image2]


#### 2. Finding good HOG parameters

I tried various combinations of parameters and found that the following parameters deliver good classification results and short computing time:

```python
hog_channels = 'ALL'
orientations = 8
pixels_per_cell = (16, 16)
cells_per_block = (2, 2)
```

#### 3. Training the classifier

I trained a linear SVM using the sklearn `LinearSVC`-function (Linear Support Vector Classification) with its default parameters.


### III. Sliding Window Search

#### 1. Sliding window search parameters


I decided to search each image with three different sliding window sizes. You can find the sizes and other window parameters in the following table.

| window size  | y_start, y_stop| xy_overlap | 
|:------------:|:--------------:|:----------:|
| 80 x 80      | 400, 520       | 0.6666     |
| 128 x 128    | 400, 656	    | 0.6666     |
| 200 x 200    | 380, 680       | 0.6666     |

The window definition can be found in the `pipeline()` function in the eighth code cell of the Jupyter notebook `Vehicle_Detection_P5.ipynb` located in the project's base directory.
The following image shows the resulting search windows on one of the test images: 

![alt text][image3]

#### 2. Example images

Ultimately I searched on three scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4] | ![alt text][image5]


### IV. Video Implementation

#### 1. Link to my final video output

Here's a [link to my video result](./output_videos/project_video.mp4)


#### 2. Filter for false positives

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

##### Here are three frames and their corresponding heatmaps:

![alt text][image6]

##### Here is the output of `scipy.ndimage.measurements.label()` on the thresholded heatmap for the same three frames and their resulting bounding boxes:
![alt text][image7]


#### 3. Providing a more stable estimate for videos

I implemented the class `HeatmapBuffer()` which can be found in code cell seven of the Jupyter notebook. The class stores the heatmaps of the last video frames. For a more stable estimate of the vehicle positions, I add up the heatmaps of the last five frames and apply an overall threshold `overall_thresh` on the resulting heatmap, see `pipeline()` function in code cell eight.


### V. Discussion

If I had more time to work on this project, I would implement a more sophisticated method to track detected vehicles. The current implementation would have problems with fast moving objects due to fact that I simply add up the heatmaps of several consecutive frames. Taking the speed of detected objects into account, would be a good idea.
The performance of my pipeline of 1.0 to 1.5 frames per second on my laptop is another issue. Calculating the hog features only once for each frame could help to improve the performance. I haven't implemented this so I would be more flexible with experimenting different search window sizes and window overlaps. In the end, I didn't have the time to integrate this step.

Last but not least, false positives are still an issue. I would suggest a second software layer which carries out sanity checks on all detections. For example a vehicle that pops out of nothing in the middle of the road and disappears again after a view frames, is most likely a false detection.
