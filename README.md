# Computer-Vision-ND
This repository was created for the [Computer Vision Nanodegree ](https://www.udacity.com/course/computer-vision-nanodegree--nd891) at [Udacity](https://Udacity.com).

## Project 1: Facial Keypoint Detection

[Facial Keypoint Detection Project](project_1_facial_keypoints)<br/>

In this project, I build a facial keypoint detection system. The system consists of a face detector that uses Haar Cascades and a Convolutional Neural Network (CNN) that predict the facial keypoints in the detected faces. The facial keypoint detection system takes in any image with faces and predicts the location of 68 distinguishing keypoints on each face.

Some results from my facial keypoint detection system:

<img src="images/beatles_resnet.png" width="512">
<img src="gifs/face_mask_test.gif?" width="512"><br>

The Udacity repository for this project: [P1_Facial_Keypoints](https://github.com/udacity/P1_Facial_Keypoints)

## Project 2: Image Captioning

[Image Captioning Project](project_2_image_captioning_project)<br/>

In this project, I design and train a CNN-RNN (Convolutional Neural Network - Recurrent Neural Network) model for  automatically generating image captions. The network is trained on the Microsoft Common Objects in COntext [(MS COCO)](http://cocodataset.org/#home) dataset. The image captioning model is displayed below.

![Image Captioning Model](images/cnn_rnn_model.png?raw=true) [Image source](https://arxiv.org/pdf/1411.4555.pdf)

One good and one not so good sample made by my model:

![sample_171](images/sample_171.png?raw=true)<br/>
![sample_193](images/sample_193.png?raw=true)<br/>

The Udacity repository for this project: [CVND---Image-Captioning-Project](https://github.com/udacity/CVND---Image-Captioning-Project)

## Project 3: Landmark Detection

[Landmark Detection Project](project_3_landmark_detection)<br/>

In this project, I implement SLAM (Simultaneous Localization and Mapping) for a 2-dimensional world.  Sensor and motion data gathered by a simulated robot is used to create a map of an environment. SLAM gives us a way to track the location of a robot in the world in real-time and identify the locations of landmarks such as buildings, trees, rocks, etc.

 <img src="images/robot_world.png?" width="512">

The Udacity repository for this project: [Project_Landmark Detection](https://github.com/udacity/CVND_Localization_Exercises/tree/master/Project_Landmark%20Detection)

## Exercises

[image-representation](exercises/1_1_Image_Representation)<br/>
[convolutional-filters-edge-detection](exercises/1_2_Convolutional_Filters_Edge_Detection)<br/>
[types-of-features-image-segmentation](exercises/1_3_Types_of_Features_Image_Segmentation)<br/>
[feature-vectors](exercises/1_4_Feature_Vectors)<br/>
[CNN-layers](exercises/1_5_CNN_Layers)<br/>
[YOLO](exercises/2_2_YOLO)<br/>
[LSTMs](exercises/2_4_LSTMs)<br/>
[attention](exercises/2_6_Attention)<br/>

The original notes can be found in the Udacity [CVND_Exercises](https://github.com/udacity/CVND_Exercises) repo.

##  Localization Exercises

[optical-flow](localization_exercises/4_1_Optical_Flow)<br/>
[robot-localization](localization_exercises/4_2_Robot_Localization)<br/>
[2D-histogram-filter](localization_exercises/4_3_2D_Histogram_Filter)<br/>
[kalman-filters](localization_exercises/4_4_Kalman_Filters)<br/>
[state-and-motion](localization_exercises/4_5_State_and_Motion)<br/>
[matrices-and-transformation-of-state](localization_exercises/4_6_Matrices_and_Transformation_of_State)<br/>
[SLAM](localization_exercises/4_7_SLAM)<br/>
[vehicle-motion-and-calculus](localization_exercises/4_8_Vehicle_Motion_and_Calculus)<br/>

The original notes can be found in the Udacity [CVND_Localization_Exercises](https://github.com/udacity/CVND_Localization_Exercises) repo.



### README TODO
- [ ] Add an image and/or a GIF from the Facial Keypoint Detection project
- [ ] Add an image and/or a GIF from the Landmark Detection project
- [x] Write better description for Project 3: Landmark Detection
