# Computer Vision Nanodegree
This repository contains my exercises and projects for the [Computer Vision Nanodegree ](https://www.udacity.com/course/computer-vision-nanodegree--nd891) at [Udacity](https://Udacity.com).

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

* [Image Representation & Classification](exercises/1_1_Image_Representation) - In this exercise, I learn how images are represented numerically and implement image processing techniques, such as color masking and binary classification.
* [Convolutional Filters and Edge Detection](exercises/1_2_Convolutional_Filters_Edge_Detection) - In this exercise, I learn about frequency in images and implement my own image filters for detecting edges and shapes in an image. Use Haar cascade classifiers from the OpenCV library to perform face detection.
* [Types of Features & Image Segmentation](exercises/1_3_Types_of_Features_Image_Segmentation) - In this exercise, I program a corner detector and learn techniques, like k-means clustering, for segmenting an image into unique parts. 
* [Feature Vectors](exercises/1_4_Feature_Vectors) - In this exercise, I learn how to describe objects and images using feature vectors (ORB, FAST, BRIEF, HOG).
* [CNN Layers and Feature Visualization](exercises/1_5_CNN_Layers) - In this exercise, I define and train my own convolution neural network for clothing recognition. Learn to use feature visualization techniques to see what the network had learned.
* [YOLO](exercises/2_2_YOLO) - In this exercise, I learn about the YOLO (You Only Look Once) multi-object detection model and work with a YOLO implementation. Implement YOLO to work with my webcam.
* [LSTMs](exercises/2_4_LSTMs) - In this exercise, I learn about Long Short-Term Memory Networks (LSTM), and similar architectures which have the benefits of preserving long-term memory. Implement a Character-Level LSTM model. 
* [Attention Mechanisms](exercises/2_6_Attention) -  Todo.

The Udacity repository for the exercises: [CVND_Exercises](https://github.com/udacity/CVND_Exercises)

##  Localization Exercises

* [Optical Flow](localization_exercises/4_1_Optical_Flow) - In this exercise, I learn about and implement Optical Flow.
* [Robot Localization](localization_exercises/4_2_Robot_Localization) - In this exercise, I learn how to implement a Bayesian filter to locate a robot in space and represent uncertainty in robot motion.
* [Mini-project: 2D Histogram Filter](localization_exercises/4_3_2D_Histogram_Filter) - In this exercise, I write sense and move functions for a (and debug) 2D histogram filter.
* [Introduction to Kalman Filters](localization_exercises/4_4_Kalman_Filters) - In this exercise, I learn the intuition behind the Kalman Filter, a vehicle tracking algorithm, and implement a one-dimensional tracker.
* [Representing State and Motion](localization_exercises/4_5_State_and_Motion) - In this exercise, I learn to represent the state of a car in a vector that can be modified using linear algebra.
* [Matrices and Transformation of State](localization_exercises/4_6_Matrices_and_Transformation_of_State) - In this exercise, I learn about the matrix operations that underly multidimensional Kalman Filters.
* [Simultaneous Localization and Mapping (SLAM)](localization_exercises/4_7_SLAM) - In this exercise, I learn how to implement SLAM: simultaneously localize an autonomous vehicle and create a map of landmarks in an environment.
* [Vehicle Motion and Calculus](localization_exercises/4_8_Vehicle_Motion_and_Calculus) - In this exercise, I review some basic calculus and learn how to derive the x and y components of a self-driving car's motion from sensor measurements and other data.

The Udacity repository for the exercises: [CVND_Localization_Exercises](https://github.com/udacity/CVND_Localization_Exercises) 
