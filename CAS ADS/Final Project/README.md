# 1	Project Overview 
In this project, object detection and localization techniques are utilized using supervised & semi-supervised machine learning to detect other vehicles on road for a self-driving car. A deep learning Convolutional Neural Network (CNN) classification model predicts if an image of the road has any car and localizes them by generating a bounding box around its whereabouts in that image. Images from camera mounted in the vehicle looking forward are fed in a Convolutional Neural Network to predict if there are other cars in the vicinity. 
An implementation of YOLO algorithm [1] is used to detected vehicles and bounding boxes are generated to localize the detection. The state-of-the-art YOLO algorithm localizes multiple cars in a single image by performing a sliding window operation. 

## 1.1	Motivation
For self-driving vehicles, safety is of a prime concern. It is therefore very important that the object is precisely and accurately detected in real time. Since autonomous cars interact most with other cars, detecting other cars timely and accurately in the vicinity is the very first step in building a fully autonomous vehicle driving system. Accordingly, this project implements a YOLO based detection that recognize and localizes cars meticulously in real time when input video stream from the cameras is given to the deep learning network. The ‘You-Only-Look-Once’ (YOLO) algorithm for object localization is designed from scratch without using the built-in libraries to understand the intuition behind it. Since the project is based on videos and images, learning computer vision and image processing was also a motivating factor along with implementing machine learning and deep learning techniques. Moreover, the raw data used in the project is also collected from scratch to add robustness to the model. In addition, some already available car images datasets have also been unutilized to complement the train set and making it more comprehensive. 

#2	Data collection
The data is primarily collected from the MP4 videos downloaded from a car’s dash-cam. Additionally, relevant images from Stanford Car Images Dataset [2] specifically those having bounding boxes of a particular proportion in the image were also added to the training data set. Furthermore, to increase car models diversity, several images from the Geneva car show dataset [3] with images of car fronts, backs and sides only were also included in the training set. A total of 5366 images were collected and labelled to train the model out of which 2810 images had cars in them, and 2556 images were no-car images. 

## 2.1	Data labelling
Almost all the training images were collected from scratch with front mounted car's dash cam and were manually to be labelled with the help of a custom labelling tool. Labelled images of cars with bounding box information were then fed to the neural network. The car images were collected and labelled manually because of the following reasons: 
•	The data is collected in the Swiss region only, to be more specific and for the model to perform better
•	Only those images that had cars from the front and back were included from the Stanford dataset, to match with the data that was collected manually
All labelled images were stored as JPEG images with dimensions of 480 x 270 pixels. Following sections explain the different types of data labelling techniques employed.

### 2.1.1	Internal labelling 
Using CV2 library, the videos are streamed frame by frame and then a bounding box is dragged and drawn manually. The images of cars and no-car (background) are then saved, whereas the bounding box information is stored in a csv representing the coordinates of the four corners of the bounding box. 
For image processing, utility functions were built to rotate image, rescale image, and crop image. 

### 2.1.2	External labelling
A hand full car images from other car datasets having bounding box information were also used. Qualifying images had car bounding boxes between 80 and 40% of the total dimension of 480 x 270 pixels. For this, the center point of the bounding box was calculated and then the frame was added covering the bounding box. Also, only those images were taken that mostly had car fronts and backs, to make it suitable for a forward mounted vehicle detection system. 

### 2.1.3	Semi supervised labelling
In semi supervised labelling, the trained model was iteratively run-on new videos and classified images with cars having high confidence were saved. These images were then verified and re-labelled for output labels and bounding boxes. The re-labelled images were then added back to the input data set to improve the training accuracy, and the model was re-trained. This resulted in an increase of overall accuracy of the model. 

## 3	Training the CNN model
All labelled images for training were added in the data directory, with images having cars in 'data\c' subdirectory and those without cars in 'data\n' subdirectory. An index CSV file contained the relative link to the image directory and the bounding box dimensions. For images with cars, the points with upper left corner and bottom right corner of the bounding boxes were stored in the index file. For images with no car, zeros were stored for bounding box points. For training the model, this file was loaded in data a frame and an additional label field was calculated for class label.

## 4 Convolutional Neural Network model
A simple model is defined with i=5 blocks of convolutional layers each having 2i filters and Relu activation, batch normalization and Max pooling 2D. In the end its flattened and two dense layers of size 256 and 64 were added respectively. 

## 5	The YOLO Implementation
The ‘You Only Look Once’ or YOLO algorithm sees the entire image and requires only a single forward propagation to detect objects on run-time. In this algorithm, a grid is placed on the image and then a single neural network is applied to the entire image. This is an overlap in successive images, both horizontally and vertically to cater for split cars by the grid slicing which was kept as 25% of image height and length by default. This outputs the image classification and localization for each cell of the grid. Hence for each grid cell, there is a Y label. 

## 6	Conclusions
In this project, Car detection with 99% accuracy and localization with 80% IoU was achieved. The data was collected from scratch and was region specific. YOLO algorithm was designed to make predictions in real-time. 
The algorithm was able to make accurate predictions of cars. However, there were few wrong predictions, long vehicles such as trucks were not detected and road signs were sometimes wrongly detected as car. This is because, the data used was limited. With more data, other vehicles such as bicycle, motorcycle and buses, the project can be further improved and used as crucial part of self-driving vehicle along with other specialized autonomous functions. 
