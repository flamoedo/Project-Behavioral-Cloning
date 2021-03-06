
# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


# * Check the driving performance on the following links:

![alt text](./images/Video_track1.PNG) 

"Video track 1"

![alt text](./images/Video_track2.PNG) 

"Video track 2"


https://youtu.be/LtgchH1XL90

https://youtu.be/2eYJb210tpo


[//]: # (Image References)

[image1]: ./images/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./images/center_2016_12_01_13_31_13_482.jpg "Center image"
[image3]: ./images/left_2016_12_01_13_31_13_482.jpg "Left Image"
[image4]: ./images/right_2016_12_01_13_31_13_482.jpg "Right Image"
[image5]: ./images/left_2016_12_01_13_31_13_482_flipped.jpg "Flipped Image"
[image6]: ./images/right_2016_12_01_13_31_13_482_flipped.jpg "Flipped Image"


## Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image  							|
| Lambda         		| Normalization: x / 255.0 - 0.5				|
| Cropping         		| Top: 75pix, Button: 25pix 					|
| Convolution      | Filter: 5x5, Depth: 24, stride: 2x2, valid padding|
| RELU					|												|
| Convolution     | Filter: 5x5, Depth: 36, stride: 2x2, valid padding|
| RELU					|												|
| Convolution       | Filter: 5x5, Depth: 48, stride: 2x2, valid padding|
| RELU					|												|
| Convolution       | Filter: 3x3, Depth: 64, stride: 1x1, valid padding|
| RELU					|												|
| Convolution     | Filter: 3x3, Depth: 64, stride: 1x1, valid padding|
| RELU					|												|
| Flatten				|												|
| Fully connected		|Output: 100									|
| RELU					|												|
| Dropout	         	|0.3           									|
| Fully connected		|Output: 50									|
| RELU					|												|
| Dropout	         	|0.3           									|
| Fully connected		|Output: 10									|
| RELU					|												|
| Fully connected		|Output: 1									|
| Mean Squared Error	|        									|


The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a network based on a known model.

My first step was to create a really simple model, so I could test the ability of the model controls the car. Than I tryed the model created in the classification images project, than finnaly I used a convolution neural network model similar to the NVidia pipeline. I thought this model might be appropriate because it was tested in a real car.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model including dropout layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I increased the number of batches in trainning process. I also colect more data, drinving the car on oposite direction.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from getting of the center of the lane.

![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would train the model to better generalizate. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]


After the collection process, I had 20.000 data points. I then preprocess incoming data, centering around zero with small standard deviation, and trimming the image to only see section with road.

I finally randomly shuffled the data set and put 2% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 32 as evidenced by the validation loss, that not reduces, increasing the number of epochs.

I used an adam optimizer so that manually training the learning rate wasn't necessary.

