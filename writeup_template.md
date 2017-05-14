# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Here are the links to the videos

#### Track 1
[![Track 1 video](https://img.youtube.com/vi/5VmdRuLCJSU/0.jpg)](https://www.youtube.com/watch?v=5VmdRuLCJSU)


#### Track 2
[![Track 2 video](https://img.youtube.com/vi/P-dPNZ0L9t4/0.jpg)](https://www.youtube.com/watch?v=P-dPNZ0L9t4)

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[center_ex]: ./imgs/center_ex.jpg "center Image"
[recovery_1]: ./imgs/recovery1.jpg "Recovery Image"
[recovery_2]: ./imgs/recovery2.jpg "Recovery Image"
[recovery_3]: ./imgs/recovery3.jpg "Recovery Image"
[flipped_recovery_1]: ./imgs/flipped_recovery1.jpg "flipped Recovery Image"
[flipped_recovery_2]: ./imgs/flipped_recovery2.jpg "flipped Recovery Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 47-56) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 49). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting with probabilty 0.3. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving data from 2 tracks.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with the nVidia model of 5 convolutional layers followed by 4 layers of dense networks.

I thought this model might be appropriate because it was already tested on a real world as mentioned in the paper and all I should be ideally doing is transfer learning

Initially,  I drove for the first track for only once. I can see that the model is underfitting. So I increased the data set to drive around the track twice and then in the mountainues track twice. This along with the right and left camera images gave me enough training data 

But still the car was not able to drive when the terrain beside the road changes. This made me think that I should convert the image to gray scale.

To combat the overfitting, I modified the model to add a Dropout layer of 0.3 probablity.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. Especially at the sharp turns. So I trained the model where the vehicle starts at the end of the road and comes towards the center.  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. And this worked on track 2 too!! :D 

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes

1. Input - 160 x 320 x 3
2. Preprocessing - normalization and cropping
3. convolution with 5x5 kernal and output 24 layers
4. convolution with 5x5 kernal and output 36 layers
5. convolution with 5x5 kernal and output 48 layers
6. convolution with 3x3 kernal and output 64 layers
7. convolution with 3x3 kernal and output 64 layers
8. Flatten
9. Fully connected layer - output 100
10. Fully connected - output 50 
11. Fully connected - output 10
12. Fully connected - output 1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

`*0 degrees*`

![alt text][center_ex]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

`*+13.5 degrees* `

![alt text][recovery_1] 

`*+18 degrees*`

![alt text][recovery_2]

`*22.5 degrees*`

![alt text][recovery_3]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles otherwise the model seeing only left curves more will tend to be more towards left For example, here is an image that has then been flipped for the same recovery image above

![alt text][flipped_recovery_1]
![alt text][flipped_recovery_2]


After the collection process, I had 29869 data points. I then preprocessed this data by converting it to gray scale.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by reducing training and validation loss. Increasing the epochs to 4 increased the validation loss which implied over fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

**A simple gray scale conversion with careful collection of training data worked well on both the tracks. Track 1 at speed 30 and track 2 at speed 12**
