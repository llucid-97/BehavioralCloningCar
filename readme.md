# **Behavioral Cloning** 

---

The goals / steps of this project are the following:
* Use the [simulator](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/1c9f7e68-3d2c-4313-9c8d-5a9ed42583dc) to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./__readmeFiles__/graph.png "Model Visualization"
[image2]: ./__readmeFiles__/center_2018_01_28_12_20_31_519.jpg "Grayscaling"
[image3]: ./__readmeFiles__/center_2018_01_28_18_31_17_085.jpg "Recovery Image"
[image4]: ./__readmeFiles__/center_2018_01_28_18_32_15_379.jpg "Recovery Image"
[image5]: ./__readmeFiles__/center_2018_01_28_18_32_58_618.jpg "Recovery Image"
[image6]: ./__readmeFiles__/placeholder_small.png "Normal Image"
[image7]: ./__readmeFiles__/placeholder_small.png "Flipped Image"

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

My model is a convolution neural network which consists:
 * A preparation section which:
 
        > Converts to the HSV color domain
        > Crops the image down to exclude the hood/bonnet and anything above the horizon
        > Performs batch normalization to approximate whitening
   
 * A Fully-Convolutional Section consisting:
 
        > 1 Conv2D layer with 32 (3x3) Filters and a stride of 2
        > 1 Separable Conv2D layer with 64 (3x3) "Filters" (no stride, no depth multiplier)
        > Relu Activations for all Conv layers to introduce nonlinearity
     
 * Fully-Connected Back end with:
        
        > A Flattening layer
        > A final layer connecting to 1 output neuron

 

#### 2. Attempts to reduce overfitting in the model

The model uses dropout on the output of the convolutional section in order to reduce overfitting (model.py). 

The model was trained and validated on different data sets to ensure that the model was not overfitting.
The model was revised to maximise accuracy in both, and early stopping was used when it began to overfit.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an RMSProp optimiser with a learning rate of 0.001 and rho=0.9

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used:
 * A single lap of center lane driving in both directions on the track
 * Recovering from the left and right sides of the road 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from a very simple model, increase its complexity based on established vision architectures until it began to overfit, then introduce regularizers/dropout or reduce complexity until an optimal point

My first step was to use a single neuron fully connected to every pixel, then build the front end based on the MobileNet architecture.
I chose that architecture as it attains state of art performance comparable to VGG-16 and Inception V2 using a fraction of the parameters.
  

In order to gauge how well the model was working, I recorded a validation data set which consisted mostly center-following behavior, but made the training dataset saturated with error-correcting behavior.
This made it much easier to see when the model was overfitting.
If I included any more layers from MobileNet or reduced dropout, the validation data took huge plunges closer to the end of training 

This is why the model only uses the first 2 Separable convolutional layers, as opposed to the 12 used in arXiv:1704.04861v1


The final step was to run the simulator to see how well the car was driving around track one.
Models that overfit "locked up" instantly, while models that didn't went further.
More error correction training data, model simplification, and regularization fixed this


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:


| Layer (type)       |          Output Shape        |      NumParams  | 
|:---------------------:|:---------------------------------------------:|
| input_1 (InputLayer)    |     (None, 160, 320, 3)   |    0 |         
| RGB_to_HSV (Lambda)     |     (None, 160, 320, 3)    |   0  |       
| crop (Cropping2D)       |     (None, 90, 320, 3)    |    0   |      
| BatchNorm_1 (BatchNorma) | (None, 90, 320, 3)      |  12      |  
| Feat_Conv_1 (Conv2D)    |     (None, 44, 159, 32)    |   896 |       
| batch_normalization_1 (BatchNorm) | (None, 44, 159, 32)   |    128 |       
| Feat_Sep_1 (SeparableConv2D) | (None, 44, 159, 64)   |    2400      |
| batch_normalization_2 (BatchNorm) | (None, 44, 159, 64)    |   256   |    
| flatten_1 (Flatten)         | (None, 447744)         |   0         |
| dropout_1 (Dropout)        |  (None, 447744)        |    0         |
| Raw_Output (Dense)         |  (None, 1)              |   447745    |

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn this behavior so it corrects offsets from center.

These images show what a recovery looks like.
From a position on the left or right as so:

![alt text][image3]
![alt text][image4]

We show examples of driving back to center:

![alt text][image5]

To augment the dataset, I also:
    
    > flipped images and negated the angles 
        This generates more data and ensures there's no bias in going one way vs the other
    > Used offsets of the initial camera position and added offsets to angles
        This generates more data and contributes to the error correction examples



I finally randomly shuffled the data set and put 40% of the data into a validation set. 

I used this training data for training the model. 
The validation set helped determine if the model was over or under fitting.
The ideal number of epochs was 4 as evidenced by the validation loss rising after this point despite training loss still falling
I used an RMSProp Optimizer so that manually training the learning rate wasn't necessary. (but I did start at 0.001 with a  rho of 0.9)
