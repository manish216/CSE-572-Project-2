Project-2             
Handwriting comparison using Linear Regression, Logistic Regression and  Neural Network.
================
Manish Reddy Challamala,
November 2, 2018 ,manishre@buffalo.edu

For detailed explaination, please visit the below link:
[link for report pdf](https://github.com/manish216/Project-2/blob/master/proj2.pdf)

## Abstract
To train a model to compare the hand written samples a known and
questioned writer using Linear regression, Logistic regression and Neural
network.

## 1 Introduction

The goal of this project is to find similarities between hand written copies of
two different writers by using CEDAR as data source. Where each instance
in the CEDAR “AND” training data consists of set of input features for each
handwritten “AND” sample.
The features are obtained from two different sources:

1. Human Observed features: Features entered by human document exam-
    iners manually
2. GSC features: Features extracted using Gradient Structural Concavity
    (GSC) algorithm.

The above feature data sets are trained by using following methods:

1. Linear Regression using Stochastic Gradient Descent.
2. Logistic Regression using Stochastic Gradient descent.
3. Neural networks.


## 2 Theory

### 2.1 Linear Regression Model using Stochastic Gradient Descent

Linear Regression is a technique to analyze the linear relationship between the
independent(x) and dependent(y) variables.Here x is input data and y is pre-
dicted values.

```
BasicEquation:y= (a 0 +a 1 )x

Herea 0 and a 1 are weights
The polynomial equation for linear regression is given by

t=w 0 x 0 +w 1 x 1 +..+wmxm=w 0 + ∑m
i=
wixi=wTx (1)
t=WTφ(x)
```
where;
t is vector of predicted outputs
W =(w 0 ,w 1 ,...,w(M−1)) is weight vector
The w weight vector (vector of coefficients): which gives the polynomial rela-
tionship between the output and the input and represents the y- axis intercept
of the model.

Here the optimal weights are found by using stochastic gradient descent:

2.1.1 Stochastic gradient descent:

1. In stochastic Gradient Descent [SGD], We initialize the random weights.
2. SGD takes the derivative of each training data instance and calculates the
    update weights immediately iteratively.
Design MatrixThe design matrix is a n-by-p dimension matrix, where
n = no of samples and p is number of features.
Basis Function For calculating a basis function we are using the prob-
ability density function.

2.1.2 Regularization
(a) By using the Feature Expansion we can fit the data even better by
using higher degree polynomials, but there is a problem that we may
over fit the data.
(b) So by over-fit of data, The model works well for training data but for
a unseen data the model fails to predict the output.
(c) To over come this effect we are using regularization.
(d) Regularization method explicitly takes care of curve not to over-fit
the data by adding error to the model.


## 3 Logistic Regression Model with stochastic gradient descent
The logistic regression model is a classification model with linear regression
algorithm which tries to predict y for a given x.

1. The logistic regression gives the probability to which output each input
    belongs.
2. To generate probability for each input logistic regression takes a function
    that maps the output between the range of 0 and 1 for all values of input
    x.
3. The hypothesis for logistic regression is used
4. The above equation is called the logistic function or the sigmoid function.
5. we calculate the loss function by taking the derivative of logistic function.
6. we calculate the maximum likelihood of the data by minimizing the loss
    function either by increasing or decreasing the weights that best fit for
    our data.we can achieve this by taking the partial derivative of loss func-
    tion.Which is nothing but using the SGD and update the weights itera-
    tively.
    
## 4 Neural Network

The neural network model is a classification model which tries to predict output
y for a given input x

1. The neural network model contain two phase:
    1 Learning Phase
    2. Prediction Phase
2. In learning phase, the neural network takes inputs and corresponding out-
    puts.
3. The neural network process the data and calculates the optimal weights
    to gain the training experience.
4. In prediction process the neural network is presented with the unseen data
    where it predicts a output by using its past training experience on that
    given data.

## 5 Experimental Setup:

The experimental setup consists of three steps:

1. Data Pre-processing
2. Linear Regression using Stochastic gradient descent
3. Logistic regression using stochastic gradient descent
4. Neural Network

### 5.0.1 Data Pre-Processing:

1. In data pre-processing, we have 2 data sets: 1. Human observed dataset
    and 2.GSC observed dataset.
2. Firstly, we are trying to generate 4 feature sets from these 2 data sets one
    is by concatenating the features of two writers and other is by subtracting
    the features two writers.
3. Total we will be having 4 feature sets generated from the 2 given data sets,
    Where we split the raw data in accordance to our program requirement.
4. we are splitting 80% of raw data has training data set , In remaining 20%
    we are splitting 10% for validation data and another 10% for testing data
    set from all the 4 feature sets.
5. In this process, we are reading the raw data from a file ’HumanObserved-
    Features-Data.csv file and the target data from same-pairs.csv and diff-
    pairs.csv file.


6. we are combining the target values from the same-pair and diff-pair file
    and merging the feature data accordingly to the image id of the writers
    from humanObserverd-Feature-Data file. which forms our human concate-
    nated dataset.On the other hand, we subtract the features of two writers
    respectively, which gives us the Human subtracted data set. The same
    process is applied for GSC dataset.
7. Dimensions for each Dataset is given Below:
    where rows = no of samples and columns = no of features.
    splitting the raw data: 80% for training data, 10% for validation data,
    10% for testing data.


### (a) Human Observed Concatenated Dataset

1. Dimension of Data frame after Merging: 1582 X 21 [Merged 791
from same pair and 791 from diffpair file]
2. Feature Matrix: 1582 X 18 [removed image id in this step]
3. Target Vector : 1582 X 1
4. Training Feature Matrix: 1266 X 18
5. Validation Feature Matrix: 158 X 18
6. Testing Feature matrix: 156 X 18

### (b) Human Observed subtracted Dataset

1. Dimension of Data frame after Merging: 1582 X 21 [Merged 791
from same pair and 791 from diffpair file]
2. Data frame after subtracting the features: 1582 X 12
3. Feature Matrix: 1582 X 9 [removed image id in this step]
4.Target Vector : 1582 X 1
5. Training Feature Matrix: 1266 X 9
6. Validation Feature Matrix: 158 X 9
7. Testing Feature matrix: 156 X 9

### (c) GSC Observed Concatenated Dataset

1. Dimension of Data frame after Merging: 1024 X 1027 [Merged 791
from same pair and 791 from diffpair file]
2. Feature Matrix: 1024 X 1024 [removed image id in this step]
3. Target Vector : 1024 X 1
4. Training Feature Matrix: 820 X 1024
5. Validation Feature Matrix: 102 X 1024
6. Testing Feature matrix: 100 X 1024

### (d) GSC Observed subtracted Dataset

1. Dimension of Data frame after Merging: 1024 X 1027 [Merged 791
from same pair and 791 from diffpair file]
2. Data frame after subtracting the features: 1024 X 512
3. Feature Matrix: 1024 X 512 [removed image id in this step]
4.Target Vector : 1024 X 1
5. Training Feature Matrix: 820 X 512
6. Validation Feature Matrix: 102 X 512
7. Testing Feature matrix: 100 X 512


### 5.0.2 Linear regression with stochastic gradient descent:

1. In linear regression we are clustering the data using the k-means clustering
    library and finding 15 means.
2. By getting the mean for each cluster we are substituting it in the proba-
    bility basis function formula equation[10] to generate a scalar value for a
    feature vector accordingly.
3. we create the phi matrix for training, validation and testing whose dimen-
    sions are displayed above for each data set respectively.
4. we are initializing the random weights and iterating the values to converge
    it to the optimum weights by using Stochastic gradient descent.
5. At each iteration, we need to calculate the error function, where the error
    function value tells us in which direction we need to converge.

### 5.0.3 Logistic Regression using Stochastic Gradient Descent :

1. In logistic Regression, initialize the random weights.
2. Calculating the optimal weights by plugging the input features and the
    transpose of weight matrix to the sigmoid function equation[17]
3. By using SGD, we are calculating the loss function at each iteration and
    updating weights by using equations [19 and 20].
4. After, getting the optimized weights we are predicting the outputs for the
    new unseen data and compare the predicted output with the actual output
    which gives our accuracy.

### 5.0.4 Neural network:

1. Creating a model with 3 layers, 1. input layer 2. hidden layer 3. output
    layer
2. No of nodes for each layer is given below:
    1.No of nodes in input layer = No of features in the data set
    2. No of nodes in hidden layer 256
    3.No of nodes in output layer 2 [because we are classifying it into only two
    classes]
3. Activation functions used in hidden layer is relu [rectified linear unit] be-
    cause it introduces the non linearity in the network and softmax function
    is used on the output layer to predict the target class.
4. By using the above parameters we create a model.
5. we run this model by plugging in the appropriate data to it and train the
    model.
6. After training the model we test the model using the unseen data to predict
    that which class output belongs.


## 6 Experimental Results:

<img src="https://github.com/manish216/Project-2/blob/master/output/Result.jpg" width=900/>

<p style="text-align: center;">Table[1] Experimental Results</p>                                   

### 6.0.1 Graphs:

  The graph results are available here:
  [Results](https://github.com/manish216/Project-2/blob/master/proj2.pdf)





