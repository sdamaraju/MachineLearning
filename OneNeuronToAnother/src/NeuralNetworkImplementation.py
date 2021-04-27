#!/usr/bin/env python
# coding: utf-8

# ## Project-3 : Says One neuron to another
# ### Goal of the project : Implement a neural network model from the scratch.
# 
# This notebook consists a naive neural network implementation that uses the following references for its implementation.
# https://medium.com/fintechexplained/neural-networks-bias-and-weights-10b53e6285da
# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

# ### Logical implementation details
# Steps in the implementation:
#     Note that all the explanations are provided inline with the implementation.
#     
#     1. Define all the relevant and required neural network implementations.
#     2. Define all the required prediction realted functions.    
#     3. Apply the implemented neural networks into two datasets taken.
# 
# Datasets taken : 
# 1. Stars.csv : Has features like spectral class, color and A_M to determine the type of start it is.
# 2. Iris.csv : has petal and sepal - width and length, and helps determine the speicies.

# ### Required Library imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# ## Define all Neural Network related functions

# In[2]:


# 1. Identify and set initial random weights across all layers 
def setInitialWeights(layerDetails):
    
    # iterate over each hidden layer and generate random weights between -1 and 1 at each layer.
    weights=[]
    for i in range(1, len(layerDetails)):
        layerLevelWeights = [[np.random.uniform(-1, 1) for k in range(layerDetails[i-1] + 1)] for j in range(layerDetails[i])]
        weights.append(np.matrix(layerLevelWeights))
    return weights
        
        
#2. Define the activation functions, sigmoid during feedForward and sigmoidDerivative during backpropagation, our case.        
def sigmoid(value):
    return 1 / (1 + np.exp(-value))

def sigmoidDerivative(value):
    return np.multiply(value, 1-value)   


## The right values for the weights and biases determines the strength of the predictions. 
## The process of fine-tuning the weights and biases from the input data is known as training the Neural Network.
## Each iteration of the training process consists of the following steps:
## Calculating the predicted output ŷ, known as feedforward
## Updating the weights and biases, known as backpropagation



#3. Define the feedForward logic.
# Apply sigmaoid (activation function) on the weights of the hidden layer multiplied by the input data and add bias for input (1). 
# Run this in loop of layers to apply the sigmoid in a loop.
# As explained above, feedForward, uses weights and generates outputs.

def feedForward(x, weights, layers):
    output, eachInput = [x], x
    for eachLayer in range(layers):
        eachInput = eachInput.astype(float) 
        tempOP = sigmoid(np.dot(eachInput, weights[eachLayer].T))
        output.append(tempOP)
        # next add the bias which in our case is assumed to be 1.
        eachInput = np.append(1, tempOP) 
    return output

# 4. propagateBackward
## After the feedForward step, we need to update the weights and manipulate the biases 
## by comparing with Y_train (actual outputs of train data and outputs we generated with random weights)
## and then identify the gradient loss

def propagateBackward(y, output, weights, layers, learningRate):
    outputFinal = output[-1]
    difference = np.matrix(y - outputFinal)
    
    #Back propagate the difference at eachLayer
    # go back from top layer to 0.
    for eachLayer in range(layers, 0, -1):
        currOutput = output[eachLayer]
        
        if(eachLayer > 1):
            # Add previous output
            prevOutput = np.append(1, output[eachLayer-1])
        else:
            prevOutput = output[0]
        
        gradient = np.multiply(difference, sigmoidDerivative(currOutput))
        # next step weights are updated.
        temp = np.multiply(gradient.T, prevOutput)
        temp.astype(float)
        weights[eachLayer-1] += learningRate * temp
        wt = np.delete(weights[eachLayer-1], [0], axis=1) # Remove bias from weights
        difference = np.dot(gradient, wt) # Calculate error for current layer
    return weights
        
#5. The actual Neural network model to identify perfect weights for our implementation using the above defined methods.

def model(X_train, Y_train, learningRate, layerDetails, runs):

    # Random values are to be set to the weights for the first time at all layers.

    weights = setInitialWeights(layerDetails)

    for eachRun in range(1, runs+1):
        layers = len(weights)
        for i in range(len(X_train)):
            x, y = X_train[i], Y_train[i]
            x = np.matrix(np.append(1, x))         
            output = feedForward(x, weights, layers)
            weights = propagateBackward(y, output, weights, layers, learningRate)
            
        if(eachRun % 50 == 0):
            print("Training Accuracy :{}".format(accuracy(X_train, Y_train, weights)))
    return weights;        


# ## Define prediction and accuracy related functions

# In[3]:


# Plain accuracy function that takes the test content and the identified weights, to predict the possible value
# and then compare with actual result and calculate the accuracy score based on that.
def accuracy(X, Y, idealWeights):
    totalMatched = 0

    for i in range(len(X)):
        x = X[i]
        y = list(Y[i])
        prediction = predict(x, idealWeights)
        if(y == prediction):
            totalMatched = totalMatched + 1

    return totalMatched / len(X)

# the predict function simply calls the feedForward algorithm, this time with the weights that are calculated from the model and 
# the output we get will have a value for each possible class of the given input.
# the higher the value, more probable class it is.
# we initialize all values to 0 in the return list but the index with maxValue and set that to 1.
def predict(content, weights):
    layersCount = len(weights)
    content = np.append(1, content)
    # Call feedForward
    output = feedForward(content, weights, layersCount)
    outputFinal = output[-1].A1
    index = identifyIndexWithMaxValue(outputFinal)
    y = [0 for i in range(len(outputFinal))]
    y[index] = 1
    return y

# as explained above, identifyIndexWithMaxValue returns the index with max value.
def identifyIndexWithMaxValue(output):
    maxValue = output[0] #set to first value of output
    index = 0 # set to first index
    index = np.argmax(output)
    return index


# ### Apply neural networks to Dataset - 1 - Stars.csv
# 
# 

# In[4]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False)

dataset = pd.read_csv("/OneNeuronToAnother/data/Stars.csv")
dataset = dataset.drop(columns=['L', 'R','Temperature'])

X = dataset[['A_M','Spectral_Class','Color']]
X = np.array(X)

Y = dataset.Type
Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)


# In[5]:


# Number of features - using A_M, Spectral_Class and Color features of Stars dataset.
features = len(X[0])
# Number of classes - 6
# Types 0-5
classes = 6
layers = [features, 5, 10, classes]
idealWeights=[]

idealWeights = model(X_train, Y_train, learningRate=0.15, layerDetails=layers, runs=200)
print("Iterations : 200")
print("\n Finally Testing Accuracy on Validation Content of Stars Dataset with iterations 200: {}".format(accuracy(X_test, Y_test, idealWeights)))

print("\nIterations : 500")
idealWeights = model(X_train, Y_train, learningRate=0.15, layerDetails=layers, runs=500)
print("\n Finally Testing Accuracy on Validation Content of Stars Dataset with iterations 500: {}".format(accuracy(X_test, Y_test, idealWeights)))


# ### Apply neural networks to Dataset - 2 - IRIS_Dataset.csv

# In[6]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False)

dataset = pd.read_csv("/OneNeuronToAnother/data/Iris.csv")
dataset = dataset.drop(columns=['Id'])

X = dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
X = np.array(X)

Y = dataset.Species
Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)


# In[7]:


# Number of features - using all of them in the iris dataset.
features = len(X[0])
# Number of classes - 3
# iris-setosa, iris-versicolor, iris-virginica
classes = 3
layers = [features, 5, 10, classes]
idealWeights=[]

print("Iterations : 200")
idealWeights = model(X_train, Y_train, learningRate=0.15, layerDetails=layers, runs=200)
print("\n Finally Testing Accuracy on Validation Content of IRIS dataset with iterations 200: {}".format(accuracy(X_test, Y_test, idealWeights)))

print("\nIterations : 500")
idealWeights = model(X_train, Y_train, learningRate=0.15, layerDetails=layers, runs=500)
print("\n Finally Testing Accuracy on Validation Content of IRIS dataset with iterations 500: {}".format(accuracy(X_test, Y_test, idealWeights)))


# ## Observations:
# 
# On applying the implemented neural network algorithm 2 datasets here are some observations : 
# 1. We can see a gradual improvement in the training accuracy on both data sets.
# 2. On intentional run of both the datasets for 200 iterations and 500 iterations, we can see that the training accuracy keeps improving as the iterations are increased and also the final validation accuracy seems to be better when there are more number of iterations.
# 3. I observed that sigmoid function doesn't run efficiently when the values in the given input are either too high or too low, and my accuracy was too low in that case, hence I had to drop few columns in the Stars dataset like, 'Temperature', 'L' and 'R' and use the other columns to achive the accuracy.
# 
# 
# 

# In[ ]:




