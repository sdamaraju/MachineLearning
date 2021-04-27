# Project - 3 - Says One Neuron to another

#Project Description
The project implements neural network algorithm using python and applies the algorithm on 2 datasets.
We also evaluate the accuracy performance of the algorithm and share few observations out of the implementation.

# Source of Raw/Processed Data

Raw data has been taken from :
https://www.kaggle.com/brsdincer/star-type-classification
and 
https://www.kaggle.com/uciml/iris 

#Logical implementation details
Steps in the implementation: Note that all the explanations are provided inline with the implementation.

1. Define all the relevant and required neural network implementations.
2. Define all the required prediction realted functions.    
3. Apply the implemented neural networks into two datasets taken.
Datasets taken :

Stars.csv : Has features like spectral class, color and A_M to determine the type of start it is.
Iris.csv : has petal and sepal - width and length, and helps determine the speicies.

# Steps followed in the algorithm implementation

1. Identify and set initial random weights across all layers 
2. Define the activation functions, sigmoid during feedForward and sigmoidDerivative during backpropagation, our case.        
3. Define the feedForward logic.
4. Define backPropagation Logic
5. Build The actual Neural network model to identify perfect weights for our implementation using the above defined methods.
6. Define accuracy and prediction functions.
 
# Reports

## STAR DATASET ACCURACY DETAILS
================================
Training Accuracy :0.765625
Training Accuracy :0.7916666666666666
Training Accuracy :0.8020833333333334
Training Accuracy :0.8020833333333334
Iterations : 200

Finally Testing Accuracy on Validation Content of Stars Dataset with iterations 200: 0.8541666666666666

Iterations : 500
Training Accuracy :0.78125
Training Accuracy :0.9635416666666666
Training Accuracy :0.8020833333333334
Training Accuracy :0.7552083333333334
Training Accuracy :0.8020833333333334
Training Accuracy :0.7916666666666666
Training Accuracy :0.8125
Training Accuracy :0.8125
Training Accuracy :0.8125
Training Accuracy :0.8072916666666666

 Finally Testing Accuracy on Validation Content of Stars Dataset with iterations 500: 0.8541666666666666


## IRIS DATA SET Accuracy details : 
================================
Iterations : 200
Training Accuracy :0.8916666666666667
Training Accuracy :0.9666666666666667
Training Accuracy :0.9666666666666667
Training Accuracy :0.9166666666666666

 Finally Testing Accuracy on Validation Content of IRIS dataset with iterations 200: 0.9333333333333333

Iterations : 500
Training Accuracy :0.9
Training Accuracy :0.975
Training Accuracy :0.9416666666666667
Training Accuracy :0.975
Training Accuracy :0.9833333333333333
Training Accuracy :0.975
Training Accuracy :0.9916666666666667
Training Accuracy :0.975
Training Accuracy :0.9833333333333333
Training Accuracy :0.975

 Finally Testing Accuracy on Validation Content of IRIS dataset with iterations 500: 1.0

# Observations

On applying the implemented neural network algorithm 2 datasets here are some observations :

We can see a gradual improvement in the training accuracy on both data sets.
On intentional run of both the datasets for 200 iterations and 500 iterations, we can see that the training accuracy keeps improving as the iterations are increased and also the final validation accuracy seems to be better when there are more number of iterations.
I observed that sigmoid function doesn't run efficiently when the values in the given input are either too high or too low, and my accuracy was too low in that case, hence I had to drop few columns in the Stars dataset like, 'Temperature', 'L' and 'R' and use the other columns to achive the accuracy.
