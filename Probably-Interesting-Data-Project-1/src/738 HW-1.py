#!/usr/bin/env python
# coding: utf-8

# # Project - 1 Goal : Mixture Models Distributions

# ## Steps in Project
# 
# ### a. Identify and understand the datasets.
# ### b. Load the datasets in to dataframes.
# ### c. Plot the distributions prior to applying our algorithm, visualize the plots to understand what can be our possible "k" values for the initial clusters count and also get approximate the mean and variance values.
# ### d. Pass the attribute values, mean and variance to the algorithm and run the expectation-maximization algorithm that we tried to implement.
# ### e. Replot the distributions to get a Gaussian distribution and share the observations.
# 

# In[1]:


## Importing the required datasets.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from math import sqrt


# ### The probability density below is defined in the “standardized” form. To shift and/or scale the distribution use the loc and scale parameters. Specifically, norm.pdf(x, loc, scale) is identically equivalent to norm.pdf(y) / scale with y = (x - loc) / scale
# 
# Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html

# In[2]:


def probabilityDensityFunction(data, mean, sigma):
    arrayfiedData = np.asarray(data)
    output = norm.pdf(arrayfiedData, mean, sigma)
    return output


# ### Convergence test for EM, to evaluate when to stop the EM model of evaluation. 
# ### The logic here compares if the values of means have a difference less than convergenceThreshold to the values of previousMean.
# ### If its less than convergenceThreshold then this method returns true, else false.

# In[3]:


def isConverged(mean, prevMean, threshold):
    k = len(mean)
    meanDelta = []

    for i in range(k):
        meanDelta.append(abs(mean[i]-prevMean[i]))

    count = 0
    for i in range(k):
        if(meanDelta[i] < threshold):
            count = count + 1
    # we need the delta of means for all the clusters to be less than 0.01
    if(count == k):
        return True
    else:
        return False


# ### Expectation-Maximization Implementation 
# #### Please see inline comments to go through the logic details.
# 
# #### Ref : https://www.geeksforgeeks.org/gaussian-mixture-model/ 

# In[4]:


def runEM(data, params, context, thresholdForConvergence):
    # initialize values of means, variances, and mixture coefficients of gaussian models
    k = params['gaussians']
    mean = params['mean']
    prevMean = []
    sigma = params['sigma']
    prevSigma = []
    probK = []
    prevProbK = []
    latentValues = []
    gaussians = []
    
    #intialize the initial previous values of mean and sigma to some randome number so that the first while condition 
    #for the convergence test doesn't fail
    
    ## also initalize the initial probobaility for each cluster which will be 1/(no. of clusters)
    
    ## suppose all the latent values to 0s.
    
    for i in range(k):
        prevMean.append(-9999)
        prevSigma.append(-9999)
        probK.append(1/k) 
        latentValues.append([0] * len(data))
     
    # running the initial probability density function on the data with each cluster's mean 
    # and sigma that we initially approximated
    for i in range(k):
        gaussians.append(probabilityDensityFunction(data, mean[i], sigma[i]))
   
    iterations = 0
    # here I'm evaluating to see if the convergence occurs within 25 iterations, 
    # if yes, then the loop breaks immediately, else it breaks after 25 iterations at maximum

    while(not isConverged(mean, prevMean, thresholdForConvergence) and iterations < 25):
        # Expectation :step where we identify the latent or hidden values
        for i in range(k):
            for j in range(len(data)):
                sumLatentVars = 0
                for l in range(k):
                    sumLatentVars = sumLatentVars + probK[l]*gaussians[l][j]
                latentValues[i][j] = probK[i]*gaussians[i][j]/sumLatentVars
                
        # Back up the mean, sigma and probability values so that we can use them for 
        prevMean = mean.copy()
        prevSigma = sigma.copy()
        prevProbK = probK.copy()
        
        # Maximization : step where we re-validate the mean, sigma and probability of the clusters using the latent variables.    
        # update the values of the means, variances, and mixing cofficient
        for i in range(k):
            tempmean = 0
            tempsigma = 0
            tempprob = 0
            for j in range(len(data)):
                tempmean = tempmean + (data[j] * latentValues[i][j])/sum(latentValues[i])
                tempsigma = tempsigma + ((data[j]-mean[i])**2 * latentValues[i][j])/sum(latentValues[i])
                tempprob = tempprob + latentValues[i][j]/len(data)
            mean[i] = tempmean
            sigma[i] = sqrt(tempsigma)
            probK[i] = tempprob
        
        # Update the gaussians here with the new mean and sigma identified from the Maximization step.
        for i in range(k):
            gaussians[i] = probabilityDensityFunction(data, mean[i] ,sigma[i])
        
        # increase iteration
        iterations = iterations + 1
    
    # Note down the iterations it has taken.
    print("Converged after", iterations, "iterations.")
    
    ## plotting the gaussian distributions, post the convergence.
    temp = []
    
    for i in range(len(data)):
        tempprob = 0
        for j in range(k):
            tempprob = tempprob + probK[j]*gaussians[j][i]
        temp.append(tempprob)
    
    output = list(zip(data,temp))
    output.sort()
    x2, y2 = zip(*output)
    # plot subplots
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,3))
    sns.distplot(data, ax=ax1)
    ax1.set_title(" Probability Distribution")
    ax2.plot(x2,y2)
    ax2.set_title(" Gaussian mixture model with EM")


# # Analysis on DataSet - 1 : Wine-Quality

# ### Read the data from files to data frames.

# In[5]:


## Load the datasets into dataframes.
wineData = pd.read_csv("/Probably-Interesting-Data-Project-1/data/winequality-red.csv")
wineData.head(10)


# ### Evaluate from the above dataset to identify two such features which are visually distributed along more than one cluster and picking those features so that I can get multiple clusters of data where there is a possible overlap of data points and that is where the Gaussian mixture models can be applied.
# 
# ####  Based on observation, considering "citric_acid" and "volatile acidity"

# In[6]:


citricAcid = wineData['citric acid']
volatileAcidity = wineData['volatile acidity']


# # Citric Acid Univariate analysis.

# ### Initial distribution plotting using the seaborn library.

# In[7]:


sns.distplot(citricAcid)


# ### Citric Acid- GMM - EM algorithm call.
# 
# ### From the plot above, I can see that there are three possible gaussians into which the data points can be classified into. 
# ### We can also observe that their possible means can be 0.05,0.25 and 0.5. So keeping them as the initial means.
# ### A rough estimation of initial sigma values being taken are calculated as the squares of difference between the mean and the midpoint of the x axis region plotted by histogram

# In[8]:


initialParams = {'gaussians':3, 'mean':[0.05, 0.25, 0.55], 'sigma':[0.12,0.02,0.02]}
runEM(citricAcid, initialParams, "Citric acid", 0.005)


# # Volatile Acidity Univariate analysis.

# ### Initial distribution plotting using the seaborn library.

# In[9]:


sns.distplot(volatileAcidity)


# ### Volatile Acidity- GMM - EM algorithm call.
# 
# ### From the plot above, I can see that there are two possible gaussians into which the data points can be classified into. 
# ### We can also observe that their possible means can be 0.375,0.675. So keeping them as the initial means.
# 
# ### A rough estimation of initial sigma values being taken are calculated as the squares of difference between the mean and the midpoint of the x axis region plotted by histogram

# In[10]:


initialParams = {'gaussians':2, 'mean':[0.375,0.675], 'sigma':[0.016,0.031]}
runEM(volatileAcidity, initialParams, "Volatile acidity", 0.005)


# # Analysis on DataSet - 2 : PowerConsumption Dataset

# In[11]:


## Load the datasets into dataframes.
powerConsumption = pd.read_csv("/Probably-Interesting-Data-Project-1/data/house_hold_power_consumption.csv")
powerConsumption.head(10)


# ### Evaluate from the above dataset to identify two such features which are visually distributed along more than one cluster and picking those features so that I can get multiple clusters of data where there is a possible overlap of data points and that is where the Gaussian mixture models can be applied.
# 
# ####  Based on observation, considering "citric_acid" and "volatile acidity"

# In[12]:


globalIntensity = powerConsumption['Global_intensity']
globalReactivepower = powerConsumption['Global_reactive_power']


# # Global Intensity Univariate analysis.

# ### Initial distribution plotting using the seaborn library.

# In[13]:


sns.distplot(globalIntensity)


# ### Global Intensity- GMM - EM algorithm call.
# 
# ### From the plot above, I can see that there are three possible gaussians into which the data points can be classified into. 
# ### We can also observe that their possible means can be 1.5,7.5 and 14. So keeping them as the initial means.
# 
# ### Evaluating the initial sigma values to be possible midpoint of the x axis where the histogram is plotted, i.e 10.

# In[14]:


initialParams = {'gaussians':3, 'mean':[1.5, 7.5, 14], 'sigma':[10,10,10]}
runEM(globalIntensity, initialParams, "Global Intensity", 0.1)


# # Global Reactive Power Univariate analysis.
# 

# ### Initial distribution plotting using the seaborn library.

# In[15]:


sns.distplot(globalReactivepower)


# ### Global Reactive Power- GMM - EM algorithm call.
# 
# ### From the plot above, I can see that there are three possible gaussians into which the data points can be classified into but I would like to consider just 2 clusters this time, to see if a business logic requirement like, segmentation into either high reactive power or low reactive power is needed, then though the picture says 3 clusters, the business actually needs just 2 clusters of classified data.
# 
# ### We can also observe that their possible means can be 0.1,0.18, So keeping them as the initial means.
# 
# ### Evaluating the initial sigma values to be possible midpoint of the x axis where the histogram is plotted, i.e 0.25 .

# In[16]:


initialParams = {'gaussians':2, 'mean':[0.1,0.18], 'sigma':[0.25,0.25]}
runEM(globalReactivepower, initialParams, "Global reactive Power", 0.005)


# ### Observations
# 
# The gaussian mixture model with EM shows a great distribution of the individual attributes of the wine quality and power consumption dataset.
# 
# Initially when I considered random initial values, I got very strange graphs as output but, once I evaluated the mean from the initial plots and approximated sigma values helped me get better plots.
# 
# So, I feel that a lot depends upon the initial values being considered as well.
# 
# Also, I felt sometimes during this project that K, as in the clusters, should not be pictorially identified rather it has to be identified more by business scenario requirement. 
# 
# Upon multiple runs, 
# I identified that the convergence condition sometimes doesn't meet so I added a max iteration count.
# That's because attribute value ranges decide the mean value difference and the threshold for the delta of the means will not be a constant value for different attributes or datasets, hence I made that a function param to runEM and passed the threshold value to the method.
# 
