# Probably Interesting Dataset - Project 1

#Project Description
The project tries to evaluate the Gaussian Mixture Model distributions
leveraging the Expectation-Maximization algorithm

# Input data and Output Information

This project takes raw data related to wine quality and power consumption.  
Loads the data into data frames and then performs univariate distribution analysis
of the possible Gaussian Mixture models, which analyzes different attributes of a dataset.

# Source of Raw/Processed Data

Raw data has been taken from :
 ##### https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
 ##### https://www.kaggle.com/uciml/electric-power-consumption-data-set?select=household_power_consumption.txt
 

# File manipulations
Initial power consumption file was a ".txt" file.
Converted that to a csv file

# Steps followed in the project

a. Identify and understand the datasets.
b. Load the datasets in to dataframes.
c. Plot the distributions prior to applying our algorithm, visualize the plots to understand what can be our possible "k" values for the initial clusters count and also get approximate the mean and sigma values.
d. Pass the attribute values, mean, sigma and threshold to the algorithm and run the expectation-maximization algorithm that we tried to implement.
e. Replot the distributions to get a Gaussian distribution and share the observations.

Please Note : 
1. Both the data sets are handled in same notebook - 738 HW-1
2. Logic for E-M has inline explanations in form of comments.
3. There are few assumptions(understanding by multiple trials) considered during the execution.
 
# Reports
Reports are available under the reports folders.

# Observations

The gaussian mixture model with EM shows a great distribution of the individual attributes of the wine quality and power 
consumption dataset.

Initially when I considered random initial values, I got very strange graphs as output but, once I evaluated the mean 
from the initial plots and approximated sigma values helped me get better plots.

So, I feel that a lot depends upon the initial values being considered as well.

Also, I felt sometimes during this project that K, as in the clusters, should not be pictorially identified rather it
has to be identified more by business scenario requirement.

Upon multiple runs, I identified that the convergence condition sometimes doesn't meet so I added a max iteration count.
That's because attribute value ranges decide the mean value difference and the threshold for the delta of the means will not be a constant value for different attributes or datasets, hence I made that a function param to runEM and passed the threshold value to the method.