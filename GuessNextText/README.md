# Probabilistic states and transitions using Hidden Markov Model

## Predict Next Text

#Project Description
The project tries to evaluate the possible and most probable "next word" for a given pair of words, which is later
extended to generate new random text.

# Input data and Output Information

This project takes raw data from Shakespeare scripts and generates output predicting the probable next word for a given
pair of words.

# Source of Raw/Processed Data

Raw data has been taken from :
https://www.kaggle.com/kingburrito666/shakespeare-plays?select=alllines.txt
 

# File manipulations
Initially consumed file had issues with few lines without '"', I had to trim those lines out. 

# Steps followed in the project

1. Load the engineered input file into dataframes.
2. Engineering the text data
3. Algorithm for frequency of pair of consecutive words.
4. Algorithm for frequency of pair of alternate words.
5. Probability calculations for pair of words together.
6. Algorithm for nextTextPredictor
7. Algorithm for new random text generator
 
# Reports
No reports have been generated for this project..

# Observations

## While Feature Engineering the input

Ref: https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html

I tried using Stemming to make sure I avoid similar words, but I was also losing some correct content, like, "palace" 
was returning as "palac" after the stemming process, hence, I'm not stemming the given text.

## While Running the algorithm for text generation

At times, due to highly randomized vocabulary selection, there is an issue where the randomly selected word is not found
in the frequency sets which results in "keyError". This can be a possibility because this algorithm considers only words 
till the ante-penultimate count in each line and also vocabulary is considered by eliminating stop words and hence we might
end up in the above error. I'm handling such case, by selecting any other word from vocabulary in random.

Often, I got correct words as the possible NEXT WORD.
At times, it fails to get the exact word, when there are a lot of possibilities with close probabilities 
with different word combinations.