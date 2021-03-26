#!/usr/bin/env python
# coding: utf-8

# ## Project - 2 Goal : Probabilistic states and transitions using Hidden Markov Model
# # Predict Next Text

# ## Steps in Project
# 
# ### 1. Load the engineered input file into dataframes.
# ### 2. Engineering the text data
# ### 3. Algorithm for frequency of pair of consecutive words.
# ### 4. Algorithm for frequency of pair of alternate words.
# ### 5. Probability calculations for pair of words together.
# ### 6. Algorithm for nextTextPredictor
# ### 7. Algorithm for new random text generator

# In[1]:


## Importing the required datasets.

import pandas as pd
import numpy as np
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from decimal import Decimal
import random


# # Step-1 Loading data into dataframes

# In[2]:


## Load the engineered files into dataframes.
script = pd.read_csv("/GuessNextText/data/alllines_trimmed.txt")
script.head(100)


# # Step-2 Engineering the text data
# 
# 
# ### 1.  Stemming and Lemmatization
# 
# #### PorterStemmer algorithm for performing Stemming and Lemmatization process on the words.
# #### Stemming and Lemmatization evaluates relevant words and categorizes and converts them into a base word.
# #### Example : student, student's, students $\Rightarrow$ student
# #### What is Stemming and Lemmatization ? 
# 
# Ref: https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
# 
# ### Note that I tried using Stemming to make sure I avoid similar words, but I was also losing some correct content, like, "palace" was returning as "palac" after the stemming process, hence, I'm not stemming the given text.
# 
# ### 2. Remove and Replace Special Characters
# 
# #### Replace all special characters with empty string.
# 
# ### 3. Change to Lower Case.
# 
# #### Convert all words to lower case, irrespective of what the original case is.
# 
# ### 4. Remove unnecessary words
# 
# #### Using Stop Words library we filter and remove unnecessary general words/terms like, he, she, but, if etc.

# In[3]:


vocabulary = []
processedVocab = []
ps = PorterStemmer()
for i in range(0, len(script)):
    processedData = re.sub('[^a-zA-Z]', ' ', script['Script'][i])
    processedData = processedData.lower()
    processedData = processedData.split()
    processedData = [word for word in processedData if not word in stopwords.words('english')]
    processedVocab.append(processedData)
    for i in range(len(processedData)):
        word = processedData[i]
        if word not in vocabulary:
            vocabulary.append(word)


# In[4]:


# All the vocabulary and processed vocabulary identified.
print(len(vocabulary))
print(processedVocab[2])


# # Step-3 : Algorithm for frequency of pair of consecutive words.
# 
# ### Evaluate the frequency of every word and next word.
# #### Example : We calculate the frequency for [enter,king] occurance and [king,henry] occurance and so on.

# In[5]:


frequencyOfPairOfWords = {} # count how many times a word and its successor appear in the text.
for i in range(0, len(processedVocab)):
    wordsInLine = processedVocab[i]
    wordCount = len(wordsInLine);
    for j in range(wordCount):
        if j != wordCount - 1: # will not consider the last words in the line.
            word = wordsInLine[j]
            successor = wordsInLine[j+1]
            if word not in frequencyOfPairOfWords:
                frequencyOfPairOfWords[word]={successor : 1}
            else:
                if successor not in frequencyOfPairOfWords[word]:
                    frequencyOfPairOfWords[word][successor] = 1
                else: frequencyOfPairOfWords[word][successor] = frequencyOfPairOfWords[word][successor] + 1   


# In[6]:


print(frequencyOfPairOfWords['enter']['king'])
print(frequencyOfPairOfWords['king']['henry'])


# # Step-4 : Algorithm for frequency of pair of alternate words.
# 
# ### Evaluate the frequency of every word and next-next word. 
# #### Example : We calculate the frequency for [enter, henry] occurance and [king, lord] occurance and so on.

# In[7]:


frequencyOfFirstAndThirdWord = {} # count how many times a word and its 2 successors appear in the text.
for i in range(0, len(processedVocab)):
    wordsInLine = processedVocab[i]
    wordCount = len(wordsInLine);
    for j in range(wordCount):
        if j < wordCount - 2: # will not consider the last 2 words in the line because they don't form the word and second successor.
            word = wordsInLine[j]
            successor2 = wordsInLine[j+2]
            if word not in frequencyOfFirstAndThirdWord:
                frequencyOfFirstAndThirdWord[word] = {successor2: 1}
            else:
                if successor2 not in frequencyOfFirstAndThirdWord[word]:
                    frequencyOfFirstAndThirdWord[word][successor2] = 1
                else:
                    frequencyOfFirstAndThirdWord[word][successor2] = frequencyOfFirstAndThirdWord[word][successor2] + 1


# In[8]:


print(frequencyOfFirstAndThirdWord['enter']['henry'])
print(frequencyOfFirstAndThirdWord['king']['lord'])


# # Step-5 : Identify the probabilities for conscutive words and alternative words occurances and assign them to respective combinations.

# In[9]:


sumOfAllFrequencies = 0
for word in frequencyOfPairOfWords:
    for successor in frequencyOfPairOfWords[word]:
        sumOfAllFrequencies = sumOfAllFrequencies + frequencyOfPairOfWords[word][successor]
    for successor in frequencyOfPairOfWords[word]:
        frequencyOfPairOfWords[word][successor] = frequencyOfPairOfWords[word][successor] / sumOfAllFrequencies
sumOfAllFrequencies = 0
for word in frequencyOfFirstAndThirdWord:
    for successor2 in frequencyOfFirstAndThirdWord[word]:
        sumOfAllFrequencies = sumOfAllFrequencies + frequencyOfFirstAndThirdWord[word][successor2]
    for successor2 in frequencyOfFirstAndThirdWord[word]:
        frequencyOfFirstAndThirdWord[word][successor2] = frequencyOfFirstAndThirdWord[word][successor2] / sumOfAllFrequencies


# In[10]:


print("Sample probabilites of words occuring in pairs")
print(frequencyOfPairOfWords['enter']['king'])
print(frequencyOfPairOfWords['king']['henry'])
print(frequencyOfFirstAndThirdWord['enter']['henry'])
print(frequencyOfFirstAndThirdWord['king']['lord'])


# # Step-6: Algorithm for next text predictor.
# 
# #### For first given word, evaluate the third possible combination    
# #### For second given word, evaluate the immediate possible combination
# #### Logic to identify a common word for both words as possibilities and calculate the probability together.
# #### Its an "and" probability condition, hence both probabilities need to be multiplied.
# #### Run through all the probabilites and return the word that has the maximum probability
#     

# In[11]:


def nextTextPredictor(line):
    predictScores = {}
    givenWords = line.lower().split()
        
    if (givenWords[0] not in vocabulary or givenWords[1] not in vocabulary): return "Error, words not found in vocabulary"   
    #For first given word, evaluate the third possible combination    
    firstGivenWordProbables = frequencyOfFirstAndThirdWord[givenWords[0]]
    #For second given word, evaluate the immediate possible combination
    secondGivenWordProbables = frequencyOfPairOfWords[givenWords[1]]
    
    #Logic below is to identify a common word for both possibilities and calculate the probability together.
    # Its an "and" probability condition, hence both probabilities need to be multiplied.
    if(len(firstGivenWordProbables)==0 or len(secondGivenWordProbables)==0): return ""
    for eachCombination in firstGivenWordProbables:
            if eachCombination not in secondGivenWordProbables:
                continue
            else: predictScores[eachCombination]=firstGivenWordProbables[eachCombination] * secondGivenWordProbables[eachCombination]
    
    if(len(predictScores)==0): return random.choices(vocabulary)[0]
    #print("Possible texts with their probabilites are \n",predictScores)
    minScore = -1;
    predictedText = ""
    # Run through all probabilities and get the text with maximum probability.
    for eachResult in predictScores:
        if (Decimal(predictScores[eachResult]) > minScore):
            minScore = Decimal(predictScores[eachResult]);
            predictedText = eachResult
    return predictedText;


# In[12]:


predictedText = nextTextPredictor("quarrelling thou")
print("The predicted text is :", predictedText)


# In[13]:


predictedText = nextTextPredictor("othello leave")
print("The predicted text is :", predictedText)


# # Step-7: Algorithm for text generator
# 
# #### Generate the first word in random. 
# #### For second given word, evaluate all probable words for the first word, then pick any word random randomly from probables.
# #### For all other words use the nextTextPredictor for every two consecutive words.

# In[14]:


def generateNewRandomText(numberOfWords):
    #First step is to pick up any word randomly from the vocabulary and use it as our first word.
    newRandomText = ""
    iterate = True
    firstWordInSequence = random.choices(vocabulary)[0]
    while iterate:
        if (firstWordInSequence not in frequencyOfPairOfWords):
            firstWordInSequence = random.choices(vocabulary)[0]
        else:
            iterate = False
    #Now, iterate over the second word possibilities for the first word and randomly select the second word.
    secondWordInSequence = random.choices(list(frequencyOfPairOfWords[firstWordInSequence]))[0]
    newRandomText = firstWordInSequence
    newRandomText = newRandomText + " " + secondWordInSequence
    #Now call the nextTextPredictor in a loop for every 2 consecutive pairs to get the next words in the random sequence. 
    for i in range(numberOfWords-2):
        nextWordInSequence = nextTextPredictor(firstWordInSequence + " " + secondWordInSequence)
        newRandomText = newRandomText + " " + (nextWordInSequence)
        firstWordInSequence = secondWordInSequence
        secondWordInSequence = nextWordInSequence
    return (newRandomText)    


# In[15]:


print("Generated random text : ",generateNewRandomText(5))


# In[16]:


print("Generated random text : ",generateNewRandomText(5))


# In[17]:


print("Generated random text : ",generateNewRandomText(8))

