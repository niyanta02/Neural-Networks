#!/usr/bin/env python
# coding: utf-8

# ## Python program to classify the different species of the iris flower.

# In[1]:


#importing basic libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


#load and read the dataset
iris_dataset = pd.read_csv("C:/Users/niyan/OneDrive/Desktop/iris.csv",  header=0, names=['sepl', 'sepw', 'petl', 'petw', 'class'])


# In[4]:


#data exploration 

print(iris_dataset.head(5))
print(iris_dataset.describe())
print(iris_dataset.info())
print(iris_dataset.shape)
print(iris_dataset['class'].value_counts())


# In[5]:


#create output values according to labels in data
iris_dataset['out_versicolor'] = iris_dataset['class'].apply( lambda x: 1 if x == 'Iris-versicolor' else 0)
iris_dataset['out_virginica'] = iris_dataset['class'].apply( lambda x: 1 if x == 'Iris-virginica' else 0)
iris_dataset['out_setosa'] = iris_dataset['class'].apply( lambda x: 1 if x == 'Iris-setosa' else 0)

iris_dataset.head(5)


# In[6]:


#versicolor validation
iris_dataset[iris_dataset['class']=='Iris-versicolor']


# In[7]:


#remove unnecessary column
iris_dataset_prep = iris_dataset.drop(labels='class', axis=1)


# In[8]:


iris_dataset_prep.head(5)


# In[9]:


#function to create random weight to initialize thr network
def create_weights(num_units, num_inputs):
    weights = np.array(np.random.rand(num_units, num_inputs))
    return weights

# function to compute the sigmoid
sigmoid = lambda x: 1/(1 + np.exp(-x));

# function to format results selecting the biggest value
def format_results( results_raw):  
    '''
    Creates a results array by selecting the maximun value per row 
    from the raw results obtained from the NN
    '''
    results = np.zeros(shape=results_raw.shape)
    for i in range(0,results_raw.shape[0]):
        results[i][np.argmax(results_raw[i])] = 1
    
    return results


# In[10]:


# function to execute final feed forward step with the trained network
def feed_forward(wl,w2, X, D):
    '''
    This function calculates only the feed forward step for the network
    It is used to calculate the final results of the network
    whl = weights hidden layer
    wol = weights output layer
    '''
    num_records = X.shape[0]
    results = a = np.zeros(D.shape)
    
    for i in range(0,num_records):
        
        #for each row of values and results
        x = X[i, :].T;    # current set of values
        d = D[i];        # current set of results
        
        # FEEDFORWARD SECTION
        # HIDDEN LAYER
        # Calculate weighted sum and output
        v1 = np.dot(wl,x);
        y1 = sigmoid(v1);
        
        # OUTPUT LAYER
        # Calculate weighted sum and output
        v = np.dot(w2,y1);
        y = sigmoid(v);
        
        #create results
        print(f'd = {d} y = {y}')
        results[i] = y
        
    return results


# In[11]:


################################################################################
# This function implements the backpropagation algorithm for a simple 3-4-1 ANN
#  W1 and W2 are the weight matrices of the respective layers:
#   W1 is the weight matrix between the input layer and hidden layer
#   W2 is the weight matrix between the hidden layer and output layer. 
#  X and D are the input and correct output of the training data (XOR), respectively.
################################################################################
def backprop(w1, w2, X, D):
    alpha = 0.9; # learning rate
     ### size variables to avoid hardcoding
    num_records = X.shape[0]
    size_input = X.shape[1]
    size_output = D.shape[1]
    units_hidden = w1.shape[0]
    units_output = w2.shape[0]
    
    for i in range(0,num_records):
        x = X[i, :].T; #inputs from training data
        #print("x=",x)
        d = D[i]; # correct output from training data
        ##########################
        # forward propagation step
        ##########################
        # calculate the weighted sum of hidden node
        v1 = np.dot(w1,x);
        #print("v1= ", v1)
        #pass the weighted sum to the activation function, this gives the outputs from hidden layer
        y1 = sigmoid(v1);
        #print("y1= ", y1)
        #calculate the weighted sum of the output layer
        v = np.dot(w2,y1);
        #print("v", v)
        # pass it to the activation function, this returns the output of the third layer
        y = sigmoid(v);
        #print("y=",y)
        #calculate the error, difference between correct output and computed output
        e = d - y;
        #print("e= ",e)
        #calculate delta, derivative of the activation function times the error
        # note that 𝜎′(𝑥)=𝜎(𝑥)∙(1− 𝜎(𝑥)) = y * (1-y)
        delta = y*(1-y)*e; # element wise multiplication
        #print("delta= ",delta)
        ###########################
        # Backward propagation step
        ###########################
        # propagate the output node delta, δ, backward, and calculate the deltas of the hidden layer.
        e1 = np.dot(w2.T, delta);
        #print("e1= ",e1)
        delta1 = y1*(1-y1)*e1;  # element wise multiplication
        #print("delta1= ",delta1)
        #
        # Adjust the weights according to the learning rule
        delta1.shape=(units_hidden,1) # column vector of deltas for the hidden layer
        x.shape=(1,size_input) # row vector of the current input
        dw1 = alpha*np.dot(delta1,x);
        w1 = w1 + dw1;
        #output layer
         # OUTPUT LAYER
        delta.shape = (size_output,1)
        y1.shape = (1, units_hidden)
        
        #print(y1.T.shape)
        dw2 = alpha*np.dot(delta,y1);
        w2 = w2 + dw2;
    #
    return w1, w2;


# In[14]:


# CREATE INPUT AND OUTPUT ARRAYS

inputs = iris_dataset_prep.iloc[:,0:4].to_numpy() #Input values (n,4)
outputs = iris_dataset_prep.iloc[:,4:].to_numpy() #Output values (n,3)

#train and test sets (70%-30%)
inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
    inputs, outputs, test_size=0.3, random_state=301133331)


# In[22]:


# MAIN PROGRAM EXECUTION
num_ephocs = 20

# Create initial weights
wl = create_weights(5,4)
w2 = create_weights(3,5)

print('--- INITIALIZE NETWORK WEIGHTS')
print('\tHidden Layer Weights')
print(wl)
print()
print('\tOutput Layer Weights')
print(w2)

print('--- NETWORK ARCHITECTURE')
print(f'Weights Hidden Layer Shape {wl.shape}')
print(f'Weights Output Layer Shape {w2.shape}')

print()
print('--- TRAINING NETWORK - ADJUST WEIGHTS')
print(f'Train Inputs Shape {inputs_train.shape}')
print(f'Train Outputs Shape {outputs_train.shape}')
print(f'Testing will proceed with {num_ephocs} ephocs!!')

# Use back propagation to obtain NN weights
for ei in range(1,num_ephocs+1):
    print(f'\tEPOCH: {ei}')
    wl, w2 = backprop(wl, w2, inputs_train, outputs_train)   


# In[20]:


print('--- TESTING RESULTS ---')
print(f'Test Inputs Shape {inputs_test.shape}')
print(f'Test Outputs Shape {outputs_test.shape}')

# Calculate results after training
print('--- RAW RESULTS ---')
results = feed_forward(wl, w2, inputs_test, outputs_test)
results_final = format_results(results)


# In[21]:


print('--- FORMATTED RESULTS ---')
print('EXPECTED', "\t", 'PREDICTED')
for i in range(0, results_final.shape[0]):
    print(outputs_test[i], "\t", results_final[i])


accuracy = accuracy_score(outputs_test,results_final)
print(f'\ACCURACY SCORE: {accuracy}')

