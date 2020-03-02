"""
Class for handing the NIH chest X-ray data. Designed to access images 
stored locally in a single directory.

Access the data:
Origin: https://nihcc.app.box.com/v/ChestXray-NIHCC
All images: https://rads-reader-images.s3.us-east-2.amazonaws.com/all_images/

"""

# Author: Justin Paul Turner <justin@justinpturner.com>
# License: BSD 3 clause
# Class for handing the chest X-ray data
# Designed to access data that is stored locally on the ec2 machine

#TODO: 
    #Currently have a massive training array with mathcing indexes to the condition dataframe.
    #Need to select only the rows from that array to train the model on.
    #Current code creates the dataframe I need but does not have the training data...
    #Going to try to index the numpy array based on the index
    #Need to verify that the indexing I did reorders the training data to match the 
    #Verified! Great work!
    #Need to train model and add it to this doc while the model trains
    #Need to load the X data onto the s3 so that the class can be called on any computer
    #Model trained
    #Need to test model
    

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Rads_Reader:
    
    def __init__(self, 
                 condition = 'Cardiomegaly', 
                 bucket='rads-reader'):
        self.condition = condition
        self.bucket = bucket
    
    def ClassificationDf(self):
        df = pd.read_csv('https://rads-reader-images.s3.us-east-2.amazonaws.com/Image_Dateframe.csv', index_col = 0)
        return df

    def HalfConditionDf(self):
        df = self.ClassificationDf()
        nf_df = df[df['No Finding'] == 1].iloc[:sum(df[self.condition])] # Dataframe of no findings of the same length of the specified condition 
        condition_df = df[df[self.condition] == 1]
        half_condition_df = pd.concat([nf_df, condition_df])[['No Finding', self.condition]].sample(frac=1, random_state=0)
        return half_condition_df[['No Finding', self.condition]]
    
    def y(self): 
        return self.HalfConditionDf().to_numpy()
        
    def X(self): # The training array of half condition, half no_finding shuffled
        training_df = self.HalfConditionDf()
        X = pickle.load( open( "training_array.p", "rb" ) ).reshape((112120, 150, 150, 1))
        X_half_selection = X[[training_df.index]]
        return X_half_selection
    
    
        
        
