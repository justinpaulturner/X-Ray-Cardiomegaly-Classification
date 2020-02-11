#Justin Turner
#Class for handing the chest X-ray data
#Designed to access data that is stored locally on the ec2 machine

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import rescale
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Rads_Reader:
    
    def __init__(self, bucket='rads-reader'):
        self.bucket = bucket
    
    def ClassificationDf(self):
        labels = pd.read_csv("https://rads-reader.s3.us-east-2.amazonaws.com/data/categorical.csv")
        conditions = ['Cardiomegaly','Emphysema','Effusion',
                      'Hernia','No Finding','Infiltration',
                      'Nodule','Mass','Pneumothorax','Atelectasis',
                     'Pleural_Thickening','Fibrosis']
        for condition in conditions:
            labels[condition] = [1 if condition in x else 0 for x in labels['Finding Labels']]
        df = labels[['Image Index',
               'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'No Finding',
               'Infiltration', 'Nodule', 'Mass', 'Pneumothorax', 'Atelectasis',
               'Pleural_Thickening', 'Fibrosis']]
        return df

    def HalfConditionDf(self, condition = 'Hernia'):
        df = self.ClassificationDf()
        nofinding_df = df[df['No Finding'] == 1]
        nofinding_df_len = nofinding_df.iloc[:sum(df[condition])]
        condition_df = df[df[condition] == 1]
        half_condition_df = pd.concat([nofinding_df_len, condition_df])
        half_condition_df = half_condition_df[['Image Index', condition]]
        half_condition_df = half_condition_df.sample(frac=1)
        indexed_w_all = df.set_index('Image Index')

        training_data= []
        for img_name in half_condition_df['Image Index']:
            condition_pos = indexed_w_all.loc[img_name][condition]
            img_array = imread(f'images/{img_name}', as_grey=True)
            new_array = rescale(img_array, 1/3, mode='reflect')
            hog_array, xray_hog_img = hog(
                new_array, pixels_per_cell=(12, 12),
                cells_per_block=(2,2),
                orientations=8,
                visualise=True,
                block_norm='L2-Hys')
            training_data.append([img_name, hog_array, condition_pos])


        hog_df = pd.DataFrame(training_data)
        hog_df = hog_df.rename(columns={0: "img_name", 
                                        1: "hog_array", 
                                        2: condition})
        smol_df = hog_df.apply(pd.Series)
        matrix_df = hog_df.hog_array.apply(pd.Series)
        df_merged = pd.concat([smol_df, matrix_df], axis=1, sort=False)
        df_merged = df_merged.drop('hog_array', axis = 1)

        return df_merged


    def HalfConditionDf_(self, condition = 'Hernia'):
        df = self.ClassificationDf()
        nofinding_df = df[df['No Finding'] == 1]
        nofinding_df_len = nofinding_df.iloc[:sum(df[condition])]
        condition_df = df[df[condition] == 1]
        half_condition_df = pd.concat([nofinding_df_len, condition_df])
        half_condition_df = half_condition_df[['Image Index', condition]]
        half_condition_df = half_condition_df.sample(frac=1)
        indexed_w_all = df.set_index('Image Index')

        training_data= []
        for img_name in half_condition_df['Image Index']:
            condition_pos = indexed_w_all.loc[img_name][condition]
            img_array = imread(f'images/{img_name}', as_grey=True)
            new_array = rescale(img_array, 1/3, mode='reflect')
            hog_array = hog(new_array)
            training_data.append([img_name, hog_array, condition_pos])


        hog_df = pd.DataFrame(training_data)
        hog_df = hog_df.rename(columns={0: "img_name", 
                                        1: "hog_array", 
                                        2: condition})
        smol_df = hog_df.apply(pd.Series)
        matrix_df = hog_df.hog_array.apply(pd.Series)
        df_merged = pd.concat([smol_df, matrix_df], axis=1, sort=False)
        df_merged = df_merged.drop('hog_array', axis = 1)

        return df_merged
        
    def IndexedDf(self):
        indexed_w_all = self.ClassificationDf().set_index('Image Index')
        return indexed_w_all
        