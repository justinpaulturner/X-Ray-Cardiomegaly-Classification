#Justin Turner
#Class for handing the chest X-ray data
import numpy as np
import pandas as pd
import boto3
import matplotlib.pyplot as plt
import cv2
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Rads_Reader:
    
    def __init__(self, bucket='rads-reader'):
        self.bucket = bucket
    
    def ClassificationDf(self):
        ### AWS S3 Setup ###
        client = boto3.client('s3') #low-level functional API
        s3 = client
        resource = boto3.resource('s3') 


        ### Adding column to dataframe for cardiomegaly ###

        obj = client.get_object(Bucket = 'rads-reader', Key = 'data/categorical.csv')
        categorical_df = pd.read_csv(obj['Body'])
        categorical_df['cardiomegaly'] = [1 if 'Cardio' in x else 0 for x in categorical_df['Finding Labels']]

        ### Create condensed version of dataframe ###
        df = categorical_df[['Image Index','Finding Labels','cardiomegaly']]
        return df
        
    def IndexedDf(self):
        indexed_w_all = self.ClassificationDf().set_index('Image Index')
        return indexed_w_all
        
    def SampleTrainingArray(self, n_images=20, IMG_SIZE=100):
        training_data = []
        s3 = boto3.resource('s3')
        bucket = s3.Bucket('rads-reader')
        for img_name in self.ClassificationDf()['Image Index'][:n_images]:
            object = bucket.Object('data/images/'+img_name)
            tmp = tempfile.NamedTemporaryFile()
            class_num = self.IndexedDf().loc[img_name]['cardiomegaly']
            with open(tmp.name, 'wb') as f:
                object.download_fileobj(f)
                img_array=cv2.imread(tmp.name) #creating image array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                avg_array = []
                for num in new_array:
                    for nums in num:
                        avg_array.append(np.mean(nums))
                training_data.append([img_name, avg_array, class_num])
        return training_data
    
    def TrainingArray(self, IMG_SIZE=100):
        training_data = []
        s3 = boto3.resource('s3')
        bucket = s3.Bucket('rads-reader')
        for img_name in self.ClassificationDf()['Image Index']:
            object = bucket.Object('data/images/'+img_name)
            tmp = tempfile.NamedTemporaryFile()
            class_num = self.IndexedDf().loc[img_name]['cardiomegaly']
            with open(tmp.name, 'wb') as f:
                object.download_fileobj(f)
                img_array=cv2.imread(tmp.name) #creating image array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                avg_array = []
                for num in new_array:
                    for nums in num:
                        avg_array.append(np.mean(nums))
                training_data.append([img_name, avg_array, class_num])
        return training_data

    def PlotImage(self, img_name='00000001_000.png'):
        s3 = boto3.resource('s3')
        bucket = s3.Bucket('rads-reader')
        object = bucket.Object(f'data/images/{img_name}')
        tmp = tempfile.NamedTemporaryFile()

        with open(tmp.name, 'wb') as f:
            object.download_fileobj(f)

            img_array=cv2.imread(tmp.name)
            plt.imshow(img_array,cmap="gray")
            plt.show()
    
    def SampleTrainingDf(self):
        image_df = pd.DataFrame(self.SampleTrainingArray())
        image_df = image_df.rename(columns={0: "img_name", 1: "avg_array", 2: "class_num"})
        smol_df = image_df.apply(pd.Series)
        matrix_df = image_df.avg_array.apply(pd.Series)
        df_merged = pd.concat([smol_df, matrix_df], axis=1, sort=False)
        df_merged = df_merged.drop('avg_array', axis = 1)
        return df_merged
        
    def SampleTrainTestVariables(self):
        y = self.SampleTrainingDf()['class_num']
        X = self.SampleTrainingDf().drop(['img_name', 'class_num'], axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=69)
        return X_train, X_test, y_train, y_test
    
    def TrainingDf(self):
        image_df = pd.DataFrame(self.TrainingArray())
        image_df = image_df.rename(columns={0: "img_name", 1: "avg_array", 2: "class_num"})
        smol_df = image_df.apply(pd.Series)
        matrix_df = image_df.avg_array.apply(pd.Series)
        df_merged = pd.concat([smol_df, matrix_df], axis=1, sort=False)
        df_merged = df_merged.drop('avg_array', axis = 1)
        return df_merged

    def TrainTestVariables(self):
        y = self.TrainingDf()['class_num']
        X = self.TrainingDf().drop(['img_name', 'class_num'], axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=69)
        return X_train, X_test, y_train, y_test
    
    
    
    
    
    
    
    
    
    
    
    
    