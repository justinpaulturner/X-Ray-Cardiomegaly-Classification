#Justin Turner
#Class for handing the chest X-ray data
class Rads_Reader:
    def __init__(self, bucket):
        self.bucket = bucket
    
    def Classification_df(self):
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
        
    def Indexed_df(self):
        indexed_w_all = self.Classification_df.set_index('Image Index')
        return indexed_w_all
        
    def Training_data_small_sample(self):
        training_data = []
        for img_name in self.Classification_df()['Image Index'][:20]:
            object = bucket.Object(key)
            tmp = tempfile.NamedTemporaryFile()
            with open(tmp.name, 'wb') as f:
                object.download_fileobj(f)
                img_array=cv2.imread(tmp.name) #creating image array
                IMG_SIZE = 100
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                avg_array = []
                for num in new_array:
                    for nums in num:
                        avg_array.append(np.mean(nums))
                training_data.append([img_name, avg_array, class_num])
        return training_data


