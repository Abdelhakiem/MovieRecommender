import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import load


class MovieRecomModel:
    def __init__(self):
        self.__model = tf.keras.models.load_model('./models/my_model')
        self.__movieScaler = load('./models/movieScaler.bin')
        self.__userScaler = load('./models/userScaler.bin')
        self.__yScaler=load('./models/targetScaler.bin')
        self.__moviesDict=pd.read_csv('./input/content_movie_list.csv')


    def predict(self,user_vec):
        u_s = 3  # start of columns to use in training, user
        i_s = 1  # start of columns to use in training, items
        self.__movies_vec=pd.read_csv('./input/movies_vec.csv')
        self.__movies_vec1=np.array(self.__movies_vec)
        self.__movies_vec1=self.__movieScaler.transform(self.__movies_vec1)

        self.__user_vec=np.array(user_vec)
        self.__user_vec = np.repeat(self.__user_vec,self.__movies_vec.shape[0],axis=0)
        self.__user_vec= self.__userScaler.transform(np.array(self.__user_vec))

        y_p=self.__model.predict([self.__user_vec[:,u_s:],self.__movies_vec1[:,i_s:]])

        y_p=self.__yScaler.inverse_transform(np.array(y_p).reshape(-1,1))
        indices_sorted=np.argsort(-y_p, axis=0).reshape(-1).tolist()
        # ind=indices_sorted[:3]

        return np.array(self.__moviesDict.loc[indices_sorted[:],['title']])[:15]


