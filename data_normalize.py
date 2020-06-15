#import libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import  MinMaxScaler

#import dataset
house_df = pd.read_csv('kc_house_dataset.csv', encoding = 'utf-8')

#house_df.describe()
#house_df.head()

#create testing and training dataset/data cleaning
selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']
X = house_df[selected_features]
y = house_df['price']

#print(X.shape)   #(2163,7)
#print(y.shape)   #(2163, )

#Minmax scaler of X
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

y = y.values.reshape(-1, 1)
y_scaled = scaler.fit_transform(y)