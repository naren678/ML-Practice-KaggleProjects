
# Reading data 
import pandas as pd
import numpy as np

# Importing necessary models for implementation of ANN
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda

Concrete = pd.read_csv("E:\\Datasets\\Neural Networks\\concrete.csv")
Concrete.head()


cont_model = Sequential()
cont_model.add(Dense(50,input_dim=8,activation="relu"))
cont_model.add(Dense(40,activation="relu"))
cont_model.add(Dense(20,activation="relu"))
cont_model.add(Dense(1,kernel_initializer="normal"))
cont_model.compile(loss="mean_squared_error",optimizer = "adam",metrics = ["mse"])

column_names = list(Concrete.columns)
predictors = column_names[0:8]
target = column_names[8]

#first_model = prep_model([8,50,1])
first_model = cont_model
first_model.fit(np.array(Concrete[predictors]),np.array(Concrete[target]),epochs=10)
pred_train = first_model.predict(np.array(Concrete[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-Concrete[target])**2))
