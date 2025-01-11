import tensorflow as tf 
import pandas as pd
import numpy as np 
import onnx
import tf2onnx


data = pd.read_csv("./Salary_Data.csv")

X = data["YearsExperience"].values.reshape(-1,1)
y = data["Salary"].values

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=32 )

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_dim = 1)
])
model.compile(optimizer="adam", loss="mse")

model.fit(X_train_scaled, y_train, epochs=100)




