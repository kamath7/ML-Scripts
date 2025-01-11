import tensorflow as tf 
import pandas as pd
import numpy as np 
import onnx
import tf2onnx 

data = pd.read_csv("./Salary_Data.csv")

x = data["YearsExperience"].values.reshape(-1,1)
y = data["Salary"].values



