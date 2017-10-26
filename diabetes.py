import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

diabetesdataset =pd.read_csv("D:\ML\diabetes.csv",header=0)
 
desdiabetes=pd.DataFrame(diabetesdataset)
X=desdiabetes['Pregnancies']
Y=desdiabetes['Age']

X_train, X_test, Y_train, Y_test = train_test_split ( X,Y )
regre= LinearRegression ()
regre.fit(X_train, Y_train)


