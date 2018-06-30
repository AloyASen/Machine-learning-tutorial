import pandas as pd
import quandl 
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm 
from sklearn.linear_model import LinearRegression
df = quandl.get('WIKI/GOOGL')
# remove the unvanted features 
df= df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
# add some extra features
df['HL_PCT']=  (df['Adj. High']- df['Adj. Close'])/df['Adj. Close']  *100
df['PCT_change']=  (df['Adj. Close']- df['Adj. Open'])/df['Adj. Open']  *100

df= df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
#add some labels
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace= True)
forecast_out= int(math.ceil(0.1*len(df)))
df['label']= df[forecast_col].shift(-forecast_out)
df.dropna(inplace= True)
print(df.head())
x= np.array(df.drop(['label'],1))
y= np.array(df['label'])

#feature scaling
x=preprocessing.scale(x)
# x= x[:-forecast_out+1]
df.dropna(inplace=True)
y=np.array(df['label'])
#just a check -- should be same
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size= 0.2)
clf= LinearRegression()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print(accuracy)
 