from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random


style.use('fivethirtyeight')
#xs = np.array([1,2,3,4,5,6], dtype= np.float64)
#ys = np.array([5,4,6,5,6,7], dtype= np.float64)


def createDataset(hm, variance, step=2, correlation=False):
    val =1 
    ys=[]
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -=step 
    xs = [i for i in range(len(ys))] 
    return np.array(xs, dtype= np.float64),np.array(ys, dtype= np.float64)

def bestFitSlopeAndIntercept(xs, ys):
    ### this is the slope of the best fit line 
    m= ((mean (xs) * mean(ys))- mean(xs*ys)) / ((mean(xs)**2)-mean(xs **2))
    b = mean(ys)-m * mean(xs)
    return m, b

def squaredError(ys_orig, ys_line):
    return sum((ys_line- ys_orig)**2)

def coeffOfDetermination(ys_orig, ys_line):
    y_mean_line= [mean(ys_orig) for y in ys_orig]
    squaredError_regr = squaredError(ys_orig, ys_line)
    squaredError_yMean= squaredError(ys_orig, y_mean_line)
    return 1 - (squaredError_regr / squaredError_yMean)


#create a test dataset
##decrease the variance in the test dataset to decrease the rendomness in the data
###chage the correlation to negative to show that it is not a dataset to linear regression
xs, ys = createDataset(40, 40, 2, correlation='pos')


m, b= bestFitSlopeAndIntercept(xs, ys)

#  print (m , b) 
regression_line =[ (m*x)+ b for x in xs ]

# use this best fit line to make a prediction 

predict_x= 8
predict_y = (m* predict_x)+b

#to determine the accuracy 
#use rsquared error

r_squared = coeffOfDetermination(ys, regression_line)
print(r_squared)

plt.scatter(predict_x,predict_y,s=150,  color='g')
plt.scatter(xs,ys)
plt.plot(xs, regression_line)
plt.show()
