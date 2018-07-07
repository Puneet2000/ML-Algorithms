import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
num=800
def Cofficients(cofficients,x_train,y_train,alpha,m,itr):
	
	x= x_train.transpose()
    
	for i in range (0,itr):
		hypothesis = np.dot(x_train,cofficients)
		
		loss=  hypothesis -y_train
		cost = np.sum(abs(loss))/(2*m)
		
		gradient = np.dot(x,loss)/m
		cofficients = cofficients - (alpha * gradient)
	print(cost)
	return cofficients

def initialize(df , x_train,y_train,x_test,y_test):
	for index,rows in df.iterrows():
		if(index <num):
			l = df.ix[index].values
			x_train[index][0]=1
			x_train[index][1:] = l[0:11:1]
			y_train[index] = l[11]
		elif index>=num and index<2*num :
			l = df.ix[index].values
			x_test[index-num][0]=1
			x_test[index-num][1:] = l[0:11:1]
			y_test[index-num] = l[11]
	return (x_train,y_train,x_test,y_test)
    
df = pd.read_csv("./redwine.csv")
x_train = np.zeros(shape=(num,12))
y_train = np.zeros(shape=num)
x_test = np.zeros(shape=(num,12))
y_test = np.zeros(shape =num)

x_train , y_train ,x_test ,y_test =initialize(df,x_train,y_train,x_test,y_test)
m,n= np.shape(x_train)
alpha =0.00035
cofficients = np.ones(n)

cofficients= Cofficients(cofficients,x_train,y_train,alpha,m,1000000)

for i in range(0,200):
	predict = np.dot(x_test[i],cofficients) 
	plt.scatter(y_test[i],predict)

plt.show()
