import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
num=100

def Cofficients(cofficients,x_train,y_train,alpha,m,itr):
	
	x= x_train.transpose()
    
	for i in range (0,itr):
		hypothesis = np.dot(x_train,cofficients)
		for j in range (0,num):
			hypothesis[j] = sigmoid(hypothesis[j])
		
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
			x_train[index][1:] = l[1:20]
			y_train[index] = l[20]-1
		elif index>=num and index<num+50 :
			l = df.ix[index].values
			x_test[index-num][0]=1
			x_test[index-num][1:] = l[1:20]
			y_test[index-num] = l[20]-1
	return (x_train,y_train,x_test,y_test)
df = pd.read_csv("./hepatitis.csv")
df.replace('?',0)

x_train = np.zeros(shape=(num,20))
y_train = np.zeros(shape = num)
x_test = np.zeros(shape = (num,20))
y_test = np.zeros(shape =num)

x_train , y_train ,x_test ,y_test =initialize(df,x_train,y_train,x_test,y_test)
m,n= np.shape(x_train)
alpha =0.000057
cofficients = np.ones(n)

cofficients= Cofficients(cofficients,x_train,y_train,alpha,m,300000)
count =0
for i in range (0,num):
	predict = sigmoid(np.dot(x_train[i],cofficients))
	if (predict<0.5 and y_train[i] ==1) or (predict>=0.5 and y_train[i] ==0):
		count=count+1
	
print(count)