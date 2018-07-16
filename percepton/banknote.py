import pandas as pd
import numpy as np

df = pd.read_csv("./banknote.csv")
df.iloc[:,-1] = df.iloc[:,-1].map({1:1 , 0:-1})
df = df.sample(frac=1)
x_train = df.iloc[:800,0:4].values
y_train = df.iloc[:800,-1].values
x_test = df.iloc[800:,0:4].values
y_test = df.iloc[800:,-1].values
weights = np.ones(4)
eta = 0.5
def train(weights,x_train,y_train,eta):
	iteration =0 
	numchanged =51
	while numchanged > 50:
		numchanged =0
		for i in range(len(x_train)):
			test = x_train[i]
			y = y_train[i]
			output = y*(np.dot(weights,test))
			if output <= 0:
				delta = np.multiply(test,eta*y)
				weights = weights + delta
				numchanged = numchanged +1
		#print(numchanged)
		#print(weights)

	return weights

def predict(weights,x_test,y_test):
	passed =0 
	for i in range(len(x_test)):
		test = x_test[i]
		y = y_test[i]
		output = np.dot(weights,test)
		if output>0:
			output =1
		else:
			output = -1

		if output ==y:
			passed+=1

	print(passed*100/len(x_test))


weights = train(weights,x_train,y_train,eta)
print(weights)
predict(weights,x_test,y_test)

