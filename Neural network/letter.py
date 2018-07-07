import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
num = 200
def sigmoid(act):
	return 1/(1+math.exp(-act))
def read_data(df , x,y):
	for index,rows in df.iterrows():
		if(index <num):
			l = df.ix[index].values
			x[index][0]=1
			x[index][1:] = l[1:]/10
			y[index][l[0]-65] =1
	return (x,y)

def initialize_network(n_inputs,n_output):
	network = list()
	outputs = list()
	errors = list()
	errors.append(np.zeros(5))
	errors.append(np.zeros(4))
	errors.append(np.zeros(n_output))
	layer1 = np.random.randint(-1,1,(5,n_inputs))
	layer2 = np.random.randint(-1,1,(4,6))
	layer6 = np.random.randint(-1,1,(n_output,5))
	outputs.append(np.zeros(5))
	outputs.append(np.zeros(4))
	outputs.append(np.zeros(n_output))
	network.append(layer1)
	network.append(layer2)
	network.append(layer6)
	return (network,outputs,errors)

def activate(weights , inputs):
	
	act = np.dot(weights,inputs)
	return act

def derivative(output):
	return output * (1.0 - output)

def forward_propagate(network,row,outputs):
	inputs = row
	for i in range(len(network)):
		new_input= list()
		new_input.append(1)
		activat = activate(network[i],inputs)
		for j in range(len(activat)):
			activat[j]= sigmoid(activat[j])
		new_input[1:] = activat
		outputs[i] = activat
		inputs = new_input
	return outputs[2]
		

def back_propogation(network,errors,y,outputs):
	for i in reversed(range(len(network))):
		if i!= len(network)-1:
			layer2 = np.transpose(network[i+1])
			errors[i] = np.dot(layer2,errors[i+1])
			errors[i] = errors[i][1:]
		else :
			errors[i] = y - outputs[i]
		
		for j in range(len(errors[i])):
			errors[i][j]= errors[i][j]*derivative(outputs[i][j])

def update_weights(network,errors,row,l_rate,outputs):
	for i in range(len(network)):
		inputs = row
		if i!=0 :
			inputs = outputs[i-1]
		for j in range(len(errors[i])):
			for k in range(len(inputs)):
				z = l_rate*errors[i][j]*inputs[k]

				network[i][j][k] = network[i][j][k] +z
		

def train_network(network ,x,l_rate,iters ,n_outputs,outputs,y,errors):
	for itr in range(iters):
		right =0
		for i in range(len(x)):
			row = x[i]
			expected = y[i]
			output = forward_propagate(network,row,outputs)
			out=[]
			for j in range(len(output)):
				if output[j]>=0.5 :
					out.append(1)
				else:
					out.append(0)
			if(np.argmax(out)==np.argmax(expected)):
				right=right+1
			back_propogation(network,errors,expected,outputs)
			update_weights(network,errors,row,l_rate,outputs)
		print(str(itr) + " : " +str(right/5))

df = pd.read_csv("./letter-recognition.csv")
df['a0'] = df['a0'].map(ord)
x = np.zeros(shape=(num,17))
y =np.zeros(shape=(num,26))
x ,y =read_data(df,x,y)
network,outputs,errors = initialize_network(17,26)
train_network(network,x,20,100,26,outputs,y,errors)
right =0
for i in range (len(x)):
	row = x[i]
	expected =y[i]
	output = forward_propagate(network,row,outputs)
	print(chr(np.argmax(expected)+65) + " : " + chr(np.argmax(output)+65))
	if(np.argmax(output)==np.argmax(expected)):
		right=right+1;

print(right/5)














