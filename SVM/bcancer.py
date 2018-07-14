import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def linear_kernel (x ,y,b=1):
	return x @ y.T +b

def gaussian_kernel(x,y,sigma =1):
	diffs = x-y
	result = np.exp(-np.linalg.norm(diffs)/(2*sigma*sigma))
	return result

class SVMModel:
	def __init__ (self,X,y,C,kernel,alphas,b,errors):
		self.X = X
		self.y = y
		self.C = C
		self.kernel = kernel
		self.alphas = alphas
		self.b = b
		self.errors = errors
		self._obj = []
		self.m = len(self.X)

def objective_function(alphas,target,kernel,X_train,):
		return np.sum(alphas) - 0.5*np.sum(target*target*kernel(X_train,X_train)*alphas*alphas)

def decision_function (alphas,target,kernel,X_train,x_test,b):
		result =  np.dot((alphas*target),kernel(X_train,x_test)) -b
		return result

def take_step(i1,i2,model):
	if i1==i2:
		return 0,model

	alph1 , alph2 = model.alphas[i1] , model.alphas[i2]
	y1 , y2 = model.y[i1] , model.y[i2]
	E1 , E2 = model.errors[i1] , model.errors[i2]

	s= y1*y2

	if (y1 !=y2):
		L = max(0,alph2-alph1)
		H= min(model.C , model.C + alph2-alph1)
	elif(y1==y2):
		L = max(0,alph2+alph1 - model.C)
		H= min(model.C , alph2-alph1)

	if(L == H):
		return 0,model

	k11 , k12 ,k22 = model.kernel(model.X[i1],model.X[i1]) , model.kernel(model.X[i1],model.X[i2]),model.kernel(model.X[i2],model.X[i2])
	eta = 2*k12 -k11 -k22

	if(eta<0):
		a2 = alph2 -y2*(E1-E2)/eta

		if L<a2<H:
			a2=a2
		elif a2 <= L:
			a2 = L
		elif a2>=H:
			a2 = H

	else:
		alphas_adj = model.alphas.copy()
		alphas_adj[i2] = L
		Lobj = objective_function(alphas_adj, model.y, model.kernel, model.X) 
		alphas_adj[i2] = H
		Hobj = objective_function(alphas_adj, model.y, model.kernel, model.X)
		if Lobj > (Hobj + eps):
			a2 = L
		elif Lobj < (Hobj - eps):
			a2 = H
		else:
			a2 = alph2

	if a2 < 1e-8:
		a2 =0.0
	elif a2 > (model.C -1e-8):
		a2 = model.C

	if (np.abs(a2-alph2) < eps *(a2 + alph2 +eps)):
		return 0,model

	a1 = alph1 +s*(alph2-a2)

	b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + model.b
	b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + model.b

	if 0 <a1  and a1 < model.C:
		b_new = b1
	elif 0 < a2 and a2 <C:
		b_new = b2

	else:
		b_new = (b1+b2)*0.5

	model.alphas[i1] , model.alphas[i2]= a1,a2

	for index,alph in zip([i1,i2],[a1,a2]):
		if 0.0 <alph <model.C:
			model.errors[index] = 0.0

	non_opt = [n for n in range(model.m) if (n!=i1 and n!=i2)]
	model.errors[non_opt] =  model.errors[non_opt] + \
	y1*(a1 - alph1)*model.kernel(model.X[i1], model.X[non_opt]) + \
	y2*(a2 - alph2)*model.kernel(model.X[i2], model.X[non_opt]) + model.b - b_new

	model.b= b_new
	return 1, model

def examine_example(i2,model):
	y2 = model.y[i2]
	alph2 = model.alphas[i2]
	E2 = model.errors[i2]
	r2 = E2*y2

	if ((r2 < -tol and alph2 < model.C) or (r2 > tol and alph2 > 0)):
		if len(model.alphas[(model.alphas != 0) & (model.alphas != model.C)]) > 1:
			if model.errors[i2] > 0:
				i1 = np.argmin(model.errors)
			elif model.errors[i2] <= 0:
				i1 = np.argmax(model.errors)
				step_result, model = take_step(i1, i2, model)
				if step_result:
					return 1, model

	for i1 in np.roll(np.where((model.alphas != 0) & (model.alphas != model.C))[0],np.random.choice(np.arange(model.m))):
		step_result, model = take_step(i1, i2, model)
		if step_result:
			return 1, model
	for i1 in np.roll(np.arange(model.m), np.random.choice(np.arange(model.m))):
		step_result, model = take_step(i1, i2, model)
		if step_result:
			return 1, model

	return 0,model

def train(model):
	numchanged =0
	examineAll =1
	while(numchanged > 0) or (examineAll):
		numchanged =0
		if examineAll:
			for i in range(model.alphas.shape[0]):
				examine_result , model = examine_example(i,model)
				numchanged += examine_result
				if examine_result:
					obj_result = objective_function(model.alphas, model.y, model.kernel, model.X)
					model._obj.append(obj_result)
		else:
			for i in np.where((models.alphas!=0) & (model.alphas != model.C))[0]:
				examine_result , model = examine_example(i,model)
				numchanged += examine_result
				if examine_result:
					obj_result = objective_function(model.alphas, model.y, model.kernel, model.X)
					model._obj.append(obj_result)

		if examineAll ==1:
			examineAll =0
		elif numchanged ==0:
			examineAll =1

		return model


def predict (x_train,model):
	re= []
	for test in x_train:
		result = sum([(model.y[i])*(model.alphas[i])*(model.kernel(model.X[i],test)) for i in range(model.m)])
		if result <=0:
			result =-1
		else:
			result = 1
		re.append(result)

	return np.asarray(re)



df = pd.read_csv("./bcancer.csv")
y_train = df.iloc[:,-1]
x_train = df.iloc[:,0:9]
y_train = y_train.values
x_train = x_train.values
C=  1000.00

m = len(x_train)
initial_alphas = np.zeros(m)
initial_b = 0.0
tol = 0.01 
eps = 0.01 

model = SVMModel(x_train, y_train, C, gaussian_kernel,
                 initial_alphas, initial_b, np.zeros(m))

initial_error = decision_function(model.alphas, model.y, model.kernel,model.X, model.X, model.b) - model.y
model.errors = initial_error
np.random.seed(0)
output = train(model)

predicted = predict(x_train,output)
diffs = predicted - y_train
zero_count = np.count_nonzero(diffs == 0)
non_zero = np.count_nonzero(diffs == 2) + np.count_nonzero(diffs == -2)
accuracy =  (zero_count *100)/(zero_count+non_zero)
print("training set accuracy : " + str(accuracy))








