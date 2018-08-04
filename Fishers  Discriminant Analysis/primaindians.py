import pandas as pd
import numpy as np
df = pd.read_csv("./PimaIndians.csv")
positives = df[df['test'] == 1]
negatives = df[df['test'] == -1]
positives = positives.iloc[:,:8]
negatives = negatives.iloc[:,:8]
positives , negatives = positives.values , negatives.values
pos_mean = np.zeros(8)
neg_mean = np.zeros(8)
for i in range (len(positives)):
	pos_mean = pos_mean + positives[i]
pos_mean =pos_mean / len(positives)
for i in range (len(negatives)):
	neg_mean = neg_mean + negatives[i]
neg_mean =neg_mean / len(negatives)

s1 ,s2 = np.zeros(shape = (8,8)) , np.zeros(shape = (8,8))
for i in range(len(positives)):
	a = np.zeros(shape=(1,8))
	a[0] = positives[i]
	s1 = s1 + np.matmul(np.matrix.transpose(a),a)

for i in range(len(negatives)):
	a = np.zeros(shape=(1,8))
	a[0] = negatives[i]
	s2 = s2 + np.matmul(np.matrix.transpose(a),a)

sw = s1+s2
sw_inv = np.linalg.inv(sw)
theta  = np.matmul(sw_inv,(pos_mean-neg_mean))
t1 = np.dot(theta,pos_mean)
t2 = np.dot(theta,neg_mean)
threshold = 0.5*(t1+t2)
def predict(theta,inputs,threshold):
	output = np.dot(theta,inputs)
	if output > threshold:
		return 1
	else:
		return -1

df = df.values
passed =0
for i in range(len(df)):
	y = df[i][-1]
	test = df[i][:8]
	out = predict(theta,test,threshold)
	if out == y:
		passed+=1
print("theta : ", theta)
print("threshold : ",threshold)
print("passed : ",passed ,"/",len(df))
print("accuracy : " ,(passed*100)/len(df))
