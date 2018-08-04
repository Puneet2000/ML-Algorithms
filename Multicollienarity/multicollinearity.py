import numpy as np

class MultiCollinearity:
	def __init__(self,X,n,k):
		self.X = X
		self.n = n
		self.k = k

	def XtX_determinant(self):
		X_t = np.transpose(self.X)
		XtX = np.matmul(X_t,self.X)
		return np.linalg.det(XtX)

	def coorelation_matrix(self):
		X_t = np.transpose(self.X)
		return np.corrcoef(X_t)

	def correlation_det(self):
		co = self.coorelation_matrix()
		return np.linalg.det(co)

	def condition_number(self):
		XtX = np.transpose(self.X)
		XtX = np.matmul(XtX,self.X)
		w,v = np.linalg.eig(XtX)
		eig_max = max(w)
		eig_min = min(w)
		return (eig_max/eig_min)

	def VIF(self):
		XtX = np.transpose(self.X)
		XtX = np.matmul(XtX,self.X)
		if np.linalg.det(XtX) !=0:
			XtX = np.linalg.inv(XtX)
			vifs = [XtX[i][i] for i in range(self.k)]
			return vifs
		else : return None

	def calculate(self):
		xtx_det = self.XtX_determinant()
		cormatrix = self.coorelation_matrix()
		co_det = self.correlation_det()
		cn = self.condition_number()
		vifs = self.VIF()
		severity = {"|X'X|":xtx_det , "D":cormatrix ,"|D|":co_det ,"CN":cn , "VIF":vifs}
		return severity



