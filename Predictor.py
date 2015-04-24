#!/usr/bin/python
from numpy import loadtxt, zeros, ones, append, array, mean, std, arange, float, fromiter;
from math import log10, ceil;

class LinRegress(object):
	def __init__(self):
		self.theta = 0; #predictor weight(s) for factors

	def normalizeFeatures(self, cols, X):
		'Helps gradient descent work faster by normalizing features with varaying scales'
		X_norm = X;

		for i in range(cols):
			m = mean(X[:,i]);
			s = std(X[:, i]);
			n = (X[:, i] - m);
			X_norm[:,i] = n / s;

		return X_norm;

	def computeCost(self, rows, theta):
		'computes square error between prediction and expected value'
		hyp = self.X.dot(theta);
		sqError = (hyp - self.Y)**2;

		return (1.0 / (2 * rows)) * sum(sqError);

	def gradientDescent(self, rows, steps, theta):
		'minimizes error between prediction and expected value, descent is optimized by use of adaptive alpha'
		curError = 1 + max(self.Y)**3; #assign the expected error some large value
		X_trans = self.X.T;

		alpha = 0.01;
		rho = 1.1;
		sig = 0.5;

		for i in range(steps):
			hyp = self.X.dot(theta);
			theta -= alpha * (1.0 / rows) * (X_trans.dot(hyp - self.Y));

			#increase alpha as long as error keeps minimizing, else take a step back and continue 
			error = self.computeCost(rows, theta);
			#print error[0], '|', curError[0];
			if error < curError[0]:
				curError = error;
				alpha = alpha * rho;
				self.theta = theta;
			else:
				alpha = alpha * sig;
				theta = self.theta;

	def checkScales(self, X, cols, deg):
		'If factors have varaying scales, module recommends to normalize data'
		mx,mn = [],[];
		for i in range(cols): #find the exponent value for each columns' max min
			mx.append( abs( log10( max( X[:, i] ) ) ) );
			mn.append( abs( log10( min( X[:, i] ) ) ) );

		for i in range(cols-1): #determine if columns deviate more than deg from each other
			for j in range(i+1, cols):
				if abs(mx[i]-mx[j]) > deg or abs(mn[i]-mn[j]) > deg:
					return True; #data needs to be normalized
		return False; #data does not need to be normalized

	def train(self, steps, deg, trainSRC, delim):
		self.delim = delim;
		data = loadtxt(trainSRC, delimiter=self.delim);
		rows, cols = data.shape;

		X = data[:, :cols-1]; #first columns are features for the outcome
		self.Y = data[:, cols-1]; #last column is outcome
		self.Y.shape = [rows, 1]; #properly define the shape of y
		self.Y = self.Y;
		self.X = ones((rows, cols)); #default bias is 1
		self.theta = zeros(shape=(cols,1));

		if self.checkScales(X, cols-1, deg): #check if data needs to be normalized
			X = self.normalizeFeatures(cols-1, X);

		self.X[:, 1:] = X;
		self.gradientDescent(rows, steps, self.theta);

	def predict(self, dat):
		'returns a prediction for the datum.'
		if type(self.theta) == int:
			print '\n> Predictor is not trained Yet. Please train before making any predictions\n';
			return None;
		else:
			dat = fromiter(dat.split(self.delim), dtype = float); #split dat string to np.array for prediction
			datum = ones((1, self.theta.shape[0])); #insert bias into input data
			datum[:, 1:] = dat;
			return datum.dot(self.theta)[0][0]; #compute prediction


