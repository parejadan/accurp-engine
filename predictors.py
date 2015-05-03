#!/usr/bin/python
from numpy import ones, append, array, mean, std;
from math import sqrt, exp, pi;
from math import log10;

def checkScales(data, deg):
	'If factors have varaying scales, module recommends to normalize data'
	cols = data.shape[1];
	mx, mn = [], [];
	for i in range(cols): #find the exponent value for each columns' max min
		mx.append( abs( log10( max( data[:, i] ) ) ) );
		mn.append( abs( log10( min( data[:, i] ) ) ) );

	for i in range(cols-1): #determine if columns deviate more than deg from each other
		for j in range(i+1, cols):
			if abs(mx[i]-mx[j]) > deg or abs(mn[i]-mn[j]) > deg:
				return True; #data needs to be normalized
	return False; #data does not need to be normalized

def calcMu(nums):
	return sum(nums)/float(len(nums));

def calcSig(nums):
	mu = calcMu(nums);
	var = sum([pow(x-mu,2) for x in nums])/float(len(nums)-1);
	return sqrt(var);

def normalize(data):
	norm = data;
	cols = data.shape[1];
	mu, sig = [], [];
	for i in range(cols):
		m = mean(data[:, i]); mu.append(m);
		s = std(data[:, i]); sig.append(s);
		norm[:, i] = (data[:, i] - m) / s;
	return norm, mu, sig;

class NaiveBayes(object):
	def __init__(self):
		self.summaries = 0;
		self.db = {}; #dictonary database of additonal information for classification and prediction

	def separateByClass(self, data):
		'separates training data by their outcome class and creates a dictonar \
		 where the class is the key, and training examples are items'

		partedClasses = {} #partition data by their respective outcome class (last column)
		for i in range( len(data) ):
			dat = data[i];
			if dat[-1] not in partedClasses:
				partedClasses[ dat[-1] ] = [] #create a dictionary entry for the new class type
			partedClasses[ dat[-1] ].append( dat[:-1] ) #append values for features

		return partedClasses;

	def summarize(self, dataset):
		'computes mean and standard derivation for a vector dataset \
		if you want to normalize a matrix, use the normalize()'
		return [(calcMu(attribute), calcSig(attribute)) for attribute in zip(*dataset)];

	def summarizeByClass(self, dataset):
		'creates a dictonary for each class where the keys are \
		the class type (decision outcomes) and the itme values are the respective data examples'
		partedClasses = self.separateByClass(dataset);
		self.summaries = {};

		for classVal, instances in partedClasses.iteritems():
			#summarize data by computing the mean and std for each column
			self.summaries[classVal] = self.summarize(instances);

	def calcProbability(slef, x, mu, sig): #formula is gaussian probability density function
		'computes the probability of a single feature to occur'
		print x, mu, sig;
		exponent = exp( -( pow(x-mu, 2) / ( 2*pow(sig, 2) ) ) );
		return (1 / (sqrt(2.0*pi) * sig)) * exponent;

	def calcClassProbabilities(self, inputVector):
		'computes probability of all class to occur for the given feature values'
		probabilities = {};
		for classVal, classSummaries in self.summaries.iteritems(): #for each class
			probabilities[classVal] = 1;
			for i in range(len(classSummaries)):#for each feature of a class
				mu, sig = classSummaries[i];
				#compute the probability for the given input vector
				x = inputVector[i];
				probabilities[classVal] *= self.calcProbability(x, mu, sig);
		return probabilities;

	def predict(self, inputVector):
		'evaluates which class is most likely to occur for the given input vector'
		probabilities = self.calcClassProbabilities(inputVector);
		label, maxProb = None, 0;
		for classVal, prob in probabilities.iteritems():
			print classVal, prob;
			if prob > maxProb:
				label = classVal;
				maxProb = prob;
		return label, maxProb;

############### REGRESSION CLASSIFIER DOES NOT WORK FOR BOUNDED TARGET VALUE #######
############### PENDING MODIFICATION TO BETA-REGRESSION ############################
class LinRegress(object):
	def __init__(self):
		self.theta = 0; #predictor weight(s) for factors

	def computeCost(self, rows, theta):
		'computes square error between prediction and expected value'
		hyp = self.X.dot(theta);
		sqError = (hyp - self.Y)**2;

		return (1.0 / (2 * rows)) * sum(sqError);

	def gradientDescent(self, rows, steps, theta):
		'minimizes error between prediction and expected value, descent is optimized by use of adaptive alpha'
		curError = 1 + max(self.Y); #assign the expected error some large value
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

	def train(self, steps, X, y):
		self.Y = y;
		rows, cols = X.shape;
		self.X = ones((rows, cols+1)); #default bias is 1
		self.theta = ones(shape=(cols+1,1));

		self.X[:, 1:] = X;
		self.gradientDescent(rows, steps, self.theta);

	def predict(self, dat):
		'returns a prediction for the datum.'

		if type(self.theta) == int:
			print '\n> Predictor is not trained Yet. Please train before making any predictions\n';

			return None;
		else:
			datum = ones((1, self.theta.shape[0])); #insert bias into input data
			datum[:, 1:] = dat;

			return datum.dot(self.theta)[0][0]; #compute prediction

#X = data[:, :cols-1]; #first columns are features for the outcome
#self.Y = data[:, cols-1]; #last column is outcome
#self.Y.shape = [rows, 1]; #properly define the shape of y
#self.Y = self.Y;

#if checkScales(X, deg): #check if data needs to be normalized
#	X = normalize(X);

#dat = fromiter(dat.split(self.delim), dtype = float); #split dat string to np.array for prediction
##################################################################