#!/usr/bin/python
from numpy import loadtxt, append, array;
from Predictors import NaiveBayes;
from math import log10;

def loadData(datSRC, delim):
	return loadtxt(datSRC, delimiter=delim);

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

#training data for classifier (actual translation data that was collected)
print '\n>> Loading translation data for training classifier...'
data = loadData('eng-to-rus.csv', ',');

print '>> training classifier with data...'
nv = NaiveBayes();
nv.summarizeByClass(data); #trained aspect is saved within the nv object

print '>> Ready for making predictions...'
test_input_one = [102,5];
prediction = '';
label, prob = nv.predict(test_input_one);
if label == 1:
	prediction = 'good';
else:
	prediction = 'bad';
print '\n>> Predicted translation accuracy for a string of 102 chars and a frequency class of 5 is: ';
print '>> ', prediction, 'translation with a', prob*100, '%  confidence probability';


test_input_two = [9,6];
prediction = '';
label, prob = nv.predict(test_input_two);
if label == 1:
	prediction = 'good';
else:
	prediction = 'bad';
print '\n>> Predicted translation accuracy for a string of 9 chars and a frequency class of 6 is: ';
print '>> ', prediction, 'translation with a', prob*100, '%  confidence probability';

#self.rows, self.cols = data.shape;

#X = data[:, :cols-1]; #first columns are features for the outcome
#self.Y = data[:, cols-1]; #last column is outcome
#self.Y.shape = [rows, 1]; #properly define the shape of y
#self.Y = self.Y;

#if checkScales(X, deg): #check if data needs to be normalized
#	X = normalize(X);

#dat = fromiter(dat.split(self.delim), dtype = float); #split dat string to np.array for prediction
##################################################################
