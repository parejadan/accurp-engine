#!/usr/bin/python
from numpy import loadtxt, append, array;
from predictors import NaiveBayes;
from pickle import load, dump;
from sys import argv;
import os, urllib, json, time;

os_brack = '/'; #directory separator for os engine is running on
categories = [1250, 4500, 8000, 16000, 35000];

def loadData(datSRC, path, delim, typ):
	return loadtxt(path + os_brack + datSRC, delimiter=delim, dtype = typ);

def saveRawInput(rawInput, dst = '.'):
	'save user input for reuse as future training data'
	datName = dst + os_brack + 'collected-data.csv';
	f = open(datName, 'a');
	f.write(rawInput);
	f.close();

def googleSearch(search):
	'google search api'
	query = "http://ajax.googleapis.com/ajax/services/search/web?v=1.0&q=%s";#search api? free to use
	results = urllib.urlopen( query % (search) );
	json_res = json.loads( results.read() );
	return int(json_res['responseData']['cursor']['estimatedResultCount']); #returns number of estimated search results

def loadClassifier(objFile = 'classifier.pickle', path = '.'):
	'loads an already trained classifier. If no classifier is passed as \
	parameter then it uses the default path and name'
	if os.path.isfile( (path + os_brack + objFile) ):
		return load( open(path + os_brack + objFile) );
	else:
		#print path + os_brack + objFile;
		print '\n[!NO CLASSIFIFER IS SAVED YET!]\n'
		return None;

def makeClassifier(data):
	'trains a classifier with a given training data'
	nv = NaiveBayes();
	nv.summarizeByClass(data); #train classififer
	#f = open( 'classifier.pickle', 'wb');
	#dump(nv, f); #save trained classififer as python pickle
	#f.close();
	return nv; #return trained classififer

def discretizeFreq(frequency, cats = categories):
	'categorizes result hits from a google saerch query \
	if no categories are passed, it uses the default defined'
	for i in len(cats):
		if frequency < cats[i]:
			return i+1;
	return len(cats)+1;

def discretizeTarget(data, threshold):
	rows, cols = data.shape;
	for i in range(rows):
		if (data[i][-1] >= threshold): data[i][-1] = 1;
		else: data[i][-1] = 0;
	return data;

def discretizeLang(lang, languages):
	index = 1;
	for l in languages:
		if l == lang:
			return index;
		index += 1;
	return None;

def testClassifier(examples, trnprt, tstprt, size):
	trnset = examples[:trnprt];
	tstset = examples[tstprt:];
	classifier = makeClassifier(trnset);

	falses = 0.0;
	for e in tstset:
		label, prob = classifier.predict( e[:-1] );
		prob *= 100;

		#print 'expected output: %d\t|predicted output: %d\t|confidence lvl: %f' % (label, e[-1], prob);
		if (label != e[-1]):
			falses += 1;

	print '\n>> Training data dimensions: %d' % ( len(examples[0][:-1]) )
	print '>> Prediction accuracy is: %f' % (1 - falses/(size-trnprt))
	print '>> For %d training examples and %d testing examples' % (len(trnset), len(tstset))
	print '>> Overall data size is %d\n\n' % size;


def main():

	#classifier = loadClassifier();
	#if classifier is None:
	datOrder = 'random'
	for i in range(1,4):
		path = '..' + os_brack +'training-data' + os_brack + '%d-dimensional' % (i+1);
		#print path
		examples = loadData('data-%s.csv' % datOrder, path, ',', float);
		examples = discretizeTarget(examples, 0.68); #60% tollerance level for coherency test

		size = len(examples);
		trnprt = size/4.0;
		tstprt = trnprt;
		testClassifier(examples, trnprt, tstprt, size);
	print 'data organization is %s\n' % datOrder;



#if-condition executes main functions when file used directly
if __name__ == '__main__':
	main();