#!/usr/bin/python
from numpy import loadtxt, append, array;
from Predictors import NaiveBayes;
from pickle import load, dump;
from sys import argv;
import os, urllib, json;

os_bracket = '/';

def loadData(datSRC, delim, typ):
	return loadtxt(datSRC, delimiter=delim, dtype = typ);

def googleSearch(search):
	'work around for google search api - i think...'
	query = "http://ajax.googleapis.com/ajax/services/search/web?v=1.0&q=%s";#search api? free to use
	results = urllib.urlopen( query % (search) );
	json_res = json.loads( results.read() );
	return int(json_res['responseData']['cursor']['estimatedResultCount']); #returns number of estimated search results

def loadClassifier(objFile, path = 'training-data'):
	'loads an already saved classifier, path by default os training data directory'
	if os.path.isfile( (path + os_bracket + objFile) ):
		return NaiveBayes( load( open(path + os_bracket + objFile) ) );
	else:
		#print path + os_bracket + objFile;
		print '\n[!NO CLASSIFIFER IS SAVED YET!]\n'
		return None;

def getClassifier(trainDat = 'train-dat.csv', delim = ',', path = 'training-data'):
	'Trains a new classifier, if user does not want ao specific training data, use default'
	data = loadData( (path + os_bracket + trainDat) , delim, float); #load training data
	nv = NaiveBayes();
	nv.summarizeByClass(data); #train classififer
	f = open( (path + os_bracket +'classifier.pickle'), 'wb');
	dump(nv.summaries, f); #save trained classififer as python pickle
	return nv; #return trained classififer

def discretizedFreq(frequency):
	if frequency < 1250: return 1;
	if frequency < 4500: return 2;
	if frequency < 8000: return 3;
	if frequency < 16000: return 4;
	if frequency < 35000: return 5;
	return 6;

def main():
	#argv[1] -> name of file that contains user input (should only contain 3 lines)
	#srcTxt -> input string user needed to translate - 1st line
	#dstTxt -> translated text online app provided - 2nd line
	srcTxt, dstTxt = loadData(argv[1], '\n', str);
	classifier = loadClassifier('classifier.pickle');
	if classifier is None:
		classifier = getClassifier('eng-to-rus.csv');

	frequency = discretizedFreq( googleSearch(dstTxt) );
	textSize = len(srcTxt);

	label, prob = classifier.predict( [textSize, frequency] );
	prediction = ''; prob *= 100; #convert prob into a percentage
	if label == 1: prediction = 'good';
	else: prediction = 'bad';

	print '\n>> source text: %s' % srcTxt;
	print '>> translated text: %s' % dstTxt;
	print '>> Predicted translation type: %s' % prediction;
	print '>> Prediction confidence percentage: %f percent\n' % prob;

#if-condition executes main functions when file used directly
if __name__ == '__main__':
	main();

