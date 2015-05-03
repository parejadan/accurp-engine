#!/usr/bin/python
from numpy import loadtxt, append, array;
from Predictors import NaiveBayes;
from pickle import load, dump;
from sys import argv;
import os, urllib, json, time;

os_brack = '/'; #directory separator for os engine is running on

def loadData(datSRC, delim, typ):
	return loadtxt(datSRC, delimiter=delim, dtype = typ);

def googleSearch(search):
	'work around for google search api - i think...'

	query = "http://ajax.googleapis.com/ajax/services/search/web?v=1.0&q=%s";#search api? free to use
	results = urllib.urlopen( query % (search) );
	json_res = json.loads( results.read() );
	return int(json_res['responseData']['cursor']['estimatedResultCount']); #returns number of estimated search results

def loadClassifier(objFile = 'classifier.pickle', path = '.'):

	if os.path.isfile( (path + os_brack + objFile) ):
		return load( open(path + os_brack + objFile) );
	else:
		#print path + os_brack + objFile;
		print '\n[!NO CLASSIFIFER IS SAVED YET!]\n'
		return None;

def getClassifier(data, languages):
	nv = NaiveBayes();
	nv.summarizeByClass(data); #train classififer
	nv.db['languages'] = languages;
	f = open( 'classifier.pickle', 'wb');
	dump(nv, f); #save trained classififer as python pickle
	return nv; #return trained classififer

def descretizeFreq(frequency):
	if frequency < 1250: return 1;
	if frequency < 4500: return 2;
	if frequency < 8000: return 3;
	if frequency < 16000: return 4;
	if frequency < 35000: return 5;
	return 6;

def descretizeTarget(data, threshold):
	rows, cols = data.shape;
	for i in range(rows):
		if (data[i][-1] >= threshold): data[i][-1] = 1;
		else: data[i][-1] = 0;
	return data;

def descretizeLang(lang, languages):
	index = 1;
	for l in languages:
		if l == lang:
			return index;
		index += 1;
	return None;

def main():

	classifier = loadClassifier();
	if classifier is None:
		examples = loadData('resources/training-examples.csv', ',', float);
		examples = descretizeTarget(examples, 0.68);
		classifier = getClassifier(examples, loadData('resources/languages.txt', '\n', str));

	#argv[1] -> name of file that contains user input (should only contain 3 lines)
	#srcTxt -> input string user needed to translate - 1st line
	#dstTxt -> translated text online app provided - 2nd line
	srcTxt, dstTxt, srcLan, dstLan = loadData(argv[1], '\n', str); #use this interface for testing purposes
	#junk, srcTxt, dstTxt, srcLan, dstLan = arvg; #use this interface for production use
	src_id = descretizeLang(srcTxt, classifier.db['languages']);
	dst_id = descretizeLang(dstTxt, classifier.db['languages']);
	frequency = descretizeFreq( googleSearch(dstTxt) );
	textSize = len(srcTxt);

	label, prob = classifier.predict( [textSize, frequency, src_id, dst_id] );
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

