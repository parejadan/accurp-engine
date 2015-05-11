#!/usr/bin/python
from numpy import loadtxt, append, array, arange;
from predictors import NaiveBayes;
from pickle import load, dump;
from sys import argv;
import os, urllib, json, time;

os_brack = '/'; #directory separator for os engine is running on

def loadData(datSRC, path, delim, typ):
	return loadtxt(path + os_brack + datSRC, delimiter=delim, dtype = typ);

def saveData(data, name, svType,dst = '.'):
	f = open( dst + os_brack + name , svType);
	f.write(data);
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
	f = open( 'classifier.pickle', 'wb');
	dump(nv, f); #save trained classififer as python pickle
	f.close();
	return nv; #return trained classififer

def discretizeFreq(frequency, cats = [1250, 4500, 8000, 16000, 35000]):
	'categorizes result hits from a google saerch query \
	if no categories are passed, it uses the default defined'
	#print cats;
	for i in range( len(cats) ):
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
	avgLvl = 0.0;
	for e in tstset:
		label, prob = classifier.predict( e[:-1] );
		avgLvl += prob * 100;

		#print 'expected output: %d\t|predicted output: %d\t|confidence lvl: %f' % (label, e[-1], prob);
		if (label != e[-1]):
			falses += 1;

	#print '\n>> Training data dimensions: %d' % ( len(examples[0][:-1]) )
	#print '>> Prediction accuracy is: %f' % (1 - falses/(size-trnprt))
	#print '>> For %d training examples and %d testing examples' % (len(trnset), len(tstset))
	#print '>> Overall data size is %d\n\n' % size;

	return (1 - falses/(size-trnprt)), (avgLvl/len(tstset));

def getWordCountDif(txt1, txt2, delim =' '):
	return abs( len(txt1.split(delim)) - len(txt2.split(delim)) );

def main():

	#setup classifier
	thresh = 0.65;
	classifier = loadClassifier();
	if classifier is None:
		path = '..' + os_brack + 'training-data' + os_brack + '3-dimensional';
		examples = loadData('data-random.csv' , path, ',', float);
		examples = discretizeTarget(examples, thresh);
		trnprt = len(examples)/3;
		trnset = examples[:trnprt];
		classifier = makeClassifier(trnset);

	#junk -> name of this python file, not needed but automatically passed by interpreter 
	#srcTxt -> source text user translated
	#dstTxt -> translated text
	#srcLng -> language of source txt
	#dstTtxt -> language source text was translated to
	#srcTxt, dstTxt, srcLng, dstLng = loadData(argv[1], '../input', '\n', str); #use this interface for basic testing
	junk, srcTxt, dstTxt, srcLng, dstLng = argv; #use this interface for production

	#setup input data
	frequency = discretizeFreq( googleSearch(dstTxt) );
	wordDif = getWordCountDif(srcTxt, dstTxt);
	txtlen = len(srcTxt);

	#make prediction
	label, prob = classifier.predict( [txtlen, frequency, wordDif] );
	prediction = ''; prob *= 100; #convert prob to a percentage
	if label == 1: prediction = 'good';
	else: prediction = 'bad';

	#display prediction
	print '\nPredicted translation type: %s' % prediction;
	print 'Prediction confidence percentage: %f' % prob;
	print 'Classifier\'s word-to-word equivalence threshold percentage %f\n' % (thresh * 100);

if __name__ == '__main__':
	main();


#################Code reserved for classifier intensive testing################## 
#	datOrder = 'random'
#	accurDat = '';
#	confiDat = '';
#	for j in arange(0.5, 1, 0.1): #threashold increases
#		for k in arange(2.0,6): #training data decreases
#			accurDat += '%f %f ' % ((1/k), j);
#			confiDat += '%f %f ' % ((1/k), j);
#			for i in range(1,4): #dimensions increase
#				path = '..' + os_brack +'training-data' + os_brack + '%d-dimensional' % (i+1);
#				examples = loadData('data-%s.csv' % datOrder, path, ',', float);
#				examples = discretizeTarget(examples, j);
#				size = len(examples);
#				trnprt = size/k;
#				tstprt = trnprt;
#				accuracy, confidence = testClassifier(examples, trnprt, tstprt, size);
#				accurDat += '%f ' % (accuracy);
#				confiDat += '%f ' % (confidence);
#			accurDat += '\n';
#			confiDat += '\n';
#	saveData(accurDat, ('acur-%s.dat' % datOrder), 'w');
#	saveData(confiDat, ('conf-%s.dat' % datOrder), 'w');
#	print 'data organization is %s\n' % datOrder;