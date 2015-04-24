from Predictor import LinRegress

#training data has 3 columns. The first two columns are house size in square meters, and bedrooms
#the last column is the price of the house for the given features.
#when making predictions, for this training data we only need to pass 2 values.

print '\n>> Defining paramaters and training data..';
steps = 400;
trainSRC = 'house-pricing.csv'; #training data
delim = ',';
deg = 2; #exponential degree difference data is allowed to have
testDat = '55400,32';

print '>> Creating object for API..';
obj = LinRegress(); #create object instance

print '>> Training learner..';
obj.train(steps, deg, trainSRC, delim);

print '>> Making Prediction..';
prediction = obj.predict(testDat);

print '\n# The predicted value of a house of 55400 sq-ft and 32 bedrooms is: $', prediction, '\n';