###Accurp-Engine

About
-----
This program is a set of modules packaged into an object for predicting target values using linear regression. The regression algorithm can handle single or multiple features for training. Other than an adaptive alpha being used for training, and the method "checkScales" which determines when data requires normalization, this is a classical regression algorithm. 

How to Use
----------
Import the "Predictor" file as a regular python library, then create a LinRegress object. The construct at the moment is very fundamental so no input parameters are needed. After an object is created, assuming this is the first use, you must train the learner. 

To train call the train() by passing 4 parameters:
* steps - number of descents the learner takes against the training data for example 400.
* deg - exponential degree difference training data is allowed to have. For default case just pass the value 2.
* trainSRC - string of file name containing training data. File extension needs to be included in the string
* delim - delimiter that separates values within the training data. For example if the file extension is csv, then the data is assumed to be seprated by a comma. Then the value for delim parameter is ','.

Module does not return anything, instead it stores the trained coefficients as the theta attribute. To checkout the trained coefficients use the dot operator with the object. Please note, file for training data for a regression algorithm should only contain numerical values. 

after training call the predict() by passing 1 parameter:
* dat - string with the same delimiter character for separating values as the training data.
* Method returns a numerical value which is the predicted result. 


Testing the Engine
------------------
The test.py gives a rundown in how the code can be used. Simply execute it and it will train use linear regression to predict housing prices from a given data set. 
