#Accurp-Engine

Intro
-----
This project takes a machine learning approach to evaluating translation applications. The code's objective is to provide users of automated translation app some level of confidence when using them, without needing significant understanding of a target language. Currently the "engine" relies on a Naive Bayes classification, but future updates will include other models for use. The project's aim is to understand machine learning concepts better and exercise python hacking skills, all while tackling a practical problem. If you find any type issues with the code please feel free to contact me, any feedback is appreciate. 

Overview
--------
The original raw data was collected by a group of classmates from my AI-2 class. Which was then shared with the rest of the class for the use as a unit-project for the course. I studied the raw data and defined a few features to then feed it to a learner. 

NOTE: The following description for the training data is my interpretation from what was explain to me. There were some fuzzy areas I had to clarify on own which lead to some alterations. Also, the training data on this repository is a discretized version of it.

So what is the data exactly? In a nutshell the data is a collection of everyday conversational sentences. For example, the type of sentences one exchanges with a waiter when ordering food at a restaurant, or asking a stranger for directions. We focus on these low complexity sentences so to assure a high quantity and variety in the data for varying languages.

Once a set of sentences are  chosen they are translated to a foreign language using an automated service (e.g. Google Translate). The translated text is then read by someone who understands the foreign language and determines the source text is logically equivalent to the translated text (coherency test). 

We then attempt to determine how common a translated text is within its respective foreign language by feeding it to a web-search engine to get a frequency score. The score is then discretized by checking which of 6 ranges it falls under. A category 1 range is considered very low, category 6 range is considered very high. 

Finally we do a reverse translation on the text to obtain a equilibrium score. This is to see how dependent a good translation is on word-to-word equivalence. Values for a equilibrium score ranges from 0-1. 


Data Description
----------------
Momentarily the training data is strucutred in the following format:

<table>
	<tr>
		<td> Character  Length </td>
		<td> Discretized Sentence Frequency </td>
		<!--<td> Source Language ID </td>-->
		<!--<td> Target Language ID </td>-->
		<td> Equilibrium Score </td>
	</tr>
	<tr>
		<td> 47 </td>
		<td> 6 </td>
		<td> 0.88 </td>
	</tr>
	<tr>
		<td> 102 </td>
		<td> 5 </td>
		<td> 0.16 </td>
	</tr>
	<tr>
		<td> 14 </td>
		<td> 6 </td>
		<td> 0.86 </td>
	</tr>

</table>

* Character length considers is self explanatory, only that it also considers spaces but excludes non-alphabetic characters
* a sentence's frequency is discretized into 6 possible ranges it can fall under; 1 being very low 6 being very high
* Equilibrium Score is the percentage of the original text that remains after a reverse translation. For example consider the srcText "I like cats", where L1 = English and L2 = some-arbitrary-language. If srcTxt is translated from L1 to L2 to produce dstText, passes the coherency 
test, and then reversed translated (L2 to L1) which outputs “I love cats”, the equilibrium score is 7/11 ~ 0.63

Predictions
-----------
Equilibrium scores are considered the target value for prediciton classifications. But due to the low training data 

Testing the Engine
------------------
Pending
