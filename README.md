#Accurp-Engine

Intro
-----
This project takes a machine learning approach to evaluating translation application. The objective for this is to provide users of automated translation applications some level of confidence when using them. Currently, the "engine" relies on a single machine learning algorithm (Naive Bayes), but future updates will include additional learners which can be located within the "Predictors.py" file. The project's goal is to obtain a deeper understanding of machine learning (hence coding algorithms from scratch) and exercise python hacking skills, all while talking a practical problem. If you find any type issues with the software please feel free to contact me, any feedback is appreciate. 

Overview
--------
The original raw data was collected by a group of classmates from my AI-2 class. Which was then shared with the rest of the class for the use as a term-project for the course. I studied the raw data and defined a few features to then feed it to a learner. 

NOTE: The following description for the training data is my interpretation from what was explain to me. There were some fuzzy areas I had to clarify on own which lead to some alterations. Also, the training data on this repository is a discretized version of it.

So what is the data exactly? In a nutshell the data is a collection of everyday conversational sentences. For example, the type of sentences one exchanges with a waiter when ordering food at a restaurant, or asking a stranger for directions. We focus on these low complexity sentences so to assure a high quantity and variety in data throughout many languages, while still targeting a challenging problem without going into much detail in language analysis. 

 Once a set of sentences is defined they are translated to a foreign language using an automated service (e.g. Google Translate). The translated text is then read by someone who can understand the foreign language to determine if the original meaning remains or not (coherency test). If the translated sentence passes the coherency test it gets a score of 1, otherwise 0. Translated sentences with a score of 1 are then fed to a web-search engine  to obtain a frequency score, which is simply the number of hits a search-result returns. By doing this we attempt to determine how common a sentence's context is within the foreign language (sentences with coherency score 0 get a frequency score of 0). Finally, the text goes through a reverse translation to obtain a equilibrium score (value range 0-1, sentences with coherency score of 0 get a equilibrium score of 0). Equilibrium scores are evaluated by determining character difference between the source text with the reverse translated text. 

Data Description
----------------
Momentarily the training data is strucutred in the following format:

| Character Length | Discretized Phrase Frequency | Equilibrium Score |
|---------------------------------------------------------------------|
| 47 | 6 | 0.88 |
| 102 | 5 | 0.16 |
| 14 | 6 | 0.86 |
-----------------
<table>
	<tr>
		<td> Character Length </td>
		<td> Discretized Phrase Frequency </td>
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

Testing the Engine
------------------
Pending
