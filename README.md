_For detailed explanation, please look at Report.pdf_

# Language Identification System

This is a language identification system in Python which uses Discriminating between Similar Languages (DSL) Shared Task 2015 corpus. The analysis is based on character unigrams, meaning that the identification of a language is based on letter frequencies. The assumption is that each language either has unique characters or use the characters with unique probabilities.
In the corpus, there are 13 languages, each with 2000 sentences, so the corpus has a balanced distribution with 26000 sentences in total. The languages in question are Bulgarian, Bosnian, Czech, Argentine Spanish, Peninsular Spanish, Croatian, Indonesian, Macedonian, Malay, Brazilian Portuguese, European Portuguese, Slovak and Serbian.

Two models are tried on this corpus to solve this problem: Generative Modeling (Naive Bayes) and Discriminative Modeling (SVM). We implemented the Bayesian approach ourselves and used Cornell’s SVM multiclass library. 

We evaluated the approaches used above with accuracy, recall, precision, F-measure.

## Program Execution

The program can be run in three modes: naive Bayes, unigram SVM and super SVM. Note that the program will not run without the mode argument.

In order to run the program in naive bayes mode, go to the directory of the python file and type:
> python identifier.py naive\_bayes

In order to run the program in SVM modes, path to the svm-multiclass
directory must also be given as an argument. 

For unigram SVM mode, type:
> python identifier.py unigram\_svm path/to/svm-multiclass 

(In unigram SVM, existence of a letter in a sentence is a feature).

For super SVM mode, type:
> python identifier.py super\_svm path/to/svm-multiclass

(In super SVM, existence and count of characters and bigram letter frequencies are taken as features).

As output, the overall accuracy, accuracies for each language, microaveraged measures and macroaveraged measures are shown.
