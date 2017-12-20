#read from file 
import io
import os 
import random 
import collections
import sys

def print_acc(accuracies):
	overall_accuracy = accuracies[0]
	lang_accuracies = accuracies[1]
	print('Overall Accuracy: ' + str(overall_accuracy))
	print()
	print('Accuracy per Language: ')
	for key, value in lang_accuracies.items():
		print(str(key) + ': ' + str(value))


def print_metrics(metrics):
	micro = metrics['micro']
	macro = metrics['macro']

	print('Micro averaged Precision: ' + str(micro[0]))
	print('Micro averaged Rank: ' + str(micro[1]))
	print('Micro averaged F-measure: ' + str(micro[2]))
	print()
	print('Macro averaged Precision: ' + str(macro[0]))
	print('Macro averaged Rank: ' + str(macro[1]))
	print('Macro averaged F-measure: ' + str(macro[2]))


def accuracy(ts):
	inverted_test_set = {}
	accuracies = {}
	overall_accuracy = 0
	test_sentence_count = len(test_set)


	for lang in languages:
		accuracies[lang] = 0
	for sentence,lang in test_set:
		if lang not in inverted_test_set:
			inverted_test_set[lang] = [sentence]
		else:
			inverted_test_set[lang].append(sentence)
			
	for correct_lang, guessed_lang in ts:
		if correct_lang == guessed_lang: #highest probability = predicted language
				accuracies[correct_lang] += 1.0
				overall_accuracy += 1.0

	overall_accuracy/=test_sentence_count
	overall_accuracy*=100
	for key in accuracies:
			accuracies[key]=(accuracies[key]/len(inverted_test_set[key]))*100

	return (overall_accuracy, accuracies)


def metrics(orig_guessed_tuples):
	matrices = {}
	
	language_count = len(languages)
	
	pisum = 0
	rosum = 0
	Fsum = 0
	
		
	for l in languages:
		for orig, guessed in orig_guessed_tuples:
			TP, FP, FN = 0,0,0
			if orig==l and guessed==l:
				TP = 1
			elif orig==l and guessed!=l:	
				FN = 1
			elif orig!=l and guessed==l:
				FP = 1
				
			if l in matrices:
				matrices[l] = (matrices[l][0] + TP, matrices[l][1] + FP, matrices[l][2] + FN)
			else:
				matrices[l] = (TP, FP, FN)
		

	tpsum = 0
	fpsum = 0
	fnsum = 0
	
	pisum = 0
	rosum = 0
	Fsum = 0
	
	for key in matrices:
	
		TP, FP, FN = matrices[key]
		
		pii, roi = 0, 0
		
		if (TP + FP) !=0:
			pii = 1.0 * TP / (TP + FP) 
		if (TP + FN) !=0:
			roi = 1.0 * TP / (TP + FN) 
		
		pisum += pii
		rosum += roi
		if not (pii==0 and roi==0):
			Fsum += (2.0 * pii * roi) / (pii + roi)
		
		
		tpsum+=TP
		fpsum+=FP
		fnsum+=FN
	
	pi = 1.0 * tpsum / (tpsum + fpsum) 
	ro = 1.0 * tpsum / (tpsum + fnsum) 
	F = (2.0 * pi * ro) / (pi + ro)
	
	pisum /= language_count
	rosum /= language_count
	Fsum /= language_count
	
	return {'micro': (pi, ro, F), 'macro': (pisum, rosum, Fsum)}



sentences = [] #list of tuples of (sentence, language) -> we can shuffle list, not map
languages = set() #languages

count_map = {} 
f = io.open('corpus.txt', 'r', encoding='utf-16') #read from file in utf-16 format

corpus_utf16 = f.readlines() #read line by line 

i=0
corpus = []
for line in corpus_utf16:
	#get the last token of the sentence as the language, the others as the sentence
	sentence, lang = line.rsplit(None,1)
	sentence, lang = sentence.strip().replace(' ', ''), lang.strip()
	#in our analysis, no space character is included
	sentences.append((sentence, lang))
	languages.add(lang.strip())


#randomly split data to 90% training and 10% testing

sentence_count = len(sentences)
random.shuffle(sentences)
training_set = sentences[:int(sentence_count/10 * 9)]
test_set = sentences[int(sentence_count * 9/10):]

training_dict = {}
vocab_sizes = {}

#make sentences of a language into one big string
for value,key in training_set:
	if key not in training_dict:
		training_dict[key] = value
	else:
		training_dict[key] += value


#vocab size of language l
for l in training_dict:
	vocab_sizes[l] = len(set(training_dict[l]))


if sys.argv[1] == 'naive_bayes' :
	letter_probabilities = {}
	for key in training_dict:
		train_sentence = training_dict[key]
		total_chars = len(training_dict[key]) #total number of characters in training set
		chars = []
		unique_chars = vocab_sizes[key]
		for ch in train_sentence:
			if ch not in chars:
				#laplace smoothing
				if key not in letter_probabilities:
					
					letter_probabilities[key] = {ch: ((train_sentence.count(ch)+1.0)/(unique_chars + total_chars))}
				else:
					letter_probabilities[key][ch] = ((train_sentence.count(ch)+1.0)/(unique_chars + total_chars))
				chars.append(ch)

		#print vocab_sizes[key], len(letter_probabilities[key])


	result = {}

	language_count = len(languages)
	for test_tuple in test_set:
		test_sentence, _ = test_tuple
		sum_probabilities = 0
		lang_probs = {}#probability that sentence is that language
		for l in languages:
			total_chars = len(training_dict[l]) #total number of characters in training set
			unique_chars = vocab_sizes[l]
			
			probs = letter_probabilities[l] #letter probabilities of that language
			for ci in test_sentence:
				if ci in probs:
					sum_probabilities += probs[ci] #sum(1..n) P(ci|l)
				else:
					sum_probabilities += (1.0/(total_chars+unique_chars+1))
					
			sum_probabilities*=1.0/language_count #P(l)
			lang_probs[l] = sum_probabilities
		result[test_tuple] = sorted(lang_probs.items(), key=lambda x:x[1], reverse=True)

	ts = []
	for test_tuple in result:
		correct_lang = test_tuple[1]
		guessed_lang = result[test_tuple][0][0] 
		ts.append((correct_lang, guessed_lang))
	print_acc(accuracy(ts))
	print()
	print_metrics(metrics(ts))
	

elif sys.argv[1] == 'unigram_svm' or sys.argv[1] == 'super_svm':
	if len(sys.argv) < 3:
		print ('Missing argument!')
	else:
		#SVM
		vocab = {}
		langs = {}
		i=1
		for key in languages:
			langs[key] = i
			i+=1
		i=1
		for sentence, lang in sentences:
			for l in sentence:
				if l not in vocab.values():
					vocab[i] = l
					i+=1
		i=1
		
		vocab = {v: k for k, v in vocab.items()}
		

		svm_sentence_dict = {}

		if sys.argv[1] == 'unigram_svm':
			for sentence, lang in sentences:
				st = str(langs[lang])
				ch_counts = {}
				for ch in sentence:
					if vocab[ch] not in ch_counts:
						ch_counts[vocab[ch]] = sentence.count(ch)	 
				ch_counts = collections.OrderedDict(sorted(ch_counts.items()))
				for key in ch_counts:
					st+=' ' + str(key) + ':' + str(1)
				svm_sentence_dict[sentence] = st #for that sentence, lang and f:v info

		elif sys.argv[1] == 'super_svm':
			bigram = {}
			for sentence, lang in sentences:
				for (f, s) in zip(sentence[0::2], sentence[1::2]):
					if str(f+s) not in bigram.values():
						bigram[i] = str(f+s)
						i+=1
			
			bigram = {v: k for k, v in bigram.items()}
			
			for sentence, lang in sentences:
				st = str(langs[lang])
				ch_counts = {}
				bigram_counts = {}
				for ch in sentence:
					if vocab[ch] not in ch_counts:
						ch_counts[vocab[ch]] = sentence.count(ch)
				for (f, s) in zip(sentence[0::2], sentence[1::2]):
					if bigram[str(f+s)] not in bigram_counts:
						bigram_counts[bigram[str(f+s)]] = sentence.count(str(f+s))		

				ch_counts = collections.OrderedDict(sorted(ch_counts.items()))
				bigram_counts = collections.OrderedDict(sorted(bigram_counts.items()))
		
				for key in ch_counts:
					st+=' ' + str(key) + ':' + str(1)
				for key in ch_counts:
					st+=' ' + str(key + len(vocab)) + ':' + str(ch_counts[key])
				for key in bigram_counts:
					st+=' ' + str(key + 2*len(vocab)) + ':' + str(bigram_counts[key])
			#	 st+=' ' + str(len(bigram) + 2*len(vocab) + 1) + ':' + str(sum(1 for c in sentence if c.isupper()))
				svm_sentence_dict[sentence] = st #for that sentence, lang and f:v info
		else:
			print('Wrong input!')

		train_strings = []
		test_strings = []

		f_train = open("svm_train.txt","w") 
		f_test = open("svm_test.txt","w") 


		for sentence, lang in training_set:
			f_train.write(svm_sentence_dict[sentence] + '\n')
		for sentence, lang in test_set:
			f_test.write(svm_sentence_dict[sentence] + '\n')
		f_train.close()
		f_test.close()

		path = str(sys.argv[2])

		os.system(path + '/svm_multiclass_learn -c 1.0 svm_train.txt model.txt')
		os.system(path + '/svm_multiclass_classify svm_test.txt model.txt output.txt')



		out = open("output.txt","r")
		test = open("svm_test.txt", "r")
		guessed_probs = out.readlines()
		orig_sent = test.readlines()
		orig_guessed_tuples = []
		langs = {v: k for k, v in langs.items()}
		for s in guessed_probs:
			guessed = langs[int(s.split(' ')[0])]
			orig = langs[int(orig_sent[guessed_probs.index(s)].split(' ')[0])]
			orig_guessed_tuples.append((orig, guessed))
		print()
		print_acc(accuracy(orig_guessed_tuples))
		print()
		print_metrics(metrics(orig_guessed_tuples))
