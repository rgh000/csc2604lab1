import math
import operator
import random
import re
import sets
import numpy as np

#STEP 1
import nltk
from nltk.corpus import brown

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr


#STEP 6. Table RG65 with lists P and S as defined in the lab.
P = [
('cord',	'smile'),('rooster',	'voyage'),('noon',	'string'),('fruit',	'furnace'),('autograph',	'shore'),('automobile',	'wizard'),('mound',	'stove'),('grin',	'implement'),('asylum',	'fruit'),('asylum',	'monk'),('graveyard',	'madhouse'),('glass',	'magician'),('boy',	'rooster'),('cushion',	'jewel'),('monk',	'slave'),('asylum',	'cemetery'),('coast',	'forest'),('grin',	'lad'),('shore',	'woodland'),('monk',	'oracle'),('boy',	'sage'),('automobile',	'cushion'),('mound',	'shore'),('lad',	'wizard'),('forest',	'graveyard'),('food',	'rooster'),('cemetery',	'woodland'),('shore',	'voyage'),('bird',	'woodland'),('coast',	'hill'),('furnace',	'implement'),('crane',	'rooster'),('hill',	'woodland'),('car',	'journey'),('cemetery',	'mound'),('glass',	'jewel'),('magician',	'oracle'),('crane',	'implement'),('brother',	'lad'),('sage',	'wizard'),('oracle',	'sage'),('bird',	'crane'),('bird',	'cock'),('food',	'fruit'),('brother',	'monk'),('asylum',	'madhouse'),('furnace',	'stove'),('magician',	'wizard'),('hill',	'mound'),('cord',	'string'),('glass',	'tumbler'),('grin',	'smile'),('serf',	'slave'),('journey',	'voyage'),('autograph',	'signature'),('coast',	'shore'),('forest',	'woodland'),('implement',	'tool'),('cock',	'rooster'),('boy',	'lad'),('cushion',	'pillow'),('cemetery',	'graveyard'),('automobile',	'car'),('midday',	'noon'),('gem',	'jewel')
]

S = [0.02,
     0.04,
     0.04,
     0.05,
     0.06,
     0.11,
     0.14,
     0.18,
     0.19,
     0.39,
     0.42,
     0.44,
     0.44,
     0.45,
     0.57,
     0.79,
     0.85,
     0.88,
     0.9,
     0.91,
     0.96,
     0.97,
     0.97,
     0.99,
     1,
     1.09,
     1.18,
     1.22,
     1.24,
     1.26,
     1.37,
     1.41,
     1.48,
     1.55,
     1.69,
     1.78,
     1.82,
     2.37,
     2.41,
     2.46,
     2.61,
     2.63,
     2.63,
     2.69,
     2.74,
     3.04,
     3.11,
     3.21,
     3.29,
     3.41,
     3.45,
     3.46,
     3.46,
     3.58,
     3.59,
     3.6,
     3.65,
     3.66,
     3.68,
     3.82,
     3.84,
     3.88,
     3.92,
     3.94,
     3.94]


#STEP 2 5000 most frequent words stored using uni and unigram
#unigram['example'] returns the index of the tuple ('example', count) in
#the uni list, where count is the unigram count of 'example'.
unigram = dict()
bigram = dict()
words = brown.words()
for w in words:
	w = re.sub('[^A-Za-z]+', '', w)
	if w != '':
		if not unigram.has_key(w.lower()):
			unigram[w.lower()] = 1
		else:
			unigram[w.lower()] += 1


old = unigram
uni = sorted(unigram.items(), key=operator.itemgetter(1), reverse=True)
sum = 0.0
for i,j in uni:
    sum += j

uni = uni[:5000]
for i in range(5000):
	unigram[uni[i][0]] = i


#The below chunk of code makes sure all words from table RG65 are included
#instead of just the 5000 most frequently occuring words. it ends up being the most
#frequently occuring 4970 words plus the other 30 words in table RG65 that were not
#already in the top 5000.
#comment the below code chunk out to keep the model using just the most frequent 5000 words
ctr = 1
for x,y in P:
	if x != 'serf':
		if x != uni[unigram[x]][0]:
			uni[5000-ctr] = (x, old[x])
			unigram[uni[5000-ctr][0]] = 5000-ctr
			ctr += 1
		if y != uni[unigram[y]][0]:
			uni[5000-ctr] = (y, old[y])
			unigram[uni[5000-ctr][0]] = 5000-ctr
			ctr += 1



#STEP 3 word-context vector M1 based on bigram counts
M1 = np.zeros(shape=(5000,5000))
for i in range(len(words) - 1):
	wi = re.sub('[^A-Za-z]+', '', words[i]).lower()
	wi1 = re.sub('[^A-Za-z]+', '', words[i+1]).lower()
	if wi != '' and wi1 != '' and wi == uni[unigram[wi]][0] and wi1 == uni[unigram[wi1]][0]:
	    M1[unigram[wi], unigram[wi1]] += 1


#STEP 4 PPMI for M1 denoted M1plus
M1plus = np.zeros(shape=(5000,5000))
for i in range(5000):
	for j in range(5000):
		M1plus[i, j] = max(math.log((M1[i, j] / sum) / ((uni[i][1] / sum) * (uni[j][1] / sum) + 1e-31) + 1e-31, 2.0), 0)


#STEP 5 latent semantic model using SVD. M2_10, M2_50, and M2_100 denote
#truncated dimensions of 10, 50, 100 respectively
A, D, Q = np.linalg.svd(M1plus, full_matrices=False)
M2_10 = A[:, :10]
M2_50 = A[:, :50]
M2_100 = A[:, :100]

#STEP 6 done at beginning

#STEP 7 cosine similiarities for M1 (SM1), M1plus(SM1plus), M2_10 (SM2_10), M2_50 (SM2_50), M2_100 (SM2_100)
#a in front of name denotes matrix has cosine similarity for all pairs of words, later we pick relevant pairs
aSM1 = cosine_similarity(M1)
aSM1plus = cosine_similarity(M1plus)
aSM2_10 = cosine_similarity(M2_10)
aSM2_50 = cosine_similarity(M2_50)
aSM2_100 = cosine_similarity(M2_100)

#pick out the cosine similarity scores for the relevant pairs in P.
#SL only includes scores from S for pairs of words which actually exist in our top 5000 (so we have data)
#since I later forced all words in table RG65 into the top 5000, SL will contain all scores from S, except
#note the word 'serf' does not occur at all in the Brown Corpus, so its pair was omitted from analysis
L = []
SL = []
for i in range(len(P)):
	x,y = P[i]
	if x != 'serf' and x == uni[unigram[x]][0] and y == uni[unigram[y]][0]:
		L.append((x, y))
		SL.append(S[i])

SM1 = []
SM1plus = []
SM2_10 = []
SM2_50 = []
SM2_100 = []
for x,y in L:
	SM1.append(aSM1[unigram[x], unigram[y]])
	SM1plus.append(aSM1plus[unigram[x], unigram[y]])
	SM2_10.append(aSM2_10[unigram[x], unigram[y]])
	SM2_50.append(aSM2_50[unigram[x], unigram[y]])
	SM2_100.append(aSM2_100[unigram[x], unigram[y]])


#STEP 8 Pearson correlation
print "Cosine Similarities:"
print "S and SM1: ", pearsonr(SL, SM1)
print "S and SM1+: ", pearsonr(SL, SM1plus)
print "S and SM2_10: ", pearsonr(SL, SM2_10)
print "S and SM2_50: ", pearsonr(SL, SM2_50)
print "S and SM2_100: ", pearsonr(SL, SM2_100)
