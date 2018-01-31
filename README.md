Lab 1

Code is in file lab1.py with comments indicating which piece of code accomplishes each step.

Notes:

Because there were only 6 pairs {(coast, forest), (coast, hill), (car, journey), (food, fruit), (coast, shore), (automobile, car)} from the set in table RG65 that had both words occuring in my top 5000, it was not enough data to work with, since everything comes from the 5000x5000 array of bigram data. The data for just these 6 pairs without modifying the most frequent 5000 is below, and does not show anything very meaningful:

Cosine Similarities:
S and SM1:  (0.6311231864891127, 0.17900860221385637)
S and SM1+:  (-0.3593446397356634, 0.48418387014880093)
S and SM2_10:  (-0.7218396086823597, 0.105298724693541)
S and SM2_50:  (0.06061063954296708, 0.9091953718121875)
S and SM2_100:  (-0.17026376582871552, 0.747072303255742)


Instead, I decided to replace some of the words in the most frequent 5000 with the rest of the words in table RG65 so that I could collecect some data for every pair of words. It ended up being the 4970 most frequent words plus 30 words from table RG65 which were not already in the top 5000. The word 'serf' never appears in the Brown Corpus, so I omitted the pair (serf, slave) from the analysis. The improved analysis for all pairs is below:

Cosine Similarities:
S and SM1:  (0.3250232591954757, 0.008780002595589535)
S and SM1+:  (0.2005972191854655, 0.11198129113414915)
S and SM2_10:  (0.05778309863951632, 0.650164850907056)
S and SM2_50:  (0.37902952506712595, 0.002011010593492061)
S and SM2_100:  (0.335235564310977, 0.006771711589437011)


The Pearson correlation scores show that there is definitely a correlation, albeit not a very strong one. I suspect this is because we are only using bigram counts and because we only used the most frequent 5000 words (and there are some words which might be meaningless like abbreviations). I also observed that the correlation score for the matrix with PPMI's was worse than the correlation score for the matrix with raw counts, which I did not expect and perhaps I made a mistake in calculating PPMI's.
