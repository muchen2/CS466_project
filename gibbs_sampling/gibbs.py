'''
Algorithm from https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/gibbs.pdf
'''
import numpy as np
import random
import math
import string

# nucleotides = ['A', 'T', 'C', 'G']
nucleotides = list(string.ascii_lowercase[:26])

# arguments: k = length of motif, strings = list of strings
# option: 'best score' -- choose x_i that best matches profile P
#		  'random' -- choose x_i randomly according to the profile P's probability distribution
# return value: best motif indices and substrings of length k
def gibbs_sampling(k, strings, option = 'best_score'):
	# list of lengths of strings
	len_strs = [len(string) for string in strings]
	num_str = len(strings)

	# select x1, ..., xp randomly
	I = [random.randint(0, len_str - k) for len_str in len_strs]

	LastI = None
	while I != LastI:
		LastI = list(I)
		# print('indices I = ', LastI)
		for i in range(num_str):
			# build a profile matrix except x_i
			profile_in = []
			for j in range(num_str):
				if j != i:
					profile_in.append(strings[j][I[j] : I[j] + k])
			# print('input to profile:', profile_in)
			P = profile(profile_in)

			# find the motif in string i that matches the profile best
			if option == 'best_score':
				best_score = -math.inf
				best_idx = 0
				for j in range(len_strs[i] - k + 1):
					score = profile_score(P, strings[i][j : j + k], strings)
					if score > best_score:
						best_score = score
						best_idx = j
				I[i] = best_idx

			# randomly find a motif according to the profile distribution
			elif option == 'random':
				scores = np.empty(len_strs[i] - k + 1)
				for j in range(len_strs[i] - k + 1):
					scores[j] = profile_score(P, strings[i][j : j + k], strings)
				# find the probability of each substring being chosen
				scores = np.exp(scores) # if keeps the log score, negative probability will appear
				# print(scores)
				scores = scores / np.sum(scores)
				I[i] = np.random.choice(np.arange(0, len_strs[i] - k + 1), p = scores)


	motifs = [strings[i][I[i] : I[i] + k] for i in range(num_str)]
	return I, motifs

# find profile matrix from a set of strings
# strings: a set of n substrings with same length k
# return value: a 2d array pf size 4 * n (the order is: A T C G)
def profile(strings):
	num_str = len(strings)
	num_nuc = len(nucleotides)
	len_str = len(strings[0])
	profile_mtx = np.zeros((num_nuc, len_str))

	# enumerate each position
	for i in range(len_str):
		# print('i = ', i)
		# for each string
		for j in range(num_str):
			# print('j = ', j)
			# print('strings[j][i] = ', strings[j][i])
			# print('index in alphabet =', nucleotides.index(strings[j][i]))
			profile_mtx[nucleotides.index(strings[j][i]), i] += 1

	# normalize the profile matrix to represent probability
	col_sum = profile_mtx.sum(axis = 0)
	profile_mtx = profile_mtx / col_sum
	return profile_mtx

# arguments: 
# profile_mtx: profile matrix build on the rest of motif candidates
# motif: the sequence left behind in profile_mtx
# strings: all sequences
# return value: a score of target_str
def profile_score(profile_mtx, motif, strings):
	background_freq = np.zeros(len(nucleotides))
	num_str = len(strings)

	# calculate background frequencies
	for i in range(len(nucleotides)):
		background_freq[i] = sum([strings[j].count(nucleotides[i]) for j in range(num_str)])
	# convert to probability
	background_freq = background_freq / sum(background_freq)

	# print(background_freq)
	score = 0
	for i in range(len(motif)):
		idx = nucleotides.index(motif[i])
		# if the candidate motif does not match the 
		if profile_mtx[idx, i] == 0:
		 	return -math.inf
		score += np.log(profile_mtx[idx, i] / background_freq[idx])
		# print(motif[i], ':', profile_mtx[idx, i] / background_freq[idx], ',', background_freq[idx])
	return score


indices, motifs = gibbs_sampling(3, (["aaabbb", "bbbaaabb", \
	'babaaab', 'ababacaaabac', 'abbbababaaabbbaba']), 'random')

# indices, motifs = gibbs_sampling(3, ["thequickdog", "browndog", "dogwood"])
print(motifs)