'''
Algorithm from https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/gibbs.pdf
'''
import numpy as np
import random
import math
import string

nucleotides = ['A', 'T', 'C', 'G']
# nucleotides = list(string.ascii_lowercase[:26])
background_freq = np.zeros(len(nucleotides))

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
	num_str = len(strings)

	score = 0
	for i in range(len(motif)):
		idx = nucleotides.index(motif[i])
		# if the candidate motif does not match the 
		if profile_mtx[idx, i] == 0:
		 	return -math.inf
		score += np.log(profile_mtx[idx, i] / background_freq[idx])
		# print(motif[i], ':', profile_mtx[idx, i] / background_freq[idx], ',', background_freq[idx])
	return score

# iterations: number of trails to ensure the best motif is found
def gibbs_sampling_wrapper(iterations, k, strings, option = 'best_score'):
	print('Running gibbs sampler...')
	best_diff = np.float(math.inf)
	num_str = len(strings)

	# num_str: number of input strings
	# motifs: candidate motifs returned by gibbs_sampling
	def measure_difference(num_str, motifs):
		diff = 0
		for i in range(num_str):
			for j in range(i + 1, num_str):
				for m in range(k):
					if motifs[i][m] != motifs[j][m]:
						diff += 1
		return diff
	
	# pre-compute background probability
	# calculate background frequencies
	global background_freq
	for i in range(len(nucleotides)):
		background_freq[i] = sum([strings[j].count(nucleotides[i]) for j in range(num_str)])
	# convert to probability
	background_freq = background_freq / sum(background_freq)

	for i in range(iterations):
		print('iteration # ' + str(i))
		I, motifs = gibbs_sampling(k, strings, option)

		# motifs_arr = np.asarray(motifs)
		# print('motifs array:', motifs)
		# print(motifs[0])
		# print(motifs[0][0])
		diff = measure_difference(num_str, motifs)
		# print('diff = ', diff)
		# print('best_diff = ', best_diff)
		if diff < best_diff:
			best_diff = diff
			best_motif = motifs
			bestI = I
	return bestI, best_motif

'''
# test code
indices, motifs = gibbs_sampling_wrapper(10, 3, (["aaabbb", "bbbaaabb", \
	'babaaab', 'ababacaaabac', 'abbbababaaabbbaba']), 'random')

print(motifs)
'''