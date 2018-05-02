"""
CS 466 Project Code
RankMotif++ Implementation
Author: Mu Chen (muchen2@illinois.edu)
"""

import tensorflow as tf 
import numpy as np 


"""
Convert numeric alignment matrix to PWM
"""

def alignment2PWM(alignMat, alphabet_len):
	pfmList = []
	for i in range(alphabet_len):
		pfmRow = np.sum((alignMat == i), axis=0)
		pfmList.append(pfmRow)
	pfm = np.vstack(pfmList)

	# add pseudocount to prevent underflow
	pfm += (pfm == 0).astype(np.int32)

	# convert pfm to ppm
	ppm = pfm / np.sum(pfm, axis=0)[None,:]

	# convert ppm to pwm
	background_prob = 1. / alphabet_len
	pwm = np.log(ppm / background_prob)
	return pwm

"""
Encode sequences to index of alphabet
Sequences should have the same length
"""
def seq2NumMat(seqs, alphabet=['A', 'C', 'G', 'T']):
	seqMatList = []
	for s in seqs:
		smRow = np.array([alphabet.index(s[i]) for i in range(len(s))])
		seqMatList.append(smRow)
	seqMat = np.vstack(seqMatList)
	return seqMat

# Convert binding intensity to preference matrix
def bscores2PrefMat(bscores):
	n = len(bscores)
	pref = np.zeros((n, n))
	for i in range(n):
		pref[i, :] = (bscores[i] > bscores).astype(np.float32)
	return pref

# remove the nan rows in target data and remove the corresponding sequence
# from the numeric sequence matrix
def removeNaN(seqMat, bscores):
	remove_indices = np.where(np.isnan(bscores))[0]
	bscores = np.delete(bscores, remove_indices)
	seqMat = np.delete(seqMat, remove_indices, axis=0)
	return (seqMat, bscores)

"""
Convert sequences to PWM
Sequences are represented as list of strings
"""

def seq2PWM(seqs, alphabet=['A', 'C', 'G', 'T']):
	# Generate alignment matrix
	return alignment2PWM(seq2NumMat(seqs, alphabet), len(alphabet))



# RankMotif model implemented in Tensorflow
class RankMotifModel():

	def __init__ (self, K=7):
		self.K = K
		self.trainedTheta = None
		self.trainedW = None




	def bindingAffinity(self, S, theta):
		# Convert numeric sequence matrix into matrix of probablity
		# using PWM theta
		seqMatFlatten = tf.reshape(S, [-1])
		seqShape = tf.shape(S)
		colIndices = tf.tile(tf.range(0, seqShape[1]), [seqShape[0]])
		indices = tf.stack((seqMatFlatten, colIndices), axis=1)
		seqProbs = tf.reshape(tf.gather_nd(theta, indices), seqShape)


		# Calculate Binding affinity for each of the sequences
		# Calculate K-mer-wise probability
		sumFilter = tf.reshape(tf.ones(self.K), [1, self.K, 1, 1])
		seqProbs4D = tf.reshape(seqProbs, [1, seqShape[0], seqShape[1], 1])
		KmersProbs4D = tf.nn.convolution(seqProbs4D, sumFilter, padding="VALID")
		KmersProbs = tf.reshape(KmersProbs4D, (seqShape[0], seqShape[1]-self.K+1))

		# convert log probabilities to true probabilities
		#KmersProbsNL = tf.exp(KmersProbs)
		#self.KmersProbsNL = KmersProbsNL

		tempLogProb = tf.reduce_sum(tf.log1p(-tf.expm1(KmersProbs)))
		return bindAffs



	def estimatedBindingPreference(self, bdAffs, w):
		# Generated estimated binding preference probability matrix using
		# PWM theta
		bdAffsWeighted = tf.pow(bdAffs, w)
		probMat = bdAffsWeighted[:,None] / (bdAffsWeighted[:,None] + bdAffsWeighted[None,:] + self.eps)
		return probMat

	"""
	Fit model on string sequence data
	"""
	def fit(self, seqs, bscores, alphabet, gdStepSize=0.1, max_iter=200, batch_size=100, verbose=False):
		# Converting string gene sequences to numeric sequences
		self.alphabet = alphabet
		seqMat = seq2NumMat(seqs, alphabet)
		alphabet_len = len(alphabet)
		self.eps = 1e-7
		#print(seqMat.shape)

		# Remove nan entries
		seqMat, bscores = removeNaN(seqMat, bscores)
		n = len(bscores)
		
		self.S = tf.placeholder(tf.int32, shape=(batch_size, seqMat.shape[1]))
		self.X = tf.placeholder(tf.float32, shape=(batch_size, batch_size))

		w_init_val = 1e-5 + np.random.normal(0.5, 1.0/6)
		theta_init_val = -np.abs(np.random.normal(-5.0, 1.0, size=(alphabet_len, seqMat.shape[1])))
		self.w = tf.Variable(w_init_val, dtype=tf.float32)
		self.theta = tf.Variable(theta_init_val, dtype=tf.float32)

		# Compute negative log likelihood
		neg_log_likelihood = self.negLogLikelihood()

		# Model training
		with tf.Session() as sess:
			optimizer = tf.train.GradientDescentOptimizer(gdStepSize)
			train = optimizer.minimize(neg_log_likelihood)

			# Initialize all varibles
			sess.run(tf.global_variables_initializer())
			for i in range(max_iter):
				indices = np.random.choice(n, size=batch_size, replace=False)
				batch_seqs = seqMat[indices, :]
				batch_pref = bscores2PrefMat(bscores[indices])
				#print (batch_pref)
				print(sess.run(self.KmersProbsNL, feed_dict={self.S: batch_seqs, self.X:batch_pref}))
				sess.run(train, feed_dict={self.S: batch_seqs, self.X:batch_pref})
				if verbose:
					negL = sess.run(neg_log_likelihood, feed_dict={self.S: batch_seqs, self.X:batch_pref})
					print ("Epoch: {0}/{1}, negative log likelihood: {2}".format(i, max_iter, negL))
			self.trainedTheta = sess.run(self.theta)
			self.trainedW = sess.run(self.w)
		return self

	# Calculate the negative log likelihood of the input data
	# Used exclusively in training
	def negLogLikelihood(self):
		# Calculating binding affinity of all sequences
		bdAffs = self.bindingAffinity(self.S, self.theta)
		self.bdAffs = bdAffs
		
		# Calculating estimated binding preference (in probabilities)
		probMat = self.estimatedBindingPreference(bdAffs, self.w)
		
		# Calculating negative log likelihood of the data
		# In the paper, the data are arranged so that only the top right corner has 1 entries
		# Here it accepts any kind of matrix
		probMatLog = tf.log(probMat + self.eps)
		probOnes = tf.multiply (self.X, probMatLog)
		neg_log_likelihood = -tf.reduce_sum(probOnes)
		return neg_log_likelihood

	"""
	Predict the matrix of likelihood (confidence) which indicate whether a specific sequence
	is prefered than another
	"""
	def predict_proba(self, seqs):
		if self.trainedTheta is None:
			raise Exception("Please fit the model before performing prediction")
		seqMat = seq2NumMat(seqs, self.alphabet)
		Seq = tf.placeholder(tf.int32, shape=seqMat.shape)
		thetaTF = tf.constant(self.trainedTheta)
		wTF = tf.constant(self.trainedW)

		bdAffs = self.bindingAffinity(Seq, thetaTF)
		probMat = self.estimatedBindingPreference(bdAffs, wTF)
		res = None
		with tf.Session() as sess:
			res = sess.run(probMat, feed_dict={Seq: seqMat})
		return res

