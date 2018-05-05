import gibbs
import pandas as pd
import os
import numpy as np

data_path = '../../project/pbm'
num_iter = 10
num_string = 30
K = 3

# Load pbm data
TFName = "TF_40"
sequences_csv = pd.read_csv(os.path.join(data_path, "sequences.tsv"), delimiter="\t")
seqs = sequences_csv['seq'].tolist()
fold_id = sequences_csv['Fold ID'].tolist()
event_id = sequences_csv['Event ID'].tolist()

seqs = seqs[:num_string]
fold_id = fold_id[:num_string]
event_id = event_id[:num_string]

targets = pd.read_csv(os.path.join(data_path, "targets.tsv"), delimiter="\t")
bscores = targets.as_matrix(columns=[TFName])
print("Number of sequences: {}".format(len(seqs)))
print("Number of binding scores: {}".format(len(bscores)))
# print(seqs)
I, motifs = gibbs.gibbs_sampling_wrapper(num_iter, K, seqs, 'random')

print('Creating dataframe...')
dataframe = []
for i in range(len(seqs)):
	dataframe.append([fold_id[i], event_id[i], I[i], motifs[i]])

print('Writing to CSV...')
df = pd.DataFrame(dataframe, columns={'Fold ID','Event ID','motif_index','motif'})
df.to_csv('random_K=3_30string.csv', index=False)
'''
# Remove sequences with mismatched lengths
from scipy.stats import mode
lenArr = np.array([len(seqs[i]) for i in range(len(seqs))])
modeLen = mode(lenArr)[0][0]
count = 0
remove_indices = []
for i in range(len(seqs)):
    if len(seqs[i]) != modeLen:
        remove_indices.append(i)
        count += 1
nseqs = [seqs[i] for i in range(len(seqs)) if i not in remove_indices]
nbscores = np.delete(bscores, remove_indices)
seqs = nseqs
bscores = nbscores
print("Removed {0} entries, number of seqs is now {1}, number of binding scores is now {2}".format(count, len(seqs), len(bscores)))

print('seqs:', seqs)
print('bscores:', bscores)
'''