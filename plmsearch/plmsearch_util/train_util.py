import sys
import pickle
import random
import torch
from logzero import logger
import numpy as np
from .util import read_fasta, get_index_protein_dic

class plmsearch_dataset:
    def __init__(self):
        self.x0 = []
        self.x1 = []
        self.y = []

    def load_dataset(self, esm_embedding, fasta_filename, ss_mat_filename, device):
        print('# loading esm result:', esm_embedding, file=sys.stderr)
        print('# loading fasta file:', fasta_filename, file=sys.stderr)
        print('# loading ss mat file:', ss_mat_filename, file=sys.stderr)

        protein_list, _ = read_fasta(fasta_filename)
        protein_dic = get_index_protein_dic(protein_list)

        ppi_net_mat = np.load(ss_mat_filename)

        logger.info(F'{ppi_net_mat.shape} {np.count_nonzero(ppi_net_mat)}')

        with open(esm_embedding, 'rb') as handle:
            embedding_dic = pickle.load(handle)
        
        for protein in embedding_dic:
            embedding_dic[protein] = embedding_dic[protein].to(device)

        high_count = 0
        low_count = 0

        high_indices = []
        low_indices = []

        # Find the indices of non-zero elements
        non_zero_indices = np.nonzero(ppi_net_mat)

        for index in zip(non_zero_indices[0], non_zero_indices[1]):
            u, v = index
            d = ppi_net_mat[u, v]
            if d > 0.5:
                high_indices.append((u, v))
                high_count += 1
            elif d <= 0.5:
                low_indices.append((u, v))
                low_count += 1

        # sample num of pos and neg
        sample_indices = random.sample(high_indices, high_count)
        sample_indices.extend(random.sample(low_indices, low_count))

        for u, v in sample_indices:
            self.x0.append(embedding_dic[protein_dic[u]])
            self.x1.append(embedding_dic[protein_dic[v]])
            self.y.append(ppi_net_mat[u, v])

        print('# loaded', len(self.x0), 'sequence pairs', file=sys.stderr)
        print('# number of samples with d > 0.5:', high_count, file=sys.stderr)
        print('# number of samples with d < 0.5:', low_count, file=sys.stderr)

    def finalize(self):
        self.y = torch.as_tensor(self.y).view(-1)
        self.y = self.y.float()

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, i):
        return self.x0[i], self.x1[i], self.y[i]
