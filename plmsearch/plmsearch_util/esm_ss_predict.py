import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as ssp
from logzero import logger
from tqdm import tqdm
from plmsearch_util.util import get_pid_list, get_index_protein_dic
class esm_ss_predict_tri(nn.Module):
    """
    Predicts contact maps as sigmoid(z_i W W W z_j + b)
    """
    def __init__(self, embed_dim):
        super(esm_ss_predict_tri, self).__init__()

        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.linear3 = nn.Linear(embed_dim, embed_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def load_pretrained(self, mtplm_path):
        #load to cpu at first, and then tranfer according to device_id
        state_dict = torch.load(mtplm_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)

    def forward(self, z1, z2):
        h = self.linear3(self.linear2(self.linear1(z1))).unsqueeze(1)
        s = torch.matmul(h, z2.unsqueeze(2)) + self.bias.squeeze()
        ss = torch.sigmoid(s.squeeze())
        return ss

class esm_ss_dataset:
    def __init__(self, mean_esm_result, 
                protein_list_filename,
                ss_mat_filename):
        print('# loading esm result:', mean_esm_result, file=sys.stderr)
        print('# loading protein list file:', protein_list_filename, file=sys.stderr)
        print('# loading ss mat file:', ss_mat_filename, file=sys.stderr)

        protein_dic = get_index_protein_dic(get_pid_list(protein_list_filename))

        ppi_net_mat = (mat_:=ssp.load_npz(ss_mat_filename)) + ssp.eye(mat_.shape[0], format='csc')
        logger.info(F'{ppi_net_mat.shape} {ppi_net_mat.nnz}')

        with open(mean_esm_result, 'rb') as handle:
            embedding_dic = pickle.load(handle)

        ppi_net_mat_coo = ssp.coo_matrix(ppi_net_mat)
        self.x0 = []
        self.x1 = []
        self.y = []
        for u, v, d in tqdm(zip(ppi_net_mat_coo.row, ppi_net_mat_coo.col, ppi_net_mat_coo.data),
                            total=ppi_net_mat_coo.nnz, desc='PPI'):
            self.x0.append(embedding_dic[protein_dic[u]])
            self.x1.append(embedding_dic[protein_dic[v]])
            self.y.append(d)

        self.y = torch.as_tensor(self.y).view(-1)
        self.y = self.y.float()

        print('# loaded', len(self.x0), 'sequence pairs', file=sys.stderr)

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, i):
        return self.x0[i], self.x1[i], self.y[i]