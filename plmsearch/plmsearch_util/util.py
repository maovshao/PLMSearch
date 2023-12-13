import json
import torch
import numpy as np
from tqdm import tqdm, trange
from Bio import SeqIO
from pathlib import Path
import torch.nn.functional as F

def get_index_protein_dic(protein_list):
    return {index: protein for index, protein in enumerate(protein_list)}

def get_protein_index_dic(protein_list):
    return {protein: index for index, protein in enumerate(protein_list)}

def read_fasta(fn_fasta):
    prot2seq = {}
    with open(fn_fasta) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq)
            prot = record.id
            prot2seq[prot] = seq
    return list(prot2seq.keys()), prot2seq

def get_family_result(pfam_result_file):
    with open(pfam_result_file) as fp:
        pfam_output = json.load(fp)

    pfam_family_output = {}
    for prot in pfam_output:
        pfam_family_output[prot] = set()
        for family in pfam_output[prot]:
            pfam_family_output[prot].add(family)
    return pfam_family_output

def get_clan_result(pfam_result_file, clan_file_path):
    family_clan_dict = {}
    with open(clan_file_path) as f:
        # Read data line by line
        for line in f:
            # split data by tab
            # store it in list
            l=line.split('\t')
            # append list to ans
            if (l[1]!=''):
                family_clan_dict[l[0]] = l[1]
            else:
                family_clan_dict[l[0]] = l[0]
    
    with open(pfam_result_file) as fp:
        pfam_output = json.load(fp)

    pfam_clan_output = {}
    for prot in pfam_output:
        pfam_clan_output[prot] = set()
        for family in pfam_output[prot]:
            if family in family_clan_dict:
                pfam_clan_output[prot].add(family_clan_dict[family])
            else:
                pfam_clan_output[prot].add(family)
    return pfam_clan_output

def get_search_list(search_result):
    search_list = []
    with open(search_result) as fp:
        for line in tqdm(fp, desc='Get search list'):
            line_list = line.strip().split('\t')
            protein1 = line_list[0].split('.pdb')[0]
            protein2 = line_list[1].split('.pdb')[0]
            score = eval(line_list[2])
            search_list.append(((protein1, protein2), score))
    return search_list

def get_search_list_without_self(search_result):
    search_list = []
    with open(search_result) as fp:
        for line in tqdm(fp, desc='Get search list without self'):
            line_list = line.strip().split('\t')
            protein1 = line_list[0].split('.pdb')[0]
            protein2 = line_list[1].split('.pdb')[0]
            if (protein1 == protein2):
                continue
            score = eval(line_list[2])
            search_list.append(((protein1, protein2), score))
    return search_list

def make_parent_dir(path):
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

def dot_product(z1, z2):
    return torch.sigmoid(torch.mm(z1, z2.t()))

def pairwise_dot_product(z1, z2):
    h = z1.unsqueeze(1)
    s = torch.matmul(h, z2.unsqueeze(2))
    ss = torch.sigmoid(s.squeeze())
    return ss

def cos_similarity(z1, z2):
    eps = 1e-8
    z1_n, z2_n = z1.norm(dim=1)[:, None], z2.norm(dim=1)[:, None]
    z1_norm = z1 / torch.max(z1_n, eps * torch.ones_like(z1_n))
    z2_norm = z2 / torch.max(z2_n, eps * torch.ones_like(z2_n))
    sim_mt = torch.mm(z1_norm, z2_norm.transpose(0, 1))
    return sim_mt

def pairwise_cos_similarity(z1, z2):
    return F.cosine_similarity(z1, z2)

def euclidean_similarity(z1, z2):
    eps = 1
    dist_matrix = torch.cdist(z1, z2)
    sim_matrix = 1 / (dist_matrix + eps)
    return sim_matrix

def pairwise_euclidean_similarity(z1, z2):
    eps = 1
    mse_loss = F.mse_loss(z1, z2, reduction='none')
    mse_loss = mse_loss.sum(dim=1)
    similarity = 1 / (mse_loss + eps)
    return similarity

def tensor_to_list(tensor):
    decimals = 4
    numpy_array = tensor.cpu().numpy()
    return np.round(numpy_array, decimals=decimals).tolist()
