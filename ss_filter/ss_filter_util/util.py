"""
Created on 2021/10/24
@author liuwei
"""
import json
from tqdm import tqdm, trange
from Bio import SeqIO

def get_pid_list(pid_list_file):
    try:
        with open(pid_list_file) as fp:
            return [line.split()[0] for line in fp]
    except TypeError:
        return pid_list_file

def get_index_protein_dic(protein_list):
    protein_dic = {}
    for index,protein in enumerate(protein_list):
        protein_dic[index] = protein
    return protein_dic

def get_protein_index_dic(protein_list):
    protein_dic = {}
    for index,protein in enumerate(protein_list):
        protein_dic[protein] = index
    return protein_dic

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
            pfam_clan_output[prot].add(family_clan_dict[family])
    return pfam_clan_output

def get_prefilter_list(prefilter_result):
    prefilter_list = []
    with open(prefilter_result) as fp:
        for line in tqdm(fp, desc='Get prefilter list'):
            line_list = line.strip().split('\t')
            protein1 = line_list[0].split('.pdb')[0]
            protein2 = line_list[1].split('.pdb')[0]
            score = eval(line_list[2])
            prefilter_list.append(((protein1, protein2), score))
    return prefilter_list

def get_prefilter_list_without_self(prefilter_result):
    prefilter_list = []
    with open(prefilter_result) as fp:
        for line in tqdm(fp, desc='Get prefilter list without self'):
            line_list = line.strip().split('\t')
            protein1 = line_list[0].split('.pdb')[0]
            protein2 = line_list[1].split('.pdb')[0]
            if (protein1 == protein2):
                continue
            score = eval(line_list[2].replace('inf', '1e30'))
            prefilter_list.append(((protein1, protein2), score))
    return prefilter_list