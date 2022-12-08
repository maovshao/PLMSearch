"""
Created on 2021/12/28
@author liuwei
"""
import os
from Bio.PDB import *
from tqdm import tqdm, trange
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import pairwise2 as pw2 
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from ss_filter_util.util import get_pid_list, get_index_protein_dic, get_protein_index_dic, read_fasta, get_prefilter_list_without_self
from ss_filter_util.esm_similarity_filter import esm_similarity_filiter
from ss_filter_util.esm_ss_predict import esm_ss_predict_tri

def get_input_output(todo_dir_list, todo_name_list):
    todo_file_list = []
    todo_fig_list = []
    for todo_dir in todo_dir_list:
        for todo_name in todo_name_list:
            todo_file = todo_dir + todo_name
            todo_file_list.append(todo_file)
            todo_fig_dir = './scientist_figures/' + todo_dir.split('/')[-3] + '/'
            os.makedirs(todo_fig_dir, exist_ok=True)
            todo_fig_dir_name = todo_name + '.png'
            todo_fig = todo_fig_dir + todo_fig_dir_name
            todo_fig_list.append(todo_fig)
    return todo_file_list, todo_fig_list

#protein_num, family_num, avg_protein_num, max_protein_num, distribution figure
def cluster_statistics(pfam_result_file, clan_file_path, result_path):
    def get_clan_cluster(pfam_result_file, clan_file_path):
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
        with open(pfam_result_file) as fp:
            pfam_output = json.load(fp)
        clan_proteins = {}
        for prot in pfam_output:
            for family in pfam_output[prot]:
                if family in family_clan_dict:
                    clan = family_clan_dict[family]
                else:
                    clan = family
                if clan in clan_proteins:
                    clan_proteins[clan].add(prot)
                else:
                    clan_proteins[clan] = set()
                    clan_proteins[clan].add(prot)
        for i in clan_proteins:
            clan_proteins[i] = list(clan_proteins[i])
        return clan_proteins

    plt.rcParams.update(plt.rcParamsDefault)
    cluster_result = get_clan_cluster(pfam_result_file, clan_file_path)
    cluster_num = len(cluster_result)
    protein_num_dic = {}
    for family_name in cluster_result:
        protein_num_dic[family_name] = len(cluster_result[family_name])
    protein_num_dic = dict(sorted(protein_num_dic.items(), key=lambda item: item[1], reverse=True))
    protein_num_array = np.asarray(list(protein_num_dic.values()))
    total_tmalign_times = np.sum(protein_num_array*(protein_num_array-1))
    max_protein_num = np.max(protein_num_array)
    avg_protein_num = np.mean(protein_num_array)
    big_familys = {}
    half_tmalign_times = 0
    small_family_num = {}
    small_family_num[1] = 0
    small_family_num[2] = 0
    small_family_num[3] = 0
    for family_name in protein_num_dic:
        protein_num = protein_num_dic[family_name]
        if (protein_num<=3):
            small_family_num[protein_num] += 1
        if (half_tmalign_times<total_tmalign_times/2):
            big_familys[family_name] = protein_num
            half_tmalign_times += protein_num*(protein_num-1)

    # make data:
    x = 0.5 + np.arange(cluster_num)

    # plot
    fig, ax = plt.subplots()
    ax.bar(x, protein_num_array, width=1, linewidth=0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.yscale('log')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set(xlim=(0, cluster_num), ylim=(0.9, 10**(np.int(np.log10(max_protein_num))+1)), yticks=10**np.arange(0, (np.int(np.log10(max_protein_num))+2)))
    

    fig_name = result_path + 'pfamclan_cluster_statistics.png'
    fig.savefig(fig_name)
    plt.show()
    plt.close()

    statistics_name = ''.join([x+'.' for x in fig_name.split('.')[:-1]]) + "txt"
    with open(statistics_name, 'w') as f:
        f.write(f"cluster_num = {cluster_num}\n")
        f.write(f"total_tmalign_times(maybe with overlap) = {total_tmalign_times}\n")
        f.write(f"max_cluster_num = {max_protein_num}\n")
        f.write(f"avg_cluster_num = {avg_protein_num}\n")
        f.write(f"big_familys_num = {len(big_familys)}\n")
        for family_name in big_familys:
            f.write(f"{family_name} = {big_familys[family_name]}\n")
        f.write(f"big_familys_tmalign_sum = {half_tmalign_times}\n")
        f.write(f"small_family1_num = {small_family_num[1]}\n")
        f.write(f"small_family2_num = {small_family_num[2]}\n")
        f.write(f"small_family3_num = {small_family_num[3]}\n")

def ss_mat_statistics(ss_mat_path, query_protein_list_path, target_protein_list_path, todo_file_list, todo_fig_list, ridge_plot_name, method_list, top_list):
    plt.rcParams.update(plt.rcParamsDefault)
    score_array_all = []
    file_array_all = []

    ss_mat = np.load(ss_mat_path)
    query_protein_dic = get_protein_index_dic(get_pid_list(query_protein_list_path))
    target_protein_dic = get_protein_index_dic(get_pid_list(target_protein_list_path))

    tmscore_cut_row_column = set()
    for index1 in range(len(query_protein_dic)):
        for index2 in range(len(target_protein_dic)):
            if (ss_mat[index1][index2] > 0.5):
                tmscore_cut_row_column.add((index1,index2))

    for index, todo_file in enumerate(todo_file_list):
        if (os.path.exists(todo_file)==False):
            print(f'{todo_file} does not exist!')
            continue
        
        prefilter_list = get_prefilter_list_without_self(todo_file)
        if (top_list[index] != 'all'):
            prefilter_list = sorted(prefilter_list, key=lambda x:x[1], reverse=True)
            prefilter_list = prefilter_list[:top_list[index]]
        
        score_array = []
        for pair in prefilter_list:
            r = query_protein_dic[pair[0][0]]
            c = target_protein_dic[pair[0][1]]
            score = ss_mat[r][c]
            score_array.append(score)
            score_array_all.append(score)
            file_array_all.append(method_list[index])
        score_array = np.asarray(score_array)
        score_avg = np.mean(score_array)

        ax = sns.displot(score_array, bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], linewidth=0)
        ax.set(xlim=(0, 1))
        fig_name = todo_fig_list[index]
        plt.savefig(fig_name)
        plt.show()
        plt.close()

        statistics_name = ''.join([x+'.' for x in fig_name.split('.')[:-1]]) + "txt"
        with open(statistics_name, 'w') as f:
            f.write(f'ss_mat_num_of_values:{score_array.shape[0]}\n')
            f.write(f'ss_mat_avg_score:{score_avg}\n')

            exist_num = 0
            for pair in prefilter_list:
                r = query_protein_dic[pair[0][0]]
                c = target_protein_dic[pair[0][1]]
                if ((r,c) in tmscore_cut_row_column):
                    exist_num += 1
            precision = exist_num/len(prefilter_list)
            recall = exist_num/len(tmscore_cut_row_column)
            f1_score = 2 * precision * recall /(precision + recall)

            f.write(f'------------------------------------------\n')
            f.write(f'Tm-score > 0.5\n')
            f.write(f'rec_rate:{recall}\n')
            f.write(f'rec:{exist_num}/{len(tmscore_cut_row_column)}\n')
            f.write(f'pre_rate:{precision}\n')
            f.write(f'pre:{exist_num}/{len(prefilter_list)}\n')
            f.write(f'f1_score:{f1_score}\n')

    # Initialize the FacetGrid object
    df = pd.DataFrame(dict(score=np.asarray(score_array_all), file=np.asarray(file_array_all)))
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    ax = sns.FacetGrid(df, row="file", hue="file")

    # Draw the densities in a few steps
    ax.map(sns.kdeplot, "score", fill=True, linewidth=1.5, clip_on=False)

    # passing color=None to refline() uses the hue mapping
    ax.refline(y=0, linewidth=1.5, linestyle="-", color=None, clip_on=False)

    # Set the subplots to overlap
    ax.figure.subplots_adjust(hspace=-.8)

    # Remove axes details that don't play well with overlap
    ax.set_titles("")
    ax.set(xticks=[], xlabel="", yticks=[], ylabel="")
    ax.despine(bottom=True, left=True)
    #add legend
    ax.add_legend(title='Cluster method')
    plt.xlim(0,1)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.savefig(ridge_plot_name)
    plt.show()
    plt.close()

def esm_similarity_statistics(query_esm_filename, target_esm_filename, query_protein_list_path, target_protein_list_path, ss_mat_path, fig_name, mode = 'cos'):
    with open(query_esm_filename, 'rb') as handle:
        query_esm_dict = pickle.load(handle)
    with open(target_esm_filename, 'rb') as handle:
        target_esm_dict = pickle.load(handle)
    
    plt.rcParams.update(plt.rcParamsDefault)
    esm_example = esm_similarity_filiter()

    query_protein_dic = get_index_protein_dic(get_pid_list(query_protein_list_path))
    target_protein_dic = get_index_protein_dic(get_pid_list(target_protein_list_path))

    ss_mat = np.load(ss_mat_path)

    df_dict = {}
    df_dict['esm_similarity'] = []
    df_dict['structure_similarity'] = []    
    structure_similarity_list = []

    for index1 in range(len(query_protein_dic)):
        for index2 in range(len(target_protein_dic)):
            if (query_protein_dic[index1] == target_protein_dic[index2]):
                continue
            structure_similarity_list.append(((query_protein_dic[index1], target_protein_dic[index2]), ss_mat[index1][index2]))

    for i in trange(100000):
        if (mode == 'cos'):
            esm_similarity = esm_example.cos_mean_esm_identity_compute(query_esm_dict[structure_similarity_list[i][0][0]], target_esm_dict[structure_similarity_list[i][0][1]]).item()
        elif (mode == 'mse'):
            esm_similarity = esm_example.mse_mean_esm_identity_compute(query_esm_dict[structure_similarity_list[i][0][0]], target_esm_dict[structure_similarity_list[i][0][1]]).item()
        else:
            print(f'Wrong Mode {mode}!!!')
            return
        df_dict['esm_similarity'].append(esm_similarity)
        df_dict['structure_similarity'].append(structure_similarity_list[i][1])
    
    df = pd.DataFrame(dict(esm_similarity=np.asarray(df_dict['esm_similarity']), 
                structure_similarity=np.asarray(df_dict['structure_similarity'])))
    df['esm_similarity'] = (df['esm_similarity']-df['esm_similarity'].min()) / (df['esm_similarity'].max()-df['esm_similarity'].min())
    from scipy.stats import pearsonr, spearmanr
    rho, _ = pearsonr(df['structure_similarity'], df['esm_similarity'])
    print(f"Pearson correlation coefficient of {mode} = {rho}")
    rho, _ = spearmanr(df['structure_similarity'], df['esm_similarity'])
    print(f"Spearman correlation coefficient of {mode} = {rho}")

    g = sns.lmplot(x="structure_similarity", y="esm_similarity", data=df, scatter_kws={"alpha":0.2, "s":1}, line_kws={"color":"r","alpha":1,"lw":1.5})
    if (mode == 'cos'):
        plt.xlabel('Structure similarity')
        plt.ylabel('ESM embedding similarity(COS)')
    if (mode == 'mse'):
        plt.xlabel('Structure similarity')
        plt.ylabel('ESM embedding similarity(Euclidean)')
    plt.xlim(0,1)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.ylim(0,1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.savefig(fig_name)
    plt.show()
    plt.close()

def ss_predictor_statistics(query_esm_filename, target_esm_filename, query_protein_list_path, target_protein_list_path, ss_mat_path, save_model_path, device_id, fig_name, cos):
    with open(query_esm_filename, 'rb') as handle:
        query_esm_dict = pickle.load(handle)
    with open(target_esm_filename, 'rb') as handle:
        target_esm_dict = pickle.load(handle)

    plt.rcParams.update(plt.rcParamsDefault)
    query_protein_dic = get_index_protein_dic(get_pid_list(query_protein_list_path))
    target_protein_dic = get_index_protein_dic(get_pid_list(target_protein_list_path))

    ss_mat = np.load(ss_mat_path)

    model = esm_ss_predict_tri(embed_dim = 1280)
    model.load_pretrained(save_model_path)
    model.eval()

    ## set the device
    if (device_id == None or device_id == []):
        print("None of GPU is selected.")
        device = "cpu"
        model.to(device)
        model_methods = model
    else:
        if torch.cuda.is_available()==False:
            print("GPU selected but none of them is available.")
            device = "cpu"
            model.to(device)
            model_methods = model
        else:
            print("We have", torch.cuda.device_count(), "GPUs in total!, we will use as you selected")
            model = nn.DataParallel(model, device_ids = device_id)
            device = f'cuda:{device_id[0]}'
            model.to(device)
            model_methods = model.module

    df_dict = {}
    df_dict['ss_sort_score'] = []
    df_dict['structure_similarity'] = []

    structure_similarity_list = []

    for index1 in range(len(query_protein_dic)):
        for index2 in range(len(target_protein_dic)):
            if (query_protein_dic[index1] == target_protein_dic[index2]):
                continue
            structure_similarity_list.append(((query_protein_dic[index1], target_protein_dic[index2]), ss_mat[index1][index2]))
    
    for i in trange(100000):
        embedding1 = query_esm_dict[structure_similarity_list[i][0][0]].unsqueeze(0)
        embedding2 = target_esm_dict[structure_similarity_list[i][0][1]].unsqueeze(0)
        if (cos == True):
            ss_sort_score = (model(embedding1, embedding2) * F.cosine_similarity(embedding1, embedding2)).item()
        else:
            ss_sort_score = model(embedding1, embedding2).item()
        df_dict['ss_sort_score'].append(ss_sort_score)
        df_dict['structure_similarity'].append(structure_similarity_list[i][1])

    df = pd.DataFrame(dict(ss_sort_score=np.asarray(df_dict['ss_sort_score']), 
            structure_similarity=np.asarray(df_dict['structure_similarity'])))
    df['ss_sort_score'] = (df['ss_sort_score']-df['ss_sort_score'].min()) / (df['ss_sort_score'].max()-df['ss_sort_score'].min())
    from scipy.stats import pearsonr, spearmanr
    rho, _ = pearsonr(df['structure_similarity'], df['ss_sort_score'])
    print(f"Pearson correlation coefficient of SS-predictor{'(COS)' if cos else ''} = {rho}")
    rho, _ = spearmanr(df['structure_similarity'], df['ss_sort_score'])
    print(f"Spearman correlation coefficient of SS-predictor{'(COS)' if cos else ''} = {rho}")
    g = sns.lmplot(x="structure_similarity", y="ss_sort_score", data=df, scatter_kws={"alpha":0.2, "s":1}, line_kws={"color":"r","alpha":1,"lw":1.5})
    plt.xlabel('Structure similarity')
    plt.ylabel(f"Predicted structure similarity of SS-predictor{'(COS)' if cos else ''}")
    plt.xlim(0,1)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.ylim(0,1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.savefig(fig_name)
    plt.show()
    plt.close()
    
def get_miss_wrong_statistics(ss_mat_path, query_protein_list, target_protein_list, todo_file_list, top_list, result_path):
    query_protein_dic = get_index_protein_dic(get_pid_list(query_protein_list))
    target_protein_dic = get_index_protein_dic(get_pid_list(target_protein_list))
    
    ss_mat = np.load(ss_mat_path)
    get_row_column = set()
    wrong_row_column = set()

    for index1 in range(len(query_protein_dic)):
        for index2 in range(len(target_protein_dic)):
            if (query_protein_dic[index1] == target_protein_dic[index2]):
                continue
            if (ss_mat[index1][index2] > 0.5):
                get_row_column.add((query_protein_dic[index1],target_protein_dic[index2]))
            if (ss_mat[index1][index2] < 0.3):
                wrong_row_column.add((query_protein_dic[index1],target_protein_dic[index2]))

    for index, todo_file in enumerate(todo_file_list):
        if (os.path.exists(todo_file)==False):
            print(f'{todo_file} does not exist!')
            continue
        
        prefilter_list = get_prefilter_list_without_self(todo_file)
        if (top_list[index] != 'all'):
            prefilter_list = sorted(prefilter_list, key=lambda x:x[1], reverse=True)
            prefilter_list = prefilter_list[:top_list[index]]

        predicted_cut_row_column = set()
        for filtered_num in trange(0, len(prefilter_list)):
            predicted_cut_row_column.add(prefilter_list[filtered_num][0])
        miss_list = list(get_row_column - predicted_cut_row_column)
        get_list = list(get_row_column & predicted_cut_row_column)
        wrong_list = list(wrong_row_column & predicted_cut_row_column)

        method_name = todo_file.split('/')[-1]
        get_result_filename = f'{result_path}get_{method_name}.txt'
        miss_result_filename = f'{result_path}miss_{method_name}.txt'
        wrong_result_filename = f'{result_path}wrong_{method_name}.txt'

        with open(miss_result_filename, 'w') as f:
            for x in miss_list:
                f.write(f'{x}\n')
        with open(get_result_filename, 'w') as f:
            for x in get_list:
                f.write(f'{x}\n')
        with open(wrong_result_filename, 'w') as f:
            for x in wrong_list:
                f.write(f'{x}\n')

def venn_graph3(filename_list, label_tuple, venn_graph_name):
    def get_set(pair_list_filename):
        with open(pair_list_filename) as fp:
            return set([eval(line) for line in fp])
    plt.rcParams.update(plt.rcParamsDefault)

    filename_list[0] = get_set(filename_list[0])
    filename_list[1] = get_set(filename_list[1])
    filename_list[2] = get_set(filename_list[2])

    venn3(filename_list, label_tuple)
    plt.savefig(venn_graph_name)
    plt.show()
    plt.close()

def pair_list_statistics(pair_list_filename, query_fasta_filename, target_fasta_filename, query_protein_list_path, target_protein_list_path, ss_mat_path):
    def get_list(pair_list_filename):
        with open(pair_list_filename) as fp:
            return [eval(line) for line in fp]
    def get_best_sequence_identity(sequence1, sequence2):
        global_align = pw2.align.globalxx(sequence1, sequence2)
        best_sequence_identity = 0
        for i in global_align:
            sequence_identity = i[2]/(i[4]-i[3])
            best_sequence_identity = max(best_sequence_identity, sequence_identity)
        return best_sequence_identity

    plt.rcParams.update(plt.rcParamsDefault)
    pair_list = get_list(pair_list_filename)
    _, query_protein_sequence = read_fasta(query_fasta_filename)
    _, target_protein_sequence = read_fasta(target_fasta_filename)
    query_protein_index_dic = get_protein_index_dic(get_pid_list(query_protein_list_path))
    target_protein_index_dic = get_protein_index_dic(get_pid_list(target_protein_list_path))
    ss_mat = np.load(ss_mat_path)

    cut_structure_similarity = 0.5
    #cut_structure_similarity = 0.3
    cut_sequence_identity = 0.3

    df_dict = {}
    df_dict['structure_similarity'] = []
    df_dict['sequence_identity'] = []

    st0_se0_num = 0
    st0_se1_num = 0
    st1_se0_num = 0
    st1_se1_num = 0

    for pair in pair_list:
        protein1 = pair[0]
        protein2 = pair[1]
        structure_similarity = ss_mat[query_protein_index_dic[protein1], target_protein_index_dic[protein2]]
        df_dict['structure_similarity'].append(structure_similarity)

        sequence1 = query_protein_sequence[protein1]
        sequence2 = target_protein_sequence[protein2]
        best_sequence_identity = get_best_sequence_identity(sequence1, sequence2)
        df_dict['sequence_identity'].append(best_sequence_identity)

        if (structure_similarity>cut_structure_similarity):
            if (best_sequence_identity>cut_sequence_identity):
                st1_se1_num +=1
            else:
                st1_se0_num +=1
        else:
            if (best_sequence_identity>cut_sequence_identity):
                st0_se1_num +=1
            else:
                st0_se0_num +=1       
        
    g = sns.JointGrid(data=df_dict, x="structure_similarity", y="sequence_identity", xlim = (0,1.1), ylim = (0,1.1))
    g.plot_joint(sns.scatterplot)
    g.plot_marginals(sns.histplot, kde=True)
    g.refline(x=cut_structure_similarity, y=cut_sequence_identity)
    fig_name = pair_list_filename.split('.txt')[0]+'_sequence_identity.png'
    g.set_axis_labels(xlabel='Structure similarity', ylabel='Sequence identity')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()
    plt.close()

    st1_se1_sum = 579
    st1_se0_sum = 2138
    st1_sum = 2717

    print('Statistics_result:')
    print(f'st1_se1_num={st1_se1_num}/{st1_se1_sum}')
    print(f'st1_se1_rate={st1_se1_num/st1_se1_sum}')
    print(f'st1_se0_num={st1_se0_num}/{st1_se0_sum}')
    print(f'st1_se0_rate={st1_se0_num/st1_se0_sum}')
    print(f'st1_num={st1_se1_num+st1_se0_num}/{st1_sum}')
    print(f'st1_rate={(st1_se1_num+st1_se0_num)/st1_sum}')
    print('---------------------------------------------')

def scop_roc(alnresult_dir, methods_filename_list, roc_plot_name, methods_name_list):
    def rocx2coord(fn, colID, method):
        df = pd.read_csv(fn, header=0, sep='\t')
        dfsort = df.sort_values(by=colID, ascending=False)
        n = 1
        for index,row in dfsort.iterrows():
            if ((n-1)%20 == 0):
                score = float(row[colID])
                qfraction = n / df.shape[0]
                qfracs.append(qfraction)
                senss.append(min(score, 1))
                methods.append(method)
            n+=1

    dfcol = ["FAM", "SFAM", "FOLD"]
    plt.rcParams.update(plt.rcParamsDefault)

    for i, cls in enumerate(dfcol):
        qfracs, senss, methods = [], [], []
        for k in range(len(methods_filename_list)):
            rocx2coord(f'{alnresult_dir}{methods_filename_list[k]}', cls, method = methods_name_list[k])        
        
        df = pd.DataFrame(dict(qfracs=np.asarray(qfracs), 
                        sens=np.asarray(senss),
                        Method=np.asarray(methods)))
        #画ROC的图
        if (cls!="FOLD"):
            rel = sns.relplot(
                data=df,
                x="qfracs", y="sens",
                hue="Method",
                style="Method",
                kind='line',
                markers=True,
                legend=False
            )
        else:
            rel = sns.relplot(
                data=df,
                x="qfracs", y="sens",
                hue="Method",
                style="Method",
                kind='line',
                markers=True
            )
        plt.xlabel('Fraction of queries')
        plt.ylabel('Sensitivity up to the 1st FP')
        rel.fig.suptitle(cls)
        #plt.tight_layout()
        plt.savefig(f"{roc_plot_name}_{cls}.png")
        plt.show()
        plt.close()

def tmscore_aupr(ss_mat_path, query_protein_list, target_protein_list, todo_file_list, plot_name, method_list):
    plt.rcParams.update(plt.rcParamsDefault)

    ss_mat = np.load(ss_mat_path)
    query_protein_dic = get_protein_index_dic(get_pid_list(query_protein_list))
    query_protein_list = get_pid_list(query_protein_list)
    target_protein_dic = get_protein_index_dic(get_pid_list(target_protein_list))
    target_protein_list = get_pid_list(target_protein_list)

    df_dict = {}
    df_dict['precision'] = []
    df_dict['recall'] = []
    df_dict['method'] = []

    true_list = ss_mat.reshape(-1)
    true_list = np.where(true_list>0.5, 1, 0)

    for index, todo_file in enumerate(todo_file_list):
        if (os.path.exists(todo_file)==False):
            print(f'{todo_file} does not exist!')
            continue
        
        prefilter_list = get_prefilter_list_without_self(todo_file)
        predict_tmscore_mat = np.zeros((len(query_protein_list), len(target_protein_list)), dtype=np.float64)
        for pair in tqdm(prefilter_list):
            protein_id1 = query_protein_dic[pair[0][0]]
            protein_id2 = target_protein_dic[pair[0][1]]
            predict_tmscore_mat[protein_id1][protein_id2] = pair[1]
                
        predict_list = predict_tmscore_mat.reshape(-1)
        
        precision, recall, thresholds = precision_recall_curve(true_list, predict_list)
        step_gap = int(len(precision) / 1000) + 1
        for index2 in range(0, len(precision), step_gap):
            df_dict['precision'].append(precision[index2])
            df_dict['recall'].append(recall[index2])
            df_dict['method'].append(method_list[index])
        average_precision = average_precision_score(true_list, predict_list, average="micro")
        print(f"AUPR of {method_list[index]}:{average_precision}")

    if (len(method_list)>8):
        df = pd.DataFrame(dict(Precision=np.asarray(df_dict['precision']), 
                        Recall=np.asarray(df_dict['recall']),
                        Method=np.asarray(df_dict['method'])))
        ax = sns.relplot(data=df, x="Recall", y="Precision", hue='Method', style="Method", kind="line")
        plt.legend([],[], frameon=False)
        plt.savefig(f"{plot_name}.png")
        plt.show()
        plt.close()

def tmscore_precision_recall(ss_mat_path, query_protein_list, target_protein_list, todo_file_list, method_list, k):
    from sklearn.metrics import average_precision_score
    def precision_at_k(y_true, y_score, k, pos_label=1):
        from sklearn.utils import column_or_1d
        from sklearn.utils.multiclass import type_of_target
        
        y_true_type = type_of_target(y_true)
        if not (y_true_type == "binary"):
            raise ValueError("y_true must be a binary column.")
        
        # Makes this compatible with various array types
        y_true_arr = column_or_1d(y_true)
        y_score_arr = column_or_1d(y_score)
        
        y_true_arr = y_true_arr == pos_label
        
        desc_sort_order = np.argsort(y_score_arr)[::-1]
        y_true_sorted = y_true_arr[desc_sort_order]
        y_score_sorted = y_score_arr[desc_sort_order]
        
        true_positives = y_true_sorted[:k].sum()
        
        return true_positives / k
    ss_mat = np.load(ss_mat_path)
    query_protein_dic = get_protein_index_dic(get_pid_list(query_protein_list))
    query_protein_list = get_pid_list(query_protein_list)
    target_protein_dic = get_protein_index_dic(get_pid_list(target_protein_list))
    target_protein_list = get_pid_list(target_protein_list)

    df_dict = {}
    df_dict['MAP'] = []
    df_dict['P@k'] = []
    df_dict['method'] = []

    true_list = np.where(ss_mat>0.5, 1, 0)

    for index, todo_file in enumerate(todo_file_list):
        if (os.path.exists(todo_file)==False):
            print(f'{todo_file} does not exist!')
            continue
        
        prefilter_list = get_prefilter_list_without_self(todo_file)
        predict_tmscore_mat = np.zeros((len(query_protein_list), len(target_protein_list)), dtype=np.float64)
        for pair in tqdm(prefilter_list):
            protein_id1 = query_protein_dic[pair[0][0]]
            protein_id2 = target_protein_dic[pair[0][1]]
            predict_tmscore_mat[protein_id1][protein_id2] = pair[1]
        
        sum_p_at_k = 0
        sum_average_precision = 0
        good_query = 0
        for query_index in range(len(query_protein_list)):
            true_query_result = true_list[query_index]
            if len(np.nonzero(np.where(true_query_result>0.5, 1, 0))[0]) == 0:
                continue
            predict_query_result = predict_tmscore_mat[query_index]
            sum_average_precision += average_precision_score(true_query_result, predict_query_result, average="micro")
            sum_p_at_k += precision_at_k(true_query_result, predict_query_result, k=k)
            good_query += 1
        MAP = sum_average_precision/good_query
        p_at_k = sum_p_at_k/good_query
        df_dict['MAP'].append(MAP)
        df_dict['P@k'].append(p_at_k)
        df_dict['method'].append(method_list[index])
        print(f"MAP of {method_list[index]}:{MAP}")
        print(f"P@{k} of {method_list[index]}:{p_at_k}")
        print(f"Good query = {good_query}")