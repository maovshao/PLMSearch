import os
from Bio.PDB import *
from tqdm import tqdm, trange
import json
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from .model import plmsearch
from .util import get_index_protein_dic, get_protein_index_dic, read_fasta, get_search_list_without_self, make_parent_dir, pairwise_dot_product, pairwise_cos_similarity, pairwise_euclidean_similarity, tensor_to_list
from .alignment_util import pairwise_sequence_align_util

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
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.set(xlim=(0, cluster_num), ylim=(0.9, 10**(np.int(np.log10(max_protein_num))+1)), yticks=10**np.arange(0, (np.int(np.log10(max_protein_num))+2)))
    ax.set_xticklabels(['{:,}'.format(int(label)) for label in ax.get_xticks()])

    plt.show()
    plt.close()

    statistics_name = result_path + 'pfamclan_cluster_statistics.txt'
    make_parent_dir(statistics_name)
    with open(statistics_name, 'w') as f:
        f.write(f"cluster_num = {cluster_num}\n")
        f.write(f"total_tmalign_times(maybe with overlap) = {total_tmalign_times}\n")
        f.write(f"max_cluster_num = {max_protein_num}\n")
        f.write(f"avg_cluster_num = {avg_protein_num}\n")
        f.write(f"big_familys_num = {len(big_familys)}\n")
        for family_name in big_familys:
            f.write(f"{family_name} = {big_familys[family_name]}\n")
        f.write(f"big_familys_tmalign_sum = {half_tmalign_times}\n")
        f.write(f"big_familys_rate = {half_tmalign_times/total_tmalign_times}\n")
        f.write(f"small_family1_num = {small_family_num[1]}\n")
        f.write(f"small_family2_num = {small_family_num[2]}\n")
        f.write(f"small_family3_num = {small_family_num[3]}\n")

def max_sequence_identity_statistics(max_sequence_identity_list, dataset_list):
    df_dict = {}
    df_dict["max_sequence_identity"] = []
    df_dict["dataset"] = []

    for index, max_sequence_identity in enumerate(max_sequence_identity_list):
        with open(max_sequence_identity, 'r') as f:
            for line in tqdm(f):
                columns = line.strip().split('\t')
                df_dict['max_sequence_identity'].append(eval(columns[1]))
                df_dict['dataset'].append(dataset_list[index])

    ax = sns.violinplot(data=df_dict, x="dataset", y="max_sequence_identity")
    plt.xticks(fontsize=18)
    plt.ylim(0,0.5)
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=22)
    plt.show()
    plt.close()

def precision_recall_statistics(ss_mat_path, query_protein_fasta, target_protein_fasta, todo_file_list, todo_fig_list, method_list, top_list, legend):
    plt.rcParams.update(plt.rcParamsDefault)
    score_array_all = []
    file_array_all = []

    query_protein_list, _ = read_fasta(query_protein_fasta)
    query_protein_index_dic = get_protein_index_dic(query_protein_list)
    target_protein_list, _ = read_fasta(target_protein_fasta)
    target_protein_index_dic = get_protein_index_dic(target_protein_list)
    ss_mat = np.load(ss_mat_path)

    tmscore_cut_row_column = set()
    for index1 in range(len(query_protein_index_dic)):
        for index2 in range(len(target_protein_index_dic)):
            if (ss_mat[index1][index2] > 0.5):
                tmscore_cut_row_column.add((index1,index2))

    recall_dict = {}
    for index, todo_file in enumerate(todo_file_list):
        if (os.path.exists(todo_file)==False):
            print(f'{todo_file} does not exist!')
            continue
        
        prefilter_list = get_search_list_without_self(todo_file)
        if (top_list[index] != 'all'):
            prefilter_list = sorted(prefilter_list, key=lambda x:x[1], reverse=True)
            prefilter_list = prefilter_list[:top_list[index]]
        
        score_array = []
        for pair in prefilter_list:
            r = query_protein_index_dic[pair[0][0]]
            c = target_protein_index_dic[pair[0][1]]
            score = ss_mat[r][c]
            score_array.append(score)
            score_array_all.append(score)
            file_array_all.append(method_list[index])
        score_array = np.asarray(score_array)
        score_avg = np.mean(score_array)

        ax = sns.displot(score_array, bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], linewidth=0)
        plt.ylabel('Count', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()
        plt.close()

        statistics_name = ''.join([x+'.' for x in todo_fig_list[index].split('.')[:-1]]) + "txt"
        make_parent_dir(statistics_name)
        with open(statistics_name, 'w') as f:
            f.write(f'ss_mat_num_of_values:{score_array.shape[0]}\n')
            f.write(f'ss_mat_avg_score:{score_avg}\n')

            exist_num = 0
            for pair in prefilter_list:
                r = query_protein_index_dic[pair[0][0]]
                c = target_protein_index_dic[pair[0][1]]
                if ((r,c) in tmscore_cut_row_column):
                    exist_num += 1
            precision = exist_num/len(prefilter_list)
            recall = exist_num/len(tmscore_cut_row_column)

            f.write(f'------------------------------------------\n')
            f.write(f'rec_rate:{recall}\n')
            f.write(f'rec:{exist_num}/{len(tmscore_cut_row_column)}\n')
            f.write(f'pre_rate:{precision}\n')
            f.write(f'pre:{exist_num}/{len(prefilter_list)}\n')

            print(f'-------------- Output for {statistics_name} --------------\n')
            print(f'rec_rate:{recall}')
            print(f'rec:{exist_num}/{len(tmscore_cut_row_column)}\n')
            print(f'pre_rate:{precision}')
            print(f'pre:{exist_num}/{len(prefilter_list)}\n')
            recall_dict[method_list[index]] = recall

    # Initialize the FacetGrid object
    df = pd.DataFrame(dict(score=np.asarray(score_array_all), file=np.asarray(file_array_all)))
    output_path = ''.join([x+'/' for x in todo_fig_list[index].split('/')[:-1]]) + "source_data.xlsx"
    df.to_excel(output_path, index=False)

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
    if legend:
        ax.add_legend(title='', fontsize=20)
    plt.xticks([0, 0.5, 1], fontsize=20)
    plt.show()
    plt.close()

    return recall_dict

def precision_recall_plot(df_dict):
    plt.rcParams.update(plt.rcParamsDefault)
    ax = sns.barplot(data=df_dict, x="dataset", y="recall", hue="method")
    for i in ax.containers:
        ax.bar_label(i,)
    sns.move_legend(ax, "upper left")

    plt.ylabel('Recall', fontsize=18)
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=10)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
    plt.show()
    plt.close()

def esm_similarity_correlation_statistics(query_esm_filename, target_esm_filename, query_protein_fasta, target_protein_fasta, ss_mat_path, device_id, mode = 'cos'):
    with open(query_esm_filename, 'rb') as handle:
        query_esm_dict = pickle.load(handle)
    with open(target_esm_filename, 'rb') as handle:
        target_esm_dict = pickle.load(handle)
    
    plt.rcParams.update(plt.rcParamsDefault)

    query_protein_list, _ = read_fasta(query_protein_fasta)
    query_index_protein_dic = get_index_protein_dic(query_protein_list)
    target_protein_list, _ = read_fasta(target_protein_fasta)
    target_index_protein_dic = get_index_protein_dic(target_protein_list)

    ss_mat = np.load(ss_mat_path)

    ## set the device
    if (device_id == None or device_id == []):
        print("None of GPU is selected.")
        device = "cpu"
    else:
        if torch.cuda.is_available()==False:
            print("GPU selected but none of them is available.")
            device = "cpu"
        else:
            print("We have", torch.cuda.device_count(), "GPUs in total!, we will use as you selected")
            device = f'cuda:{device_id[0]}'

    df_dict = {}
    df_dict['esm_similarity'] = []
    df_dict['structure_similarity'] = []    
    query_embedding = []
    target_embedding = []

    pairs = []
    for index1 in range(len(query_index_protein_dic)):
        for index2 in range(len(target_index_protein_dic)):
            if query_index_protein_dic[index1] == target_index_protein_dic[index2]:
                continue
            similarity = ss_mat[index1][index2]
            pairs.append((index1, index2, similarity))

    pairs.sort(key=lambda x: x[2], reverse=True)

    selected_pairs = pairs[:100000]

    for index1, index2, similarity in selected_pairs:
        query_embedding.append(query_esm_dict[query_index_protein_dic[index1]].to(device))
        target_embedding.append(target_esm_dict[target_index_protein_dic[index2]].to(device))
        df_dict['structure_similarity'].append(similarity)

    query_embedding = torch.stack(query_embedding)
    target_embedding = torch.stack(target_embedding)
    if (mode == 'cos'):
        predict_score_tensor = pairwise_cos_similarity(query_embedding, target_embedding)
    elif (mode == 'euclidean'):
        predict_score_tensor = pairwise_euclidean_similarity(query_embedding, target_embedding)
    else:
        print(f'Wrong Mode {mode}!!!')
    df_dict['esm_similarity'] = tensor_to_list(predict_score_tensor)
    
    df = pd.DataFrame(dict(esm_similarity=np.asarray(df_dict['esm_similarity']), 
                structure_similarity=np.asarray(df_dict['structure_similarity'])))
    df['esm_similarity'] = (df['esm_similarity']-df['esm_similarity'].min()) / (df['esm_similarity'].max()-df['esm_similarity'].min())
    from scipy.stats import spearmanr
    rho_s, _ = spearmanr(df['structure_similarity'], df['esm_similarity'])
    print(f"Spearman correlation coefficient of {mode} = {rho_s}")

    ax = sns.lmplot(x="structure_similarity", y="esm_similarity", data=df, scatter_kws={"alpha":0.2, "s":1}, line_kws={"color":"r","alpha":1,"lw":1.5})

    # add text box for the statistics
    stats = (f"Spearman's $\\rho$ = {rho_s:.4f}")
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    from matplotlib.font_manager import FontProperties
    font = FontProperties(family='monospace', size=15)
    plt.text(0.6, -0.05, stats, font=font, bbox=bbox)

    plt.xlabel('TM-score', fontsize=22)
    if (mode == 'cos'):
        plt.ylabel('COS', fontsize=22)
    if (mode == 'euclidean'):
        plt.ylabel('Euclidean', fontsize=22)
    plt.xlim(0.3,1.1)
    plt.xticks([0.4, 0.6, 0.8, 1.0], fontsize=22)
    plt.ylim(-0.1,1.1)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=22)
    plt.show()
    plt.close()

def plmsearch_correlation_statistics(query_esm_filename, target_esm_filename, query_protein_fasta, target_protein_fasta, ss_mat_path, save_model_path, device_id):
    with open(query_esm_filename, 'rb') as handle:
        query_esm_dict = pickle.load(handle)
    with open(target_esm_filename, 'rb') as handle:
        target_esm_dict = pickle.load(handle)

    plt.rcParams.update(plt.rcParamsDefault)
    query_protein_list, _ = read_fasta(query_protein_fasta)
    query_index_protein_dic = get_index_protein_dic(query_protein_list)
    target_protein_list, _ = read_fasta(target_protein_fasta)
    target_index_protein_dic = get_index_protein_dic(target_protein_list)

    ss_mat = np.load(ss_mat_path)

    model = plmsearch(embed_dim = 1280)
    model.load_pretrained(save_model_path)
    model.eval()

    ## set the device
    if (device_id == None or device_id == []):
        print("None of GPU is selected.")
        device = "cpu"
        model.to(device)
    else:
        if torch.cuda.is_available()==False:
            print("GPU selected but none of them is available.")
            device = "cpu"
            model.to(device)
        else:
            print("We have", torch.cuda.device_count(), "GPUs in total!, we will use as you selected")
            model = nn.DataParallel(model, device_ids = device_id)
            device = f'cuda:{device_id[0]}'
            model.to(device)

    with torch.no_grad():
        df_dict = {}
        df_dict['ss_predictor_score'] = []
        df_dict['structure_similarity'] = []
        query_embedding = []
        query_raw_embedding = []
        target_embedding = []

        pairs = []
        for index1 in range(len(query_index_protein_dic)):
            for index2 in range(len(target_index_protein_dic)):
                if query_index_protein_dic[index1] == target_index_protein_dic[index2]:
                    continue
                similarity = ss_mat[index1][index2]
                pairs.append((index1, index2, similarity))

        pairs.sort(key=lambda x: x[2], reverse=True)

        selected_pairs = pairs[:100000]

        for index1, index2, similarity in selected_pairs:
            query_embedding.append(model(query_esm_dict[query_index_protein_dic[index1]].to(device)))
            query_raw_embedding.append(query_esm_dict[query_index_protein_dic[index1]].to(device))
            target_embedding.append(target_esm_dict[target_index_protein_dic[index2]].to(device))
            df_dict['structure_similarity'].append(similarity)

        query_embedding = torch.stack(query_embedding)
        query_raw_embedding = torch.stack(query_raw_embedding)
        target_embedding = torch.stack(target_embedding)
        sim_score = tensor_to_list(pairwise_dot_product(query_embedding, target_embedding))
        cos_score = tensor_to_list(pairwise_cos_similarity(query_raw_embedding, target_embedding))

        df_dict['ss_predictor_score'] = [cos_score[i] if (cos_score[i]>0.995) else cos_score[i] * sim_score[i] for i in range(len(sim_score))]

        df = pd.DataFrame(dict(ss_predictor_score=np.asarray(df_dict['ss_predictor_score']), 
                structure_similarity=np.asarray(df_dict['structure_similarity'])))
        df['ss_predictor_score'] = (df['ss_predictor_score']-df['ss_predictor_score'].min()) / (df['ss_predictor_score'].max()-df['ss_predictor_score'].min())
        from scipy.stats import spearmanr
        rho_s, _ = spearmanr(df['structure_similarity'], df['ss_predictor_score'])
        print(f"Spearman correlation coefficient of SS-predictor = {rho_s}")
        g = sns.lmplot(x="structure_similarity", y="ss_predictor_score", data=df, scatter_kws={"alpha":0.2, "s":1}, line_kws={"color":"r","alpha":1,"lw":1.5})

        # add text box for the statistics
        stats = (f"Spearman's $\\rho$ = {rho_s:.4f}")
        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
        from matplotlib.font_manager import FontProperties
        font = FontProperties(family='monospace', size=15)
        plt.text(0.6, -0.05, stats, font=font, bbox=bbox)

        plt.xlabel('TM-score', fontsize=22)
        plt.ylabel(f"SS-predictor", fontsize=22)
        plt.xlim(0.3,1.1)
        plt.xticks([0.4, 0.6, 0.8, 1.0], fontsize=22)
        plt.ylim(-0.1,1.1)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=22)
        plt.show()
        plt.close()
    
def get_miss_wrong_statistics(ss_mat_path, query_protein_fasta, target_protein_fasta, todo_file_list, top_list, result_path):
    query_protein_list, _ = read_fasta(query_protein_fasta)
    query_index_protein_dic = get_index_protein_dic(query_protein_list)
    target_protein_list, _ = read_fasta(target_protein_fasta)
    target_index_protein_dic = get_index_protein_dic(target_protein_list)
    
    ss_mat = np.load(ss_mat_path)
    get_row_column = set()
    wrong_row_column = set()

    for index1 in range(len(query_index_protein_dic)):
        for index2 in range(len(target_index_protein_dic)):
            if (query_index_protein_dic[index1] == target_index_protein_dic[index2]):
                continue
            if (ss_mat[index1][index2] > 0.5):
                get_row_column.add((query_index_protein_dic[index1],target_index_protein_dic[index2]))
            if (ss_mat[index1][index2] < 0.2):
                wrong_row_column.add((query_index_protein_dic[index1],target_index_protein_dic[index2]))

    for index, todo_file in enumerate(todo_file_list):
        if (os.path.exists(todo_file)==False):
            print(f'{todo_file} does not exist!')
            continue
        
        search_list = get_search_list_without_self(todo_file)
        search_list = sorted(search_list, key=lambda x:x[1], reverse=True)
        search_list = search_list[:top_list[index]]

        predicted_cut_row_column = set()
        for filtered_num in trange(0, len(search_list)):
            predicted_cut_row_column.add(search_list[filtered_num][0])
        miss_list = list(get_row_column - predicted_cut_row_column)
        get_list = list(get_row_column & predicted_cut_row_column)
        wrong_list = list(wrong_row_column & predicted_cut_row_column)

        method_name = todo_file.split('/')[-1]
        get_result_filename = f'{result_path}get_{method_name}.txt'
        miss_result_filename = f'{result_path}miss_{method_name}.txt'
        wrong_result_filename = f'{result_path}wrong_{method_name}.txt'
        make_parent_dir(get_result_filename)
        make_parent_dir(miss_result_filename)
        make_parent_dir(wrong_result_filename)

        with open(miss_result_filename, 'w') as f:
            for x in miss_list:
                f.write(f'{x}\n')
        with open(get_result_filename, 'w') as f:
            for x in get_list:
                f.write(f'{x}\n')
        with open(wrong_result_filename, 'w') as f:
            for x in wrong_list:
                f.write(f'{x}\n')

def sequence_structure_statistics(query_protein_fasta, target_protein_fasta, ss_mat_path):
    query_protein_list, query_protein_sequence = read_fasta(query_protein_fasta)
    target_protein_list, target_protein_sequence = read_fasta(target_protein_fasta)

    query_protein_index_dic = get_protein_index_dic(query_protein_list)
    target_protein_index_dic = get_protein_index_dic(target_protein_list)

    ss_mat = np.load(ss_mat_path)

    cut_structure_similarity = 0.5
    cut_sequence_identity = 0.3

    st1_se1_sum = 0
    st1_se0_sum = 0
    st1_sum = 0

    for query_protein in query_protein_list:
        for target_protein in target_protein_list:
            if (query_protein == target_protein):
                continue
            structure_similarity = ss_mat[query_protein_index_dic[query_protein], target_protein_index_dic[target_protein]]
            if (structure_similarity<cut_structure_similarity):
                continue
            st1_sum += 1
            sequence1 = query_protein_sequence[query_protein]
            sequence2 = target_protein_sequence[target_protein]
            best_sequence_identity, _ = pairwise_sequence_align_util(sequence1, sequence2)
            if (best_sequence_identity>cut_sequence_identity):
                st1_se1_sum +=1
            else:
                st1_se0_sum +=1
    
    return st1_se1_sum, st1_se0_sum, st1_sum

def pair_list_statistics(pair_list_filename, query_protein_fasta, target_protein_fasta, ss_mat_path, st1_se1_sum, st1_se0_sum, st1_sum):
    def get_list(pair_list_filename):
        with open(pair_list_filename) as fp:
            return [eval(line) for line in fp]

    plt.rcParams.update(plt.rcParamsDefault)
    pair_list = get_list(pair_list_filename)
    query_protein_list, query_protein_sequence = read_fasta(query_protein_fasta)
    target_protein_list, target_protein_sequence = read_fasta(target_protein_fasta)

    query_protein_index_dic = get_protein_index_dic(query_protein_list)
    target_protein_index_dic = get_protein_index_dic(target_protein_list)

    ss_mat = np.load(ss_mat_path)

    cut_structure_similarity = 0.5
    cut_sequence_identity = 0.3

    df_dict = {}
    df_dict['structure_similarity'] = []
    df_dict['sequence_identity'] = []

    st0_se0_num = 0
    st0_se1_num = 0
    st1_se0_num = 0
    st1_se1_num = 0

    for pair in pair_list:
        query_protein = pair[0]
        target_protein = pair[1]
        structure_similarity = ss_mat[query_protein_index_dic[query_protein], target_protein_index_dic[target_protein]]
        df_dict['structure_similarity'].append(structure_similarity)

        sequence1 = query_protein_sequence[query_protein]
        sequence2 = target_protein_sequence[target_protein]
        best_sequence_identity, _ = pairwise_sequence_align_util(sequence1, sequence2)
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
        
    g = sns.JointGrid(data=df_dict, x="structure_similarity", y="sequence_identity", xlim = (-0.1,1.1), ylim = (-0.1,1.1))
    g.plot_joint(sns.scatterplot)
    g.plot_marginals(sns.histplot, kde=True)
    g.refline(x=cut_structure_similarity, y=cut_sequence_identity)

    g.ax_joint.set_xlabel('TM-score', fontsize=22)
    g.ax_joint.set_ylabel('Sequence identity', fontsize=22)

    g.ax_joint.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    g.ax_joint.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    g.ax_joint.tick_params(axis='x', labelsize=22)
    g.ax_joint.tick_params(axis='y', labelsize=22)
    
    plt.tight_layout()
    plt.show()
    plt.close()

    print('Statistics_result:')
    print(f'st1_se1_num={st1_se1_num}/{st1_se1_sum}')
    print(f'st1_se1_rate={st1_se1_num/st1_se1_sum}')
    print(f'st1_se0_num={st1_se0_num}/{st1_se0_sum}')
    print(f'st1_se0_rate={st1_se0_num/st1_se0_sum}')
    print(f'st1_num={st1_se1_num+st1_se0_num}/{st1_sum}')
    print(f'st1_rate={(st1_se1_num+st1_se0_num)/st1_sum}')
    print('---------------------------------------------')

def scop_roc(alnresult_dir, methods_filename_list, methods_name_list, line_style, color_dict, legend_pos = 0.85, time = False):
    def rocx2coord(fn, colID, method):
        df = pd.read_csv(fn, header=0, sep='\t')
        dfsort = df.sort_values(by=colID, ascending=False)
        n = 1
        for index, row in dfsort.iterrows():
            if ((n-1)%20 == 0):
                score = float(row[colID])
                qfraction = n / df.shape[0]
                qfracs.append(qfraction)
                senss.append(min(score, 1))
                methods.append(method)
            n+=1

    dfcol = ["FAM", "SFAM", "FOLD"]
    dfcol_dict = {"FAM":"Family", "SFAM":"Superfamily", "FOLD":"Fold"}
    plt.rcParams.update(plt.rcParamsDefault)

    for i, cls in enumerate(dfcol):
        qfracs, senss, methods = [], [], []
        for k in range(len(methods_filename_list)):
            rocx2coord(f'{alnresult_dir}{methods_filename_list[k]}', cls, method = methods_name_list[k])        
        
        df = pd.DataFrame(dict(qfracs=np.asarray(qfracs), 
                        sens=np.asarray(senss),
                        Method=np.asarray(methods)))

        if (cls=="FOLD"):
            rel = sns.relplot(
                data=df,
                x="qfracs", y="sens",
                hue="Method",
                style="Method",
                kind='line',
                markers=False,
                dashes=line_style,
                palette=color_dict.values(),
                legend='brief'
            )
            legend = rel._legend
            legend.set_bbox_to_anchor((legend_pos, 0.6))
            if (time == False):
                legend.set_title('')
            else:
                from matplotlib.font_manager import FontProperties
                legend.set_title('Search time')
                legend_title = legend.get_title()
                legend_title.set_fontsize('12')
                font = FontProperties(family='monospace', size=12)
                for text in legend.texts:
                    text.set_fontproperties(font)

        else:
            rel = sns.relplot(
                data=df,
                x="qfracs", y="sens",
                hue="Method",
                style="Method",
                kind='line',
                markers=False,
                dashes=line_style,
                palette=color_dict.values(),
                legend=False
            )
        plt.xlabel('Fraction of queries', fontsize=18)
        plt.ylabel('Sensitivity up to the 1st FP', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        rel.fig.suptitle(dfcol_dict[cls], fontsize=18)
        if (cls=="FOLD"):
            rel.fig.suptitle(dfcol_dict[cls], x=0.39, fontsize=18)
        plt.show()
        plt.close()

def scop_pr(alnresult_dir, methods_filename_list, methods_name_list, line_style, color_dict, legend_pos = 0.85, time = False):
    def prx2coord(fn, colID, method):
        df = pd.read_csv(fn, header=0, sep='\t')
        precision = df['PREC_' + colID].tolist()
        recall = df['RECALL_' + colID].tolist()
        step_gap = int(len(precision) / 1000) + 1
        for i in range(0, len(precision), step_gap):
            precisions.append(precision[i])
            recalls.append(recall[i])
            methods.append(method)
        aupr = auc(recall, precision)
        print(f"AUPR for {method} ({dfcol_dict[cls]}): {aupr}")

    dfcol = ["FAM", "SFAM", "FOLD"]
    dfcol_dict = {"FAM":"Family", "SFAM":"Superfamily", "FOLD":"Fold"}
    plt.rcParams.update(plt.rcParamsDefault)

    for i, cls in enumerate(dfcol):
        precisions, recalls, methods = [], [], []
        for k in range(len(methods_filename_list)):
            prx2coord(f'{alnresult_dir}{methods_filename_list[k]}', cls, method = methods_name_list[k])        
        
        df = pd.DataFrame(dict(Recall=np.asarray(recalls), 
                        Precision=np.asarray(precisions),
                        Method=np.asarray(methods)))

        if (cls=="FOLD"):
            rel = sns.relplot(
                data=df,
                x="Recall", y="Precision",
                hue="Method",
                style="Method",
                kind='line',
                markers=False,
                dashes=line_style,
                palette=color_dict.values(),
                legend='brief'
            )
            legend = rel._legend
            legend.set_bbox_to_anchor((legend_pos, 0.6))
            if (time == False):
                legend.set_title('')
            else:
                from matplotlib.font_manager import FontProperties
                legend.set_title('Search time')
                legend_title = legend.get_title()
                legend_title.set_fontsize('12')
                font = FontProperties(family='monospace', size=12)
                for text in legend.texts:
                    text.set_fontproperties(font)

        else:
            rel = sns.relplot(
                data=df,
                x="Recall", y="Precision",
                hue="Method",
                style="Method",
                kind='line',
                markers=False,
                dashes=line_style,
                palette=color_dict.values(),
                legend=False
            )
        plt.xlabel('Recall', fontsize=18)
        plt.ylabel('Precision', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        rel.fig.suptitle(dfcol_dict[cls], fontsize=18)
        if (cls=="FOLD"):
            rel.fig.suptitle(dfcol_dict[cls], x=0.39, fontsize=18)
        plt.show()
        plt.close()

def map_pk(ss_mat_path, query_protein_fasta, target_protein_fasta, todo_file_list, method_list, k_list, special_protein_list=None):
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

    query_protein_list, _ = read_fasta(query_protein_fasta)
    query_protein_index_dic = get_protein_index_dic(query_protein_list)
    target_protein_list, _ = read_fasta(target_protein_fasta)
    target_protein_index_dic = get_protein_index_dic(target_protein_list)

    df_dict = {}
    df_dict['metric'] = []
    df_dict['score'] = []
    df_dict['method'] = []

    true_list = np.where(ss_mat>0.5, 1, 0)

    for index, todo_file in enumerate(todo_file_list):
        if (os.path.exists(todo_file)==False):
            print(f'{todo_file} does not exist!')
            continue
        
        search_list = get_search_list_without_self(todo_file)
        predict_tmscore_mat = np.zeros((len(query_protein_list), len(target_protein_list)), dtype=np.float64)
        for pair in tqdm(search_list):
            protein_id1 = query_protein_index_dic[pair[0][0]]
            protein_id2 = target_protein_index_dic[pair[0][1]]
            predict_tmscore_mat[protein_id1][protein_id2] = pair[1]
        
        sum_p_at_k = []
        for i in range(len(k_list)):
            sum_p_at_k.append(0)
        sum_average_precision = 0
        good_query = 0
        for query_index in range(len(query_protein_list)):
            true_query_result = true_list[query_index]
            if len(np.nonzero(np.where(true_query_result>0.5, 1, 0))[0]) == 0:
                continue
            predict_query_result = predict_tmscore_mat[query_index]
            if special_protein_list is not None:
                protein_id = query_protein_list[query_index]
                if protein_id in special_protein_list:
                    sum_average_precision += average_precision_score(true_query_result, predict_query_result, average="micro")
                    for i in range(len(k_list)):
                        sum_p_at_k[i] += precision_at_k(true_query_result, predict_query_result, k=k_list[i])
                    good_query += 1
            else:
                sum_average_precision += average_precision_score(true_query_result, predict_query_result, average="micro")
                for i in range(len(k_list)):
                    sum_p_at_k[i] += precision_at_k(true_query_result, predict_query_result, k=k_list[i])
                good_query += 1
        MAP = sum_average_precision/good_query
        print(f"MAP of {method_list[index]}:{MAP}")
        df_dict['metric'].append('MAP')
        df_dict['score'].append(MAP)
        df_dict['method'].append(method_list[index])

        for i in range(len(k_list)):
            p_at_k = sum_p_at_k[i]/good_query
            print(f"P@{k_list[i]} of {method_list[index]}:{p_at_k}")
            df_dict['metric'].append(f'P@{k_list[i]}')
            df_dict['score'].append(p_at_k)
            df_dict['method'].append(method_list[index])
    
    return df_dict

def map_pk_plot(df_dict, color_dict, legend):
    plt.rcParams.update(plt.rcParamsDefault)
    ax = sns.barplot(data=df_dict, x="metric", y="score", hue="method", palette=color_dict.values())
    if legend:
        sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
    else:
        ax.get_legend().remove()

    plt.ylabel('Score', fontsize=18)
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=18)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
    plt.show()
    plt.close()

def scope_similarity_statistics(similarity_file, fold_file, check_set, x_gap, is_dot_similarity = False):
    plt.rcParams.update(plt.rcParamsDefault)
    fold_dict = {}
    fold_similarity_list = []
    same_different_list = []
    same_fold_sum = 0
    different_fold_sum = 0
    p_similarity_same_fold = {}
    p_similarity_different_fold = {}
    probability = {}
    probability["Similarity"] = []
    probability["Posterior Probability"] = []
    probability["Type"] = []
    for i in range(1001):
        x_axis = i / 1000
        p_similarity_same_fold[x_axis] = 0
        p_similarity_different_fold[x_axis] = 0

    with open(fold_file) as fp:
        for line in fp:
            line_list = line.split()
            protein = line_list[0]
            scop = line_list[1]
            scope_list = scop.split('.')
            fold_dict[protein] = scope_list[0] + '.' + scope_list[1]
    
    with open(similarity_file) as fp:
        for line in tqdm(fp):
            line_list = line.split()
            query_protein = line_list[0]
            target_protein = line_list[1]
            similarity = int(eval(line_list[2])*1000) / 1000
            if is_dot_similarity:
                similarity = similarity / 10
                similarity = int(similarity*1000) / 1000
                similarity = min(similarity, 1)
                similarity = max(similarity, 0)
            fold_similarity_list.append(similarity)
            if (fold_dict[query_protein] != fold_dict[target_protein]):
                same_different_list.append("Different")
                different_fold_sum += 1
                p_similarity_different_fold[similarity] += 1
            else:
                same_different_list.append("Same")
                same_fold_sum += 1
                p_similarity_same_fold[similarity] += 1

    p_same_fold = same_fold_sum / (same_fold_sum + different_fold_sum)
    p_different_fold = 1 - p_same_fold

    print(f"Same fold pairs = {same_fold_sum}")
    print(f"Different fold pairs = {different_fold_sum}")
    print(f"P(Same fold) = {p_same_fold}")
    print(f"P(Different fold) = {p_different_fold}")

    p_f_tm = 0
    p_not_f_tm = 1
    for i in range(1, 1001):
        x_axis = i / 1000
        p_tm_f = p_similarity_same_fold[x_axis] / same_fold_sum
        p_tm_not_f = p_similarity_different_fold[x_axis] / different_fold_sum
        if ((p_tm_f!=0) or (p_tm_not_f!=0)):
            p_f_tm = (p_tm_f*p_same_fold) / (p_tm_f*p_same_fold+p_tm_not_f*p_different_fold)
            p_not_f_tm = (p_tm_not_f*p_different_fold) / (p_tm_f*p_same_fold+p_tm_not_f*p_different_fold)

        if is_dot_similarity:
            x_axis = x_axis * 10

        if (i % x_gap == 0):
            probability["Similarity"].append(x_axis)
            probability["Posterior Probability"].append(p_f_tm)
            probability["Type"].append("Same")

            probability["Similarity"].append(x_axis)
            probability["Posterior Probability"].append(p_not_f_tm)
            probability["Type"].append("Different")

        if x_axis in check_set:
            print(f"Similarity = {x_axis}, Same Fold Posterior Probability = {p_f_tm}")
            print(f"Similarity = {x_axis}, Different Fold Posterior Probability = {p_not_f_tm}")

    df = pd.DataFrame(dict(Similarity=np.asarray(probability['Similarity']), 
                    Probability=np.asarray(probability['Posterior Probability']),
                    Type=np.asarray(probability['Type'])))
    ax = sns.relplot(data=df, x="Similarity", y="Probability", hue='Type', style="Type", kind="line", legend='brief')
    plt.setp(ax._legend.get_texts(), fontsize='18')
    legend = ax._legend
    legend.set_bbox_to_anchor((0.45, 0.5))
    legend.set_title('')

    plt.ylabel('Posterior Probability', fontsize=18)
    plt.xlabel('Similarity', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()
    plt.close()

    # Initialize the FacetGrid object
    df = pd.DataFrame(dict(score=np.asarray(fold_similarity_list), file=np.asarray(same_different_list)))
    output_path = similarity_file + "_source_data.xlsx"
    # Limitation from excel sheet
    df_subset = df.iloc[:int(len(df) * 0.1)]
    df_subset.to_excel(output_path, index=False)
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    ax = sns.FacetGrid(df, row="file", hue="file")

    # Draw the densities in a few steps
    ax.map(sns.kdeplot, "score", fill=True, linewidth=1.5, bw_adjust=3, clip_on=False)

    # passing color=None to refline() uses the hue mapping
    ax.refline(y=0, linewidth=1.5, linestyle="-", color=None, clip_on=False)

    # Set the subplots to overlap
    ax.figure.subplots_adjust(hspace=-.8)

    # Remove axes details that don't play well with overlap
    ax.set_titles("")
    ax.set(xticks=[], xlabel="", yticks=[], ylabel="")
    ax.despine(bottom=True, left=True)
    #add legend
    ax.add_legend(title='', fontsize=20)
    plt.xlim(0,1)
    plt.xticks([0, 0.5, 1], fontsize=20)
    plt.show()
    plt.close()

#def server_time(species, sex_counts):
def server_time(methods, time_counts):
    from matplotlib.patches import Patch

    time_part_list = list(time_counts.keys())
    x = np.arange(len(methods)) * 3  # the label locations
    width = 0.5  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Get the default color cycle

    legend_colors = color_cycle[:4]  # Choose the first four colors for the legend

    legend_patches = []  # Store legend patches

    for query_count in range(4):
        legend_patches.append(Patch(color=legend_colors[query_count]))

    for query_count in range(4):
        offset = width * multiplier
        bottom = [0,0]
        for part, time_list in time_counts.items():
            height = [time_list[methods[0]][query_count], time_list[methods[1]][query_count]]
            rects = ax.bar(x + offset, height, width, label=part, bottom=bottom, color=legend_colors[time_part_list.index(part)])
            if (part == 'Alignment'):
                ax.bar_label(rects, [10**query_count, 10**query_count],
                    padding=3, color='black', fontweight='bold')
            for i in range(len(bottom)):
                bottom[i] += height[i]
        multiplier += 1

    ax.set_ylabel('Time')
    ax.set_xticks(x + width * (len(time_counts) - 1) / 2)  # Adjust x-axis tick positions
    ax.set_xticklabels(methods)

    # Create a custom legend with the desired labels and colors
    ax.legend(handles=legend_patches, labels=time_counts.keys(), loc='upper center', ncols=2)

    ax.set_ylim(0, 2500)
    plt.show()
    plt.close()