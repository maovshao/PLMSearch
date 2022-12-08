"""
Created on 2021/10/24
@author liuwei
"""

import os
import time
import torch
import torch.nn.functional as F
import pickle
import torch.nn as nn
import argparse
from tqdm import tqdm, trange
from logzero import logger
from ss_filter_util.esm_ss_predict import esm_ss_predict_tri
from ss_filter_util.esm_similarity_filter import esm_similarity_filiter
from ss_filter_util.util import get_prefilter_list

def esm_ss_predict_sort(input_prefilter_result, threshold, nocos, query_esm_result, target_esm_result, save_model_path, device_id):
    
    def threshold_filter(protein1_list, protein2_list, nocos, device):        
        x0_tensor = []
        x1_tensor = []
        for index in range(len(protein1_list)):
            x0_tensor.append(query_embedding_dic[protein1_list[index]])
            x1_tensor.append(target_embedding_dic[protein2_list[index]])
        x0_tensor = torch.stack(x0_tensor)
        x1_tensor = torch.stack(x1_tensor)
        x0_tensor = x0_tensor.to(device)
        x1_tensor = x1_tensor.to(device)
        predict_score_tensor = model(x0_tensor, x1_tensor)
        if (nocos == False):
            predict_score_tensor = predict_score_tensor * F.cosine_similarity(x0_tensor, x1_tensor)
        predict_score_tensor = predict_score_tensor.tolist()

        for index in range(len(protein1_list)):
            protein1 = protein1_list[index]
            protein2 = protein2_list[index]
            if (protein1 == protein2):
                predict_score = 1
            else:
                predict_score = predict_score_tensor[index]

            if ((threshold!= None) and (predict_score < threshold)):
                continue
            else:
                protein_pair_dict[protein1].append((protein2, predict_score))

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

    with torch.no_grad():
        with open(query_esm_result, 'rb') as handle:
            query_embedding_dic = pickle.load(handle)
        with open(target_esm_result, 'rb') as handle:
            target_embedding_dic = pickle.load(handle)

        protein_pair_dict = {}
        for protein in query_embedding_dic:
            protein_pair_dict[protein] = []
        
        batch_size = 10000
        batch_count = 0
        protein1_list = []
        protein2_list = []
        if (input_prefilter_result == None):
            for protein1 in tqdm(query_embedding_dic, desc = "query protein list"):
                for protein2 in target_embedding_dic:
                    protein1_list.append(protein1)
                    protein2_list.append(protein2)
                    batch_count += 1
                    if (batch_count == batch_size):
                        threshold_filter(protein1_list, protein2_list, nocos, device)
                        batch_count = 0
                        protein1_list = []
                        protein2_list = []
            if (protein1_list != []):
                threshold_filter(protein1_list, protein2_list, nocos, device) #solve the left
                batch_count = 0
                protein1_list = []
                protein2_list = []
        else:
            prefilter_list = get_prefilter_list(input_prefilter_result)
            logger.info(f"prefilter num = {len(prefilter_list)}")
            for index in range(len(prefilter_list)):
                protein1_list.append(prefilter_list[index][0][0])
                protein2_list.append(prefilter_list[index][0][1])
                batch_count += 1
                if (batch_count == batch_size):
                    threshold_filter(protein1_list, protein2_list, nocos, device)
                    batch_count = 0
                    protein1_list = []
                    protein2_list = []
            if (protein1_list != []):
                threshold_filter(protein1_list, protein2_list, nocos, device) #solve the left
                batch_count = 0
                protein1_list = []
                protein2_list = []
            no_pfam_list = []
            for protein1 in tqdm(query_embedding_dic, desc = "query protein list"):
                if (protein_pair_dict[protein1] == []) or (protein_pair_dict[protein1] == [(protein1, 1)]):
                    no_pfam_list.append(protein1)
                    for protein2 in target_embedding_dic:
                        if (protein1 == protein2):
                            continue
                        protein1_list.append(protein1)
                        protein2_list.append(protein2)
                        batch_count += 1
                        if (batch_count == batch_size):
                            threshold_filter(protein1_list, protein2_list, nocos, device)
                            batch_count = 0
                            protein1_list = []
                            protein2_list = []
            if (protein1_list != []):
                threshold_filter(protein1_list, protein2_list, nocos, device) #solve the left
                batch_count = 0
                protein1_list = []
                protein2_list = []
            # with open('./no_pfam_list.txt', 'w') as handle:
            #     for protein in no_pfam_list:
            #         handle.write(f"{protein}\n")
    
    for query_protein in query_embedding_dic:
        protein_pair_dict[query_protein] = sorted(protein_pair_dict[query_protein], key=lambda x:x[1], reverse=True)
    logger.info(f'Sort end.')

    return protein_pair_dict

def esm_similarity_sort(query_esm_result, target_esm_result, device_id, mode = 'mse'):
    with open(query_esm_result, 'rb') as handle:
        query_embedding_dic = pickle.load(handle)
    with open(target_esm_result, 'rb') as handle:
        target_embedding_dic = pickle.load(handle)
    
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

    esm_example = esm_similarity_filiter()

    with torch.no_grad():
        x0 = []
        x1 = []
        for protein1 in tqdm(query_embedding_dic, desc = "query protein list"):
            for protein2 in target_embedding_dic:
                x0.append(protein1)
                x1.append(protein2)

        protein_pair_dict = {}
        for protein in query_embedding_dic:
            protein_pair_dict[protein] = []

        for index in trange(len(x0), desc = "get esm identity"):
            protein1 = x0[index]
            protein2 = x1[index]
            x1_tensor = query_embedding_dic[protein1]
            x2_tensor = target_embedding_dic[protein2]
            x1_tensor = x1_tensor.to(device)
            x2_tensor = x2_tensor.to(device)
            if (mode == 'mse'):
                predict_score = esm_example.mse_mean_esm_identity_compute(x1_tensor, x2_tensor).item()
            elif (mode == 'cos'):
                predict_score = esm_example.cos_mean_esm_identity_compute(x1_tensor, x2_tensor).item()
            else:
                print(f'Wrong Mode {args.mode}!!!')
                return
            protein_pair_dict[protein1].append((protein2, predict_score))
    
    for protein in query_embedding_dic:
        protein_pair_dict[protein] = sorted(protein_pair_dict[protein], key=lambda x:x[1], reverse=True)
    logger.info(f'Sort end.')
    return protein_pair_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #input
    parser.add_argument('-qer', '--query_esm_result', type=str)
    parser.add_argument('-ter', '--target_esm_result', type=str)

    #sort methods choose
    parser.add_argument('-smp', '--save_model_path', type=str, default=None, help="Esm_ss_predict model path.")
    parser.add_argument('-m', '--mode', type=str, default='cos', help="Mean esm result.")
    parser.add_argument('-ipr','--input_prefilter_result', type=str, default=None, help="Input structurepair list name, sort based on it.")
    parser.add_argument('-nocos', '--ss_predictor_without_cos', action='store_true', help="(Optional) Whether to use ss_predictor * cos")

    #output
    parser.add_argument('-opr','--output_prefilter_result', type=str)

    #parameter
    parser.add_argument('-d', '--device-id', default=[0], nargs='*', help='gpu device list, if only cpu then set it None or empty')
    parser.add_argument('-t', '--threshold', type=float, default=None, help="Threshold. Choose the threshold you want to filter pairs")

    args = parser.parse_args()
    #start
    time_start=time.time()

    #get sorted pairlist
    if (args.save_model_path != None):
        prefilter_result = esm_ss_predict_sort(args.input_prefilter_result, args.threshold, args.ss_predictor_without_cos, args.query_esm_result, args.target_esm_result, args.save_model_path, args.device_id)
    else:
        prefilter_result = esm_similarity_sort(args.query_esm_result, args.target_esm_result, args.device_id, mode=args.mode)
    
    #output prefilter_result
    if (args.output_prefilter_result != None):
        output_prefilter_result = args.output_prefilter_result
    else:
        #get default result_path
        result_path = ''.join([x+'/' for x in args.query_esm_result.split('/')[:-1]]) + 'prefilter_result/'
        os.makedirs(result_path, exist_ok=True)
        #result_path += "main_similarity"
        if (args.input_prefilter_result != None):
            result_path += f"ss_filter"
            if (args.threshold != None):
                result_path += f"_{args.threshold}"
            if (args.ss_predictor_without_cos):
                result_path += f"_without_cos"
        elif (args.save_model_path != None):
            result_path += f"ss_sort"
            if (args.threshold != None):
                result_path += f"_{args.threshold}"
            if (args.ss_predictor_without_cos == False):
                result_path += f"_cos"
        else:
            result_path += f"{args.mode}"
        output_prefilter_result = result_path

    with open(output_prefilter_result, 'w') as f:
        for protein in prefilter_result:
            for pair in prefilter_result[protein]:
                f.write(f"{protein}\t{pair[0]}\t{pair[1]}\n")
    
    time_end=time.time()

    print('Esm embedding generate time cost:', time_end-time_start, 's')