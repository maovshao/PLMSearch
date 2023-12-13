import os
import time
import torch
import pickle
import torch.nn as nn
import argparse
from tqdm import tqdm, trange
from logzero import logger
from plmsearch_util.model import plmsearch
from plmsearch_util.util import get_search_list, cos_similarity, euclidean_similarity, tensor_to_list

def plmsearch_search(query_embedding_dic, target_embedding_dic, device, model, search_dict):
    with torch.no_grad():
        query_proteins = list(query_embedding_dic.keys())
        query_embedding = torch.stack([query_embedding_dic[key] for key in query_proteins])
        query_embedding = query_embedding.to(device)

        target_proteins = list(target_embedding_dic.keys())
        target_embedding = torch.stack([target_embedding_dic[key] for key in target_proteins])
        target_embedding = target_embedding.to(device)

        similarity_dict = {}
        for protein in query_proteins:
            similarity_dict[protein] = {}

        cos_matrix = cos_similarity(query_embedding, target_embedding)
        cos_matrix_list = tensor_to_list(cos_matrix)
        
        sim_matrix = model.predict(query_embedding, target_embedding)
        sim_matrix_list = tensor_to_list(sim_matrix)

        for i, query_protein in enumerate(query_proteins):
            for j, target_protein in enumerate(target_proteins):
                similarity_dict[query_protein][target_protein] = cos_matrix_list[i][j] if (cos_matrix_list[i][j]>0.995) else cos_matrix_list[i][j] * sim_matrix_list[i][j]

        protein_pair_dict = {}
        for protein in query_proteins:
            protein_pair_dict[protein] = []

        if (search_dict == None):
            for query_protein in query_proteins:
                for target_protein in similarity_dict[query_protein]:
                    protein_pair_dict[query_protein].append((target_protein, similarity_dict[query_protein][target_protein]))
        else:
            for query_protein in query_proteins:
                for target_protein in search_dict[query_protein]:
                    protein_pair_dict[query_protein].append((target_protein, similarity_dict[query_protein][target_protein]))

            for query_protein in query_proteins:
                if (protein_pair_dict[query_protein] == []) or ((len(protein_pair_dict[query_protein]) == 1) and (protein_pair_dict[query_protein][0][1] >= 0.999)):
                    protein_pair_dict[query_protein] = []
                    for target_protein in target_proteins:
                        protein_pair_dict[query_protein].append((target_protein, similarity_dict[query_protein][target_protein]))

        for query_protein in query_proteins:
            protein_pair_dict[query_protein] = sorted(protein_pair_dict[query_protein], key=lambda x:x[1], reverse=True)

    return protein_pair_dict

def esm_similarity_search(query_embedding_dic, target_embedding_dic, device, mode = 'cos'):    
    protein_pair_dict = {}
    for protein in query_embedding_dic:
        protein_pair_dict[protein] = []

    query_proteins = list(query_embedding_dic.keys())
    query_embedding = torch.stack([query_embedding_dic[key] for key in query_proteins])
    query_embedding = query_embedding.to(device)

    target_proteins = list(target_embedding_dic.keys())
    target_embedding = torch.stack([target_embedding_dic[key] for key in target_proteins])
    target_embedding = target_embedding.to(device)

    if mode == "cos":
        sim_matrix = cos_similarity(query_embedding, target_embedding)
    else:
        sim_matrix = euclidean_similarity(query_embedding, target_embedding)
    sim_matrix_list = tensor_to_list(sim_matrix)
    for i, query_protein in enumerate(query_proteins):
        protein_pair_dict[query_protein] = [(target_proteins[j], sim_matrix_list[i][j]) for j in range(len(sim_matrix_list[i]))]
        protein_pair_dict[query_protein] = sorted(protein_pair_dict[query_protein], key=lambda x:x[1], reverse=True)
    return protein_pair_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #input
    parser.add_argument('-iqe', '--input_query_embedding', type=str)
    parser.add_argument('-ite', '--input_target_embedding', type=str)

    #sort methods choose
    parser.add_argument('-smp', '--save_model_path', type=str, default=None, help="plmsearch model path.")
    parser.add_argument('-m', '--mode', type=str, default='cos')
    parser.add_argument('-isr','--input_search_result', type=str, default=None)

    #output
    parser.add_argument('-osr','--output_search_result', type=str)

    #parameter
    parser.add_argument('-d', '--device-id', default=[0], nargs='*', help='gpu device list, if only cpu then set it None or empty')

    args = parser.parse_args()
    #start
    time_start=time.time()

    with open(args.input_query_embedding, 'rb') as handle:
        query_embedding_dic = pickle.load(handle)
    with open(args.input_target_embedding, 'rb') as handle:
        target_embedding_dic = pickle.load(handle)
    
    time_embedding_load=time.time()
    print('Embedding load time cost:', time_embedding_load-time_start, 's')

    ## set the device
    if (args.device_id == None or args.device_id == []):
        print("None of GPU is selected.")
        device = "cpu"
    else:
        if torch.cuda.is_available()==False:
            print("GPU selected but none of them is available.")
            device = "cpu"
        else:
            print("We have", torch.cuda.device_count(), "GPUs in total!, we will use as you selected")
            device = f'cuda:{args.device_id[0]}'

    if (args.save_model_path != None):
        model = plmsearch(embed_dim = 1280)
        model.load_pretrained(args.save_model_path)
        model.eval()
        model_methods = model
        if (device != "cpu"):
            model = nn.DataParallel(model, device_ids = args.device_id)
            model_methods = model.module
        model.to(device)

    #output search_result
    if (args.output_search_result != None):
        output_search_result = args.output_search_result
    else:
        #get default result_path
        result_path = ''.join([x+'/' for x in args.input_query_embedding.split('/')[:-1]]) + 'search_result/'
        os.makedirs(result_path, exist_ok=True)
        if (args.input_search_result != None):
            result_path += f"plmsearch"
        elif (args.save_model_path != None):
            result_path += f"ss_predictor"
        else:
            result_path += f"{args.mode}"
        output_search_result = result_path
    with open(output_search_result, 'w') as f:
        pass

    batch_size = 50

    search_dict = None
    if (args.input_search_result != None):
        search_list = get_search_list(args.input_search_result)
        logger.info(f"presearch num = {len(search_list)}")
        search_dict = {}
        for query_protein in list(query_embedding_dic.keys()):
            search_dict[query_protein] = []
        for index in range(len(search_list)):
            query_protein = search_list[index][0][0]
            target_protein = search_list[index][0][1]
            search_dict[query_protein].append(target_protein)

    query_keys_all = list(query_embedding_dic.keys())

    for i in trange(0, len(query_keys_all), batch_size, desc="Search query proteins batch by batch"):
        batch_keys = query_keys_all[i:i+batch_size]
        batch_query_embedding_dic = {k: query_embedding_dic[k] for k in batch_keys}

        #get sorted pairlist
        if (args.save_model_path != None):
            search_result = plmsearch_search(batch_query_embedding_dic, target_embedding_dic, device, model_methods, search_dict)
        else:
            search_result = esm_similarity_search(batch_query_embedding_dic, target_embedding_dic, device, mode=args.mode)

        with open(output_search_result, 'a') as f:
            for protein in search_result:
                for pair in search_result[protein]:
                    f.write(f"{protein}\t{pair[0]}\t{pair[1]}\n")
    
    time_end=time.time()

    print('Search time cost:', time_end-time_embedding_load, 's')