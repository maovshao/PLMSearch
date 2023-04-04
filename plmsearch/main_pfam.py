"""
Created on 2021/10/24
@author liuwei
"""

import os
import argparse
from tqdm import tqdm
from logzero import logger
from plmsearch_util.util import get_family_result, get_clan_result

def get_prefilter_result(query_pfam_result, target_pfam_result):
    logger.info(f"query protein num = {len(query_pfam_result)}")
    logger.info(f"target protein num = {len(target_pfam_result)}")

    protein_pair_score_dict = {}
    for protein in query_pfam_result:
        protein_pair_score_dict[protein] = []

    for query_protein in tqdm(query_pfam_result, desc = "query protein list"):
        for target_protein in target_pfam_result:
            if ((len(query_pfam_result[query_protein])>0) and (len(query_pfam_result[query_protein] & target_pfam_result[target_protein])>0)):
                score = 0
                protein_pair_score_dict[query_protein].append((target_protein, score))
    
    return protein_pair_score_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #input
    parser.add_argument('-qpr', '--query_pfam_result', type=str, default=None)
    parser.add_argument('-tpr', '--target_pfam_result', type=str, default=None)

    #pfam methods choose
    parser.add_argument('-c', '--clan', action='store_true', help="(Optional, based on pfam) Whether to use clan")

    #output
    parser.add_argument('-opr','--output_prefilter_result', type=str)

    args = parser.parse_args()
    
    if ((args.query_pfam_result != None) and (os.path.exists(args.query_pfam_result))):
        if (args.clan == False):
            query_pfam_result = get_family_result(args.query_pfam_result)
            target_pfam_result = get_family_result(args.target_pfam_result)
        else:
            clan_file_path = "./plmsearch_data/Pfam_db/Pfam-A.clans.tsv"
            query_pfam_result = get_clan_result(args.query_pfam_result, clan_file_path)
            target_pfam_result = get_clan_result(args.target_pfam_result, clan_file_path)
    else:
        logger.info(f"Without pfam_cluster or pfam_result is not found in {args.query_pfam_result}")

    prefilter_result = get_prefilter_result(query_pfam_result, target_pfam_result)

    #output prefilter_result
    if (args.output_prefilter_result != None):
        output_prefilter_result = args.output_prefilter_result
    else:
        #get default result_path
        result_path = ''.join([x+'/' for x in args.query_pfam_result.split('/')[:-1]]) + 'prefilter_result/'
        os.makedirs(result_path, exist_ok=True)
        if (args.query_pfam_result != None):
            if args.clan:
                result_path += "pfamclan"
            else:
                result_path += "pfamfamily"
        output_prefilter_result = result_path

    with open(output_prefilter_result, 'w') as f:
        for protein in prefilter_result:
            for pair in prefilter_result[protein]:
                f.write(f"{protein}\t{pair[0]}\t{pair[1]}\n")