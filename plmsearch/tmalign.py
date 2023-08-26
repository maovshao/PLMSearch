"""
Created on 2021/10/24
@author liuwei
"""

import argparse
from tqdm import tqdm
from plmsearch_util.alignment_util import tmalign_util

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #input
    parser.add_argument('-qsd','--query_structure_dir', type=str, default=None)
    parser.add_argument('-tsd','--target_structure_dir', type=str, default=None)
    parser.add_argument('-ipr','--input_search_result', type=str, default=None)
    parser.add_argument('-c', '--cpu_num', type=int, default=56)

    #parameter
    args = parser.parse_args()

    #start
    if (args.input_search_result != None):
        tm_scores, alignments = tmalign_util(args.query_structure_dir, args.target_structure_dir, args.input_search_result, args.cpu_num)
    else:
        print("Nothing to compute!!!")

    #output
    output_tmalign_similarity = args.input_search_result + '_tmalign_similarity'
    output_tmalign_alignment = args.input_search_result + '_tmalign_alignment'
    with open(output_tmalign_similarity, 'w') as f1:
        with open(output_tmalign_alignment, 'w') as f2:
            f2.write(f"Note that ':' denotes aligned residue pairs of d < 5.0 A, '.' denotes other aligned residues.\n\n")
            with open(args.input_search_result, "r") as f:
                pairs = f.readlines()
            for line in tqdm(pairs, desc="tmalign output"):
                protein1, protein2, _ = line.strip().split()
                f1.write(f"{protein1}\t{protein2}\t{tm_scores[protein1][protein2]}\n")
                print(f"\n{protein1}\t{protein2}\t{tm_scores[protein1][protein2]}")
                f2.write(f">{protein1}\t{protein2}\n{alignments[protein1][protein2]}\n\n")
                print(f">{protein1}\t{protein2}\n{alignments[protein1][protein2]}")