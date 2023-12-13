import argparse
from tqdm import tqdm
from plmsearch_util.alignment_util import sequence_align_util

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #input
    parser.add_argument('-qf','--query_fasta', type=str, default=None)
    parser.add_argument('-tf','--target_fasta', type=str, default=None)
    parser.add_argument('-ipr','--input_search_result', type=str, default=None)
    parser.add_argument('-c', '--cpu_num', type=int, default=56)

    #parameter
    args = parser.parse_args()

    #start
    if (args.input_search_result != None):
        score, sequence_alignments = sequence_align_util(args.query_fasta, args.target_fasta, args.input_search_result, args.cpu_num)
    else:
        print("Nothing to compute!!!")

    #output
    output_score = args.input_search_result + '_sequence_identity'
    output_sequence_alignment = args.input_search_result + '_sequence_alignment'
    with open(output_score, 'w') as f1:
        with open(output_sequence_alignment, 'w') as f2:
            with open(args.input_search_result, "r") as f:
                pairs = f.readlines()
            for line in tqdm(pairs, desc="sequence align output"):
                protein1, protein2, _ = line.strip().split()
                f1.write(f"{protein1}\t{protein2}\t{score[protein1][protein2]}\n")
                print(f"\n{protein1}\t{protein2}\t{score[protein1][protein2]}")
                f2.write(f">{protein1}\t{protein2}\n{sequence_alignments[protein1][protein2]}\n\n")
                print(f">{protein1}\t{protein2}\n{sequence_alignments[protein1][protein2]}")