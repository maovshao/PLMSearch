import os
import re
import multiprocessing
from tqdm import tqdm
from Bio import pairwise2 as pw2
from Bio.pairwise2 import format_alignment
from .util import read_fasta

def pairwise_tmalign_util(pdb1_path, pdb2_path):
    cmd = f"TMalign {pdb1_path} {pdb2_path} -a"
    output = os.popen(cmd).read()

    tm_score = float(re.findall(r"TM-score= (\d\.\d+) \(if normalized by average length of chains", output)[0])

    alignment_start = output.find('(":" denotes aligned residue pairs of d < 5.0 A, "." denotes other aligned residues)')
    alignment = output[alignment_start:]
    alignment = alignment.replace('(":" denotes aligned residue pairs of d < 5.0 A, "." denotes other aligned residues)', '').strip()

    return tm_score, alignment

def tmalign_util(query_structure_dir, target_structure_dir, input_search_result, cpu_num=56):
    with open(input_search_result, "r") as f:
        pairs = f.readlines()

    pool = multiprocessing.Pool(processes=cpu_num)

    results = []
    for line in tqdm(pairs, desc="pairwise tmalign"):
        protein1, protein2, _ = line.strip().split()

        query_structure = os.path.join(query_structure_dir, protein1)
        target_structure = os.path.join(target_structure_dir, protein2)

        result = pool.apply_async(pairwise_tmalign_util, args=(query_structure, target_structure))
        results.append((protein1, protein2, result))

    pool.close()
    pool.join()

    tm_scores = {}
    alignments = {}
    for protein1, protein2, result in results:
        tm_score, alignment = result.get()
        tm_scores.setdefault(protein1, {})
        tm_scores[protein1][protein2] = tm_score
        alignments.setdefault(protein1, {})
        alignments[protein1][protein2] = alignment

    return tm_scores, alignments

def pairwise_sequence_align_util(sequence1, sequence2):
    global_align = pw2.align.globalxx(sequence1, sequence2)
    best_sequence_identity = -1
    align = "None"
    for i in global_align:
        sequence_identity = i[2]/(i[4]-i[3])
        if (sequence_identity > best_sequence_identity):
            align = i
            best_sequence_identity = sequence_identity
    return best_sequence_identity, format_alignment(*align)

def sequence_align_util(query_fasta, target_fasta, input_search_result, cpu_num=56):
    with open(input_search_result, "r") as f:
        pairs = f.readlines()

    pool = multiprocessing.Pool(processes=cpu_num)
    _, query_sequence_dict = read_fasta(query_fasta)
    _, target_sequence_dict = read_fasta(target_fasta)

    results = []
    for line in tqdm(pairs, desc="pairwise sequence align"):
        protein1, protein2, _ = line.strip().split()

        query_sequence = query_sequence_dict[protein1]
        target_sequence = target_sequence_dict[protein2]

        result = pool.apply_async(pairwise_sequence_align_util, args=(query_sequence, target_sequence))
        results.append((protein1, protein2, result))

    pool.close()
    pool.join()

    sequence_identity = {}
    sequence_alignments = {}
    for protein1, protein2, result in results:
        identity, alignment = result.get()
        sequence_identity.setdefault(protein1, {})
        sequence_identity[protein1][protein2] = identity
        sequence_alignments.setdefault(protein1, {})
        sequence_alignments[protein1][protein2] = alignment

    return sequence_identity, sequence_alignments
