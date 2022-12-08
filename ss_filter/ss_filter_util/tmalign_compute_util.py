"""
Created on 2021/10/24
@author liuwei
"""

from tqdm import tqdm, trange

def tmalign_compute(structures):
    #print(f"structures:{structures}\n")
    from pytmalign import pytm
    pytm_obj = pytm.Pypytmalign()
    score = pytm_obj.get_score(structures[0], structures[1])
    #print(f"score:{score}")
    return score

def tmalign_compute_with_spark(protein_pair, query_structure_dir, target_structure_dir):
    structure_pair = []
    for i in trange(len(protein_pair)):
        structure1 = query_structure_dir + protein_pair[i][0][0]
        structure2 = target_structure_dir + protein_pair[i][0][1]
        structure_pair.append((structure1, structure2))

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.config("spark.driver.memory", "15g").appName("ss_mat_compute_with_spark").getOrCreate()

    rdd = spark.sparkContext.parallelize(structure_pair, 50).map(lambda x: (x,tmalign_compute(x)))
    from tempfile import NamedTemporaryFile
    tempFile = NamedTemporaryFile(delete=True)
    tempFile.close()
    rdd.saveAsTextFile(tempFile.name)

    protein_pair_dict = {}
    r, c, v = [], [], []
    from glob import glob
    for i in glob(tempFile.name + "/part-*"):
        with open(i) as fp:
            for line in fp:
                pair, score = eval(line.strip('\n'))
                protein1, protein2 = pair
                protein1 = protein1.split('/')[-1]
                protein2 = protein2.split('/')[-1]
                protein_pair_dict.setdefault(protein1,[])
                protein_pair_dict[protein1].append((protein2, score))
    for query_protein in protein_pair_dict:
        protein_pair_dict[query_protein] = sorted(protein_pair_dict[query_protein], key=lambda x:x[1], reverse=True)
    return protein_pair_dict

def tmalign_compute_without_spark(protein_pair, query_structure_dir, target_structure_dir):
    structure_pair = []
    for i in trange(len(protein_pair)):
        structure1 = query_structure_dir + protein_pair[i][0][0]
        structure2 = target_structure_dir + protein_pair[i][0][1]
        structure_pair.append((structure1, structure2))

    protein_pair_dict = {}
    r, c, v = [], [], []
    for pair in tqdm(structure_pair, desc=f"ss_mat compute without spark"):
                score = tmalign_compute(pair)
                protein1, protein2 = pair
                protein1 = protein1.split('/')[-1]
                protein2 = protein2.split('/')[-1]
                protein_pair_dict.setdefault(protein1,[])
                protein_pair_dict[protein1].append((protein2, score))
    for query_protein in protein_pair_dict:
        protein_pair_dict[query_protein] = sorted(protein_pair_dict[query_protein], key=lambda x:x[1], reverse=True)
    return protein_pair_dict