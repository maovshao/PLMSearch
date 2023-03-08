import json
import re
import argparse
import os
import time
from ss_filter_util.util import read_fasta

def txt_to_json(fasta_path, outfile_path):
    with open('./tmp.txt','r') as f1:
        data1=f1.readlines()
        f1.close()
    os.system("rm -rf ./tmp.txt")
    results = []
    for line in data1:
        if line.isspace():
            continue
        data2=line.split()
        m = re.findall(r"\w+", data2[0])
        if m:
             results.append(line)
    
    data=[]
    pattern = r"\|*(\w+)\|*"
    for item in results:
        data.append(re.split('\s+',item))
    for item in data:
        tmp=re.findall(pattern,item[0])
        tmp2=re.findall(r"(\w+)\.",item[5])
        item[0]="".join(tmp)
        item[5]="".join(tmp2)

    protein_list, _ = read_fasta(fasta_path)
    dict={}
    for protein in protein_list:
        dict.setdefault(protein,{})
    for item in data:
        dict.setdefault(item[0],{}).update({"".join(item[5]):""})
    with open(outfile_path,'w') as f1:
        json.dump(dict, f1, indent = 6)
        f1.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--fasta_path',type=str, help='Place where fasta are stored')
    parser.add_argument('-o', '--outfile_path',type=str, help='Place where outfile are placed')
    args = parser.parse_args()
    #start
    time_start=time.time()
    print(time_start)
    print("perl ./ss_filter_data/PfamScan/pfam_scan.pl -fasta " + args.fasta_path +" -dir ./ss_filter_data/Pfam_db -outfile ./tmp.txt")
    os.system("perl ./ss_filter_data/PfamScan/pfam_scan.pl -fasta " + args.fasta_path +" -dir ./ss_filter_data/Pfam_db -outfile ./tmp.txt")
    txt_to_json(args.fasta_path, args.outfile_path)
    #structure pair_list make end
    time_end=time.time()

    print('Pfam local generate time cost', time_end-time_start, 's')