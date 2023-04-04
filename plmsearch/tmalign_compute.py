"""
Created on 2021/10/24
@author liuwei
"""

import time
import argparse
from plmsearch_util.util import get_prefilter_list
from plmsearch_util.tmalign_compute_util import tmalign_compute_with_spark, tmalign_compute_without_spark

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #input
    parser.add_argument('-qsd','--query_structure_dir', type=str, default=None)
    parser.add_argument('-tsd','--target_structure_dir', type=str, default=None)
    parser.add_argument('-ipr','--input_prefilter_result', type=str, default=None)
    parser.add_argument('-s', '--spark', action='store_true', help="(Optional) To use pyspark or not.")

    #parameter
    args = parser.parse_args()

    #start
    time_start=time.time()
    if (args.input_prefilter_result != None):
        prefilter_list = get_prefilter_list(args.input_prefilter_result)
        #compute result
        if args.spark:
            prefilter_result = tmalign_compute_with_spark(prefilter_list, args.query_structure_dir, args.target_structure_dir)
        else:
            prefilter_result = tmalign_compute_without_spark(prefilter_list, args.query_structure_dir, args.target_structure_dir)
    else:
        print("Nothing to compute!!!")

    #output
    output_prefilter_result = args.input_prefilter_result + '_tmalign'
    with open(output_prefilter_result, 'w') as f:
        for protein in prefilter_result:
            for pair in prefilter_result[protein]:
                f.write(f"{protein}\t{pair[0]}\t{pair[1]}\n")

    #end
    time_end=time.time()
    print('Compute total time cost', time_end-time_start, 's')